/**
 * @file src/platform/windows/game_capture/game_source.cpp
 * @brief Host side of in-game capture. See game_source.h and protocol.h.
 *
 * The shared block is writable by the game's user, so every field read from
 * it is bounded and validated here; the frame's dimensions and format come
 * from the texture the host itself opened, never from the block.
 */
#include "game_source.h"
#include "dxgi_symbols.h"
#include "vk_layer_registration.h"

#include "src/config.h"
#include "src/logging.h"
#include "src/platform/windows/display.h"
#include "src/platform/windows/misc.h"
#include "src/utility.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include <d3d11_1.h>
#include <psapi.h>
#include <sddl.h>

#define GAME_CAPTURE_SHADERS_DIR SUNSHINE_ASSETS_DIR "/shaders/directx"

using namespace std::literals;

namespace platf::dxgi {
  // display_vram.cpp
  blob_t compile_pixel_shader(LPCSTR file);
  blob_t compile_vertex_shader(LPCSTR file);
}  // namespace platf::dxgi

namespace platf::dxgi::game_capture {
  namespace gc = ::game_capture;

  namespace {
    // A hooked game that has not published for this long is treated as not
    // presenting (loading screen, stall): desktop capture takes over
    constexpr auto kFrameStaleAfter = 250ms;
    constexpr auto kReapInterval = 1s;
    constexpr std::uint32_t kMaxDimension = 16384;

    // Live source_t instances of this process, by host_lock value
    std::mutex g_instances_lock;
    std::set<std::uint64_t> g_instances;
    std::uint32_t g_next_instance = 0;

    // Whether the host named by a host_lock value still exists
    bool host_alive(std::uint64_t lock) {
      const auto pid = static_cast<DWORD>(lock >> 32);
      if (pid == GetCurrentProcessId()) {
        std::lock_guard lg(g_instances_lock);
        return g_instances.count(lock) != 0;
      }
      // Fails closed: only proof of absence counts as dead
      HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
      if (!process) {
        return GetLastError() != ERROR_INVALID_PARAMETER;  // no such process
      }
      const bool exited = WaitForSingleObject(process, 0) == WAIT_OBJECT_0;
      CloseHandle(process);
      return !exited;
    }

    std::wstring hook_dll_path() {
      wchar_t exe[MAX_PATH];
      const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
      if (n == 0 || n >= MAX_PATH) {
        return {};
      }
      return (std::filesystem::path(exe).parent_path() / L"tools" / L"vibepollo_game_hook.dll").wstring();
    }

    std::string lower(std::string s) {
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      return s;
    }

    // Launchers, their embedded browsers, browsers and overlay tools: full
    // screen now and then, never a game
    constexpr const char *kNotGames[] = {
      "steam.exe", "steamwebhelper.exe", "gameoverlayui.exe", "gameoverlayui64.exe",
      "epicgameslauncher.exe", "epicwebhelper.exe", "battle.net.exe", "agent.exe",
      "eadesktop.exe", "eabackgroundservice.exe", "galaxyclient.exe", "ubisoftconnect.exe", "upc.exe",
      "discord.exe", "chrome.exe", "msedge.exe", "msedgewebview2.exe", "firefox.exe", "brave.exe",
      "obs64.exe", "rtss.exe", "msiafterburner.exe", "nvcontainer.exe", "nvidia app.exe", "nvidia overlay.exe",
    };

    // Never inject into Windows itself, Vibepollo's own processes or known
    // non-games
    bool is_candidate_process(DWORD pid, const std::string &exe) {
      if (pid == 0 || pid == GetCurrentProcessId() || exe.empty()) {
        return false;
      }
      const auto path = lower(exe);
      wchar_t windir_w[MAX_PATH];
      if (GetWindowsDirectoryW(windir_w, MAX_PATH) != 0) {
        const auto windir = lower(std::filesystem::path(windir_w).string());
        if (path.rfind(windir, 0) == 0) {
          return false;
        }
      }
      for (const auto *own : {"sunshine", "vibepollo", "apollo\\tools"}) {
        if (path.find(own) != std::string::npos) {
          return false;
        }
      }
      const auto name = std::filesystem::path(path).filename().string();
      for (const auto *not_game : kNotGames) {
        if (name == not_game) {
          return false;
        }
      }
      return true;
    }

    enum class anti_cheat_e { clear,
                              detected,
                              unknown };

    // Games under an anti-cheat are never injected: the hook would be flagged
    // or blocked. Detected from the modules the process has loaded and from
    // the anti-cheat runtimes shipped next to the executable. An inspection
    // that could not complete is `unknown`, which is treated as detected.
    anti_cheat_e detect_anti_cheat(HANDLE process, const std::string &exe, std::string &which) {
      static constexpr const char *kModules[] = {
        "easyanticheat",  // EAC (EasyAntiCheat.dll, EasyAntiCheat_EOS.dll, ...)
        "beclient",  // BattlEye
        "vgc.dll",  // Vanguard
        "punkbuster",
        "pbcl.dll",
        "faceit",
        "esea",
        "gameguard",
        "nprotect",
        "xigncode",
        "x3.xem",
        "hackshield",
        "vac.dll",
        "denuvo_anti_cheat",
        "ricochet",
        "mhyprot",  // miHoYo
        "zksvc",
        "anticheat",
      };
      std::error_code ec;
      const auto dir = std::filesystem::path(exe).parent_path();
      for (const auto *runtime : {"EasyAntiCheat", "EasyAntiCheat_EOS_Setup.exe", "BattlEye", "EasyAntiCheat.exe"}) {
        if (std::filesystem::exists(dir / runtime, ec)) {
          which = runtime;
          return anti_cheat_e::detected;
        }
      }

      // The module list changes while the game loads; retry until an
      // enumeration fits and is consistent
      std::vector<HMODULE> modules(512);
      bool enumerated = false;
      for (int attempt = 0; attempt < 5 && !enumerated; ++attempt) {
        DWORD needed = 0;
        const DWORD bytes = static_cast<DWORD>(modules.size() * sizeof(HMODULE));
        if (!EnumProcessModulesEx(process, modules.data(), bytes, &needed, LIST_MODULES_ALL)) {
          Sleep(50);
          continue;
        }
        if (needed > bytes) {
          modules.resize(needed / sizeof(HMODULE) + 64);
          continue;
        }
        modules.resize(needed / sizeof(HMODULE));
        enumerated = true;
      }
      if (!enumerated) {
        return anti_cheat_e::unknown;
      }
      for (HMODULE module : modules) {
        char name[MAX_PATH];
        if (GetModuleBaseNameA(process, module, name, MAX_PATH) == 0) {
          return anti_cheat_e::unknown;
        }
        const auto lowered = lower(name);
        for (const auto *pattern : kModules) {
          if (lowered.find(pattern) != std::string::npos) {
            which = name;
            return anti_cheat_e::detected;
          }
        }
      }
      return anti_cheat_e::clear;
    }

    const char *state_name(std::uint32_t state) {
      switch (static_cast<gc::hook_state_e>(state)) {
        case gc::hook_state_e::none:
          return "loading";
        case gc::hook_state_e::hooked:
          return "hooked";
        case gc::hook_state_e::capturing:
          return "capturing";
        case gc::hook_state_e::unsupported:
          return "unsupported";
        case gc::hook_state_e::failed:
          return "failed";
        case gc::hook_state_e::fatal:
          return "fatal";
      }
      return "?";
    }

    // The hook's error text, bounded and terminated here: the block is
    // writable by the game's user, so the field is not trusted to end
    std::string bounded_error(const gc::shared_block_t *block) {
      char buffer[gc::kErrorLength + 1];
      std::memcpy(buffer, block->last_error, gc::kErrorLength);
      buffer[gc::kErrorLength] = '\0';
      std::string text(buffer);
      for (auto &c : text) {
        if (!std::isprint(static_cast<unsigned char>(c))) {
          c = '?';
        }
      }
      return text;
    }

    // Security descriptor for the shared block: SYSTEM and administrators
    // full access, the game's logon session read/write (the logon SID, so
    // other sessions of the same account cannot reach it), and a medium
    // integrity label so the (medium integrity) game may write it at all.
    PSECURITY_DESCRIPTOR make_block_descriptor(HANDLE process) {
      winrt::handle token;
      if (!OpenProcessToken(process, TOKEN_QUERY, token.put())) {
        return nullptr;
      }
      DWORD size = 0;
      GetTokenInformation(token.get(), TokenLogonSid, nullptr, 0, &size);
      std::vector<std::uint8_t> buffer(size);
      if (!GetTokenInformation(token.get(), TokenLogonSid, buffer.data(), size, &size)) {
        return nullptr;
      }
      auto *groups = reinterpret_cast<TOKEN_GROUPS *>(buffer.data());
      if (groups->GroupCount < 1) {
        return nullptr;
      }
      LPWSTR sid_string = nullptr;
      if (!ConvertSidToStringSidW(groups->Groups[0].Sid, &sid_string)) {
        return nullptr;
      }
      std::wstring sddl = L"D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GRGW;;;";
      sddl += sid_string;
      sddl += L")S:(ML;;NW;;;ME)";
      LocalFree(sid_string);
      PSECURITY_DESCRIPTOR sd = nullptr;
      if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &sd, nullptr)) {
        return nullptr;
      }
      return sd;
    }

    bool create_block(HANDLE process, DWORD pid, winrt::handle &mapping, gc::shared_block_t *&block, bool &existed) {
      wchar_t name[96];
      gc::shared_block_name(name, 96, pid);
      PSECURITY_DESCRIPTOR sd = make_block_descriptor(process);
      if (!sd) {
        return false;
      }
      SECURITY_ATTRIBUTES sa {sizeof(sa), sd, FALSE};
      mapping.attach(CreateFileMappingW(INVALID_HANDLE_VALUE, &sa, PAGE_READWRITE, 0, sizeof(gc::shared_block_t), name));
      existed = GetLastError() == ERROR_ALREADY_EXISTS;
      LocalFree(sd);
      if (!mapping) {
        return false;
      }
      block = static_cast<gc::shared_block_t *>(MapViewOfFile(mapping.get(), FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(gc::shared_block_t)));
      if (!block) {
        return false;
      }
      // A block that already exists is a hook still loaded from an earlier
      // Vibepollo run: keep its state, it is live
      if (!existed) {
        std::memset(static_cast<void *>(block), 0, sizeof(gc::shared_block_t));
        block->magic = gc::kMagic;
        block->version = gc::kVersion;
      }
      return block->magic == gc::kMagic && block->version == gc::kVersion;
    }

    // Hands the hook the address of dxgi!CDXGISwapChain::PresentImpl, once the
    // host has resolved it (dxgi_symbols.h); nothing until then
    void publish_dxgi_symbols(gc::shared_block_t *block) {
      if (const auto symbols = gc::dxgi_symbols(); symbols && block->dxgi_present_impl_rva.load(std::memory_order_relaxed) == 0) {
        block->dxgi_timestamp.store(symbols->timestamp, std::memory_order_relaxed);
        block->dxgi_image_size.store(symbols->image_size, std::memory_order_relaxed);
        block->dxgi_present_impl_rva.store(symbols->present_impl_rva, std::memory_order_release);
      }
    }

    bool inject(HANDLE process, const std::wstring &dll, std::string &error) {
      if (GetFileAttributesW(dll.c_str()) == INVALID_FILE_ATTRIBUTES) {
        error = "hook DLL not found";
        return false;
      }
      const SIZE_T bytes = (dll.size() + 1) * sizeof(wchar_t);
      void *remote = VirtualAllocEx(process, nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      if (!remote) {
        error = "VirtualAllocEx failed (" + std::to_string(GetLastError()) + ")";
        return false;
      }
      auto free_remote = util::fail_guard([&]() {
        VirtualFreeEx(process, remote, 0, MEM_RELEASE);
      });
      if (!WriteProcessMemory(process, remote, dll.c_str(), bytes, nullptr)) {
        error = "WriteProcessMemory failed (" + std::to_string(GetLastError()) + ")";
        return false;
      }
      // kernel32 is mapped at the same address in every process of a boot
      auto load_library = reinterpret_cast<LPTHREAD_START_ROUTINE>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW"));
      winrt::handle thread {CreateRemoteThread(process, nullptr, 0, load_library, remote, 0, nullptr)};
      if (!thread) {
        error = "CreateRemoteThread failed (" + std::to_string(GetLastError()) + ")";
        return false;
      }
      if (WaitForSingleObject(thread.get(), 10000) != WAIT_OBJECT_0) {
        error = "LoadLibrary in the game did not return";
        free_remote.disable();  // the thread may still read the path
        return false;
      }
      DWORD module = 0;
      GetExitCodeThread(thread.get(), &module);
      if (module == 0) {
        error = "LoadLibrary in the game failed";
        return false;
      }
      return true;
    }

    // Reads a slot's record consistently (seqlock; the render thread is its
    // only writer). Validated: enum range, timestamps within the last second.
    // A slot record's motion fields, as read (validated by the caller)
    struct slot_motion_t {
      std::uint64_t id = 0;
      std::uint32_t out_width = 0, out_height = 0, count = 0;
      std::uint32_t width[gc::kMotionCandidates] = {}, height[gc::kMotionCandidates] = {};
      float scale_x[gc::kMotionCandidates] = {}, scale_y[gc::kMotionCandidates] = {};
    };

    float float_bits(std::uint32_t u) {
      float f;
      std::memcpy(&f, &u, sizeof(f));
      return f;
    }

    bool read_slot(const gc::slot_record_t &record, std::uint32_t &version, std::uint32_t &generation, gc::color_space_e &color_space, std::uint64_t &frame_id, std::uint64_t &present_qpc, std::uint64_t &gpu_done_qpc, std::uint64_t &release_qpc, slot_motion_t &motion) {
      for (int attempt = 0; attempt < 8; ++attempt) {
        const auto s1 = record.seq.load(std::memory_order_acquire);
        if (s1 & 1u) {
          continue;
        }
        generation = record.generation.load(std::memory_order_relaxed);
        const auto cs = record.color_space.load(std::memory_order_relaxed);
        frame_id = record.frame_id.load(std::memory_order_relaxed);
        present_qpc = record.present_qpc.load(std::memory_order_relaxed);
        gpu_done_qpc = record.gpu_done_qpc.load(std::memory_order_relaxed);
        release_qpc = record.release_qpc.load(std::memory_order_relaxed);
        motion.id = record.motion_id.load(std::memory_order_relaxed);
        motion.out_width = record.motion_out_width.load(std::memory_order_relaxed);
        motion.out_height = record.motion_out_height.load(std::memory_order_relaxed);
        motion.count = record.motion_count.load(std::memory_order_relaxed);
        for (int i = 0; i < gc::kMotionCandidates; ++i) {
          motion.width[i] = record.motion_width[i].load(std::memory_order_relaxed);
          motion.height[i] = record.motion_height[i].load(std::memory_order_relaxed);
          motion.scale_x[i] = float_bits(record.motion_scale_x[i].load(std::memory_order_relaxed));
          motion.scale_y[i] = float_bits(record.motion_scale_y[i].load(std::memory_order_relaxed));
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (record.seq.load(std::memory_order_relaxed) != s1) {
          continue;
        }
        if (cs > static_cast<std::uint32_t>(gc::color_space_e::hdr10)) {
          return false;
        }
        const auto now = static_cast<std::uint64_t>(qpc_counter());
        auto plausible = [&](std::uint64_t qpc) {
          return qpc != 0 && qpc <= now && qpc_time_difference(static_cast<int64_t>(now), static_cast<int64_t>(qpc)) < 1s;
        };
        if (!plausible(present_qpc)) {
          return false;
        }
        if (gpu_done_qpc != 0 && !plausible(gpu_done_qpc)) {
          gpu_done_qpc = 0;
        }
        // The release starts the frame: it precedes its Present
        if (release_qpc != 0 && (!plausible(release_qpc) || release_qpc > present_qpc)) {
          release_qpc = 0;
        }
        version = s1 >> 1;
        color_space = static_cast<gc::color_space_e>(cs);
        return true;
      }
      return false;
    }
  }  // namespace

  // Attachment (open, check, create the block, inject) runs on a worker with
  // its own lifetime: the target only holds a reference and reads the
  // results once `state` says they are complete. A source destroyed while
  // the worker waits on the game's loader simply drops its reference.
  struct source_t::attach_job_t {
    enum class state_e { running,
                         done,
                         failed };

    DWORD pid = 0;
    std::string exe;
    std::wstring dll;

    // Written by the worker, read by the capture thread only after `state`
    // is done or failed
    winrt::handle process;
    winrt::handle mapping;
    gc::shared_block_t *block = nullptr;
    std::string error;
    std::atomic<state_e> state {state_e::running};

    ~attach_job_t() {
      if (block) {
        block->capture_enabled.store(0, std::memory_order_release);
        block->motion_enabled.store(0, std::memory_order_release);
        UnmapViewOfFile(block);
      }
    }

    void run() {
      auto fail = [&](std::string why) {
        error = std::move(why);
        state.store(state_e::failed, std::memory_order_release);
      };
      process.attach(OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ | PROCESS_DUP_HANDLE | SYNCHRONIZE, FALSE, pid));
      if (!process) {
        return fail("cannot open the process (" + std::to_string(GetLastError()) + ")");
      }
      BOOL wow64 = FALSE;
      if (IsWow64Process(process.get(), &wow64) && wow64) {
        return fail("32-bit process");
      }
      std::string anti_cheat;
      switch (detect_anti_cheat(process.get(), exe, anti_cheat)) {
        case anti_cheat_e::detected:
          return fail("anti-cheat present (" + anti_cheat + ")");
        case anti_cheat_e::unknown:
          return fail("could not inspect the process for anti-cheat");
        case anti_cheat_e::clear:
          break;
      }
      bool existed = false;
      if (!create_block(process.get(), pid, mapping, block, existed)) {
        return fail("cannot create the shared block (" + std::to_string(GetLastError()) + ")");
      }
      if (!existed) {
        std::string why;
        publish_dxgi_symbols(block);
        if (!inject(process.get(), dll, why)) {
          return fail("injection failed: " + why);
        }
        BOOST_LOG(info) << "Game capture: injected the hook into " << exe << " (pid " << pid << ')';
      } else {
        BOOST_LOG(info) << "Game capture: reattached to the hook already in " << exe << " (pid " << pid << ')';
      }
      state.store(state_e::done, std::memory_order_release);
    }
  };

  struct source_t::target_t {
    DWORD pid = 0;
    std::string exe;
    std::shared_ptr<attach_job_t> job;
    bool attached = false;  // job results consumed: process/block below are valid
    HANDLE process = nullptr;  // owned by the job
    gc::shared_block_t *block = nullptr;  // owned by the job
    bool given_up = false;
    bool host_busy_logged = false;
    std::uint32_t logged_state = 0xffffffffu;

    winrt::handle frame_event;
    int open_failures = 0;
    std::uint32_t open_failures_generation = 0;
    std::uint32_t opened_generation = 0;  // 0 = none opened (the hook's first generation is 1)
    winrt::com_ptr<ID3D11Texture2D> textures[gc::kSlots];
    winrt::com_ptr<IDXGIKeyedMutex> mutexes[gc::kSlots];
    D3D11_TEXTURE2D_DESC texture_desc {};
    winrt::com_ptr<ID3D11Texture2D> motion_textures[gc::kSlots];  // of the opened generation; empty: none
    winrt::com_ptr<IDXGIKeyedMutex> motion_mutexes[gc::kSlots];  // keyed-mutex sync: each motion texture's own
    D3D11_TEXTURE2D_DESC motion_desc {};
    std::uint32_t motion_logged_generation = 0;
    gc::sync_e sync = gc::sync_e::keyed_mutex;  // of the opened generation

    // sync_e::owner: slots whose owner word we hold until our reads of them
    // (the conversion draw) completed on the GPU
    winrt::com_ptr<ID3D11Query> read_done[gc::kSlots];
    bool held[gc::kSlots] = {};
    bool host_locked = false;  // we hold block->host_lock

    std::uint64_t consumed_latest = 0;
    std::uint64_t consumed_frame_id = 0;

    void give_up(const std::string &why) {
      if (!given_up) {
        BOOST_LOG(info) << "Game capture: not capturing " << exe << " (pid " << pid << "): " << why << "; using desktop capture";
      }
      given_up = true;
      if (block) {
        block->capture_enabled.store(0, std::memory_order_release);
        block->motion_enabled.store(0, std::memory_order_release);
      }
    }

    bool exited() const {
      return attached && process && WaitForSingleObject(process, 0) == WAIT_OBJECT_0;
    }

    // Reads the setup consistently (seqlock)
    struct motion_setup_t {
      std::uint32_t format = 0, width = 0, height = 0;
      std::uint64_t handles[gc::kSlots] = {};
    };

    bool read_setup(std::uint32_t &generation, std::uint64_t handles[gc::kSlots], std::uint32_t &handle_kind, std::uint32_t &sync, LUID &luid, std::uint32_t &width, std::uint32_t &height, motion_setup_t *motion = nullptr) const {
      for (int attempt = 0; attempt < 8; ++attempt) {
        const auto s1 = block->setup_seq.load(std::memory_order_acquire);
        if (s1 & 1u) {
          continue;
        }
        generation = block->setup.generation.load(std::memory_order_relaxed);
        for (int i = 0; i < gc::kSlots; ++i) {
          handles[i] = block->setup.textures[i].load(std::memory_order_relaxed);
        }
        handle_kind = block->setup.handle_kind.load(std::memory_order_relaxed);
        sync = block->setup.sync.load(std::memory_order_relaxed);
        luid.LowPart = block->setup.adapter_luid_low.load(std::memory_order_relaxed);
        luid.HighPart = block->setup.adapter_luid_high.load(std::memory_order_relaxed);
        width = block->setup.width.load(std::memory_order_relaxed);
        height = block->setup.height.load(std::memory_order_relaxed);
        if (motion) {
          motion->format = block->setup.motion_format.load(std::memory_order_relaxed);
          motion->width = block->setup.motion_width.load(std::memory_order_relaxed);
          motion->height = block->setup.motion_height.load(std::memory_order_relaxed);
          for (int i = 0; i < gc::kSlots; ++i) {
            motion->handles[i] = block->setup.motion_textures[i].load(std::memory_order_relaxed);
          }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (block->setup_seq.load(std::memory_order_relaxed) == s1) {
          return true;
        }
      }
      return false;
    }
  };

  source_t::source_t(ID3D11Device *device) {
    gc::start_dxgi_symbol_resolution();
    gc::ensure_vk_layer_registered();
    {
      std::lock_guard lg(g_instances_lock);
      _instance_id = (static_cast<std::uint64_t>(GetCurrentProcessId()) << 32) | ++g_next_instance;
      g_instances.insert(_instance_id);
    }
    device->QueryInterface(__uuidof(ID3D11Device1), _device.put_void());
    device->GetImmediateContext(_context.put());
    if (winrt::com_ptr<IDXGIDevice> dxgi_device; SUCCEEDED(device->QueryInterface(__uuidof(IDXGIDevice), dxgi_device.put_void()))) {
      winrt::com_ptr<IDXGIAdapter> adapter;
      if (SUCCEEDED(dxgi_device->GetAdapter(adapter.put()))) {
        DXGI_ADAPTER_DESC desc {};
        adapter->GetDesc(&desc);
        _adapter_luid = desc.AdapterLuid;
      }
    }
  }

  source_t::~source_t() {
    unlock();
    bool still_reading = false;
    for (auto &[pid, t] : _targets) {
      if (!t->attached || !t->block || !t->host_locked) {
        continue;
      }
      // Stop the hook first: no new copies, and the game runs unpaced
      t->block->capture_enabled.store(0, std::memory_order_release);
      t->block->motion_enabled.store(0, std::memory_order_release);
      t->block->limiter_period_ps.store(0, std::memory_order_release);
      if (!t->exited()) {
        release_reads(*t, true);
      }
      bool holding = false;
      for (bool held : t->held) {
        holding = holding || held;
      }
      // Still reading after the wait (a stuck GPU): keep the lock, and with it
      // our claims, rather than let a successor clear them
      if (!holding) {
        auto mine = _instance_id;
        t->block->host_lock.compare_exchange_strong(mine, 0, std::memory_order_acq_rel);
      }
      still_reading = still_reading || holding;
    }
    // An instance that kept a lock stays registered as alive: its claims
    // outlive it until the process exits, rather than being cleared by a
    // successor while its reads may still run
    if (!still_reading) {
      std::lock_guard lg(g_instances_lock);
      g_instances.erase(_instance_id);
    }
  }

  bool source_t::acquire_host(target_t &t) {
    if (t.host_locked) {
      return true;
    }
    auto current = t.block->host_lock.load(std::memory_order_acquire);
    if (current != 0 && current != _instance_id && host_alive(current)) {
      if (!t.host_busy_logged) {
        t.host_busy_logged = true;
        BOOST_LOG(info) << "Game capture: another host is attached to " << t.exe << "; waiting for it to let go";
      }
      return false;
    }
    if (!t.block->host_lock.compare_exchange_strong(current, _instance_id, std::memory_order_acq_rel)) {
      return false;
    }
    // The previous host (if any) is gone: slots it still claimed are free
    for (int i = 0; i < gc::kSlots; ++i) {
      auto expected = gc::kOwnerHost;
      t.block->owner[i].compare_exchange_strong(expected, gc::kOwnerNone, std::memory_order_acq_rel);
    }
    t.host_locked = true;
    return true;
  }

  source_t::target_t *source_t::current() {
    const auto it = _targets.find(_current_pid);
    return it == _targets.end() ? nullptr : it->second.get();
  }

  // Exited games are dropped whatever the foreground is, so their shared
  // textures (three 4K FP16 frames are ~190 MiB) do not accumulate
  void source_t::reap_exited() {
    const auto now = std::chrono::steady_clock::now();
    if (now - _last_reap < kReapInterval) {
      return;
    }
    _last_reap = now;
    for (auto it = _targets.begin(); it != _targets.end();) {
      auto &t = *it->second;
      // A job that finished while its game was not in the foreground still
      // holds a process handle: adopt it so exit is noticed
      if (!t.attached && t.job && t.job->state.load(std::memory_order_acquire) != attach_job_t::state_e::running && t.job->process) {
        t.process = t.job->process.get();
        t.block = t.job->block;
        t.attached = t.job->state.load(std::memory_order_acquire) == attach_job_t::state_e::done;
        if (!t.attached) {
          t.give_up(t.job->error);
        }
      }
      const bool exited = t.process && WaitForSingleObject(t.process, 0) == WAIT_OBJECT_0;
      if (exited) {
        BOOST_LOG(info) << "Game capture: " << t.exe << " (pid " << t.pid << ") exited";
        if (it->first == _current_pid) {
          unlock();
          _current_pid = 0;
        }
        it = _targets.erase(it);
      } else {
        ++it;
      }
    }
  }

  bool source_t::active(const foreground_app::state_t &foreground) {
    if (!config::video.game_capture || !_device) {
      return false;
    }
    reap_exited();
    const std::uint64_t limiter_period = config::video.game_capture_limiter ? _limiter_period_ps : 0;
    const auto heartbeat = static_cast<std::uint64_t>(qpc_counter());
    for (auto &[pid, t] : _targets) {
      if (t->attached && t->block && t->host_locked) {
        release_reads(*t, false);
        // Every hooked game is paced while we stream, focused or not
        t->block->limiter_period_ps.store(limiter_period, std::memory_order_release);
        t->block->host_heartbeat_qpc.store(heartbeat, std::memory_order_release);
        publish_dxgi_symbols(t->block);  // (a resolution that finished after injection)
      }
    }

    const bool eligible = foreground.valid_window && !foreground.shell_window && foreground.fullscreen_on_capture_display && foreground.foreground_pid != 0;
    const DWORD pid = eligible ? foreground.foreground_pid : 0;
    if (pid != _current_pid) {
      // Stop copying in the process we are leaving; its hook stays loaded
      // for when it comes back into focus
      unlock();
      if (auto *previous = current(); previous && previous->attached && previous->block && previous->host_locked) {
        previous->block->capture_enabled.store(0, std::memory_order_release);
        previous->block->motion_enabled.store(0, std::memory_order_release);
      }
      _current_pid = pid;
    }
    if (!pid) {
      return false;
    }

    auto &slot = _targets[pid];
    if (!slot) {
      slot = std::make_unique<target_t>();
      slot->pid = pid;
      slot->exe = foreground.foreground_exe;
    }
    auto &t = *slot;
    if (t.given_up) {
      return false;
    }

    if (!t.attached) {
      if (!t.job) {
        if (!is_candidate_process(pid, t.exe)) {
          t.give_up("excluded process");
          return false;
        }
        t.job = std::make_shared<attach_job_t>();
        t.job->pid = pid;
        t.job->exe = t.exe;
        t.job->dll = hook_dll_path();
        std::thread([job = t.job]() {
          job->run();
        }).detach();
        return false;
      }
      switch (t.job->state.load(std::memory_order_acquire)) {
        case attach_job_t::state_e::running:
          return false;
        case attach_job_t::state_e::failed:
          t.give_up(t.job->error);
          return false;
        case attach_job_t::state_e::done:
          t.process = t.job->process.get();
          t.block = t.job->block;
          t.attached = true;
          break;
      }
    }
    if (!acquire_host(t)) {
      return false;
    }

    t.block->motion_enabled.store(config::video.nv.motion_hints ? 1 : 0, std::memory_order_release);
    t.block->capture_enabled.store(1, std::memory_order_release);

    const auto state = t.block->hook_state.load(std::memory_order_acquire);
    if (state != t.logged_state) {
      t.logged_state = state;
      const auto error = bounded_error(t.block);
      BOOST_LOG(info) << "Game capture: " << t.exe << " hook " << state_name(state) << (error.empty() ? std::string() : " (" + error + ")");
    }
    if (state == static_cast<std::uint32_t>(gc::hook_state_e::fatal)) {
      // Windows changed under the hook (e.g. DXGI dropped the colour-space
      // getter every frame depends on). Deliberately loud: stop the host
      // rather than stream on with a hook that can no longer be trusted.
      BOOST_LOG(fatal) << "Game capture: " << t.exe << ": " << bounded_error(t.block);
      logging::log_flush();
      std::abort();
    }
    if (state == static_cast<std::uint32_t>(gc::hook_state_e::unsupported) || state == static_cast<std::uint32_t>(gc::hook_state_e::failed)) {
      t.give_up(std::string(state_name(state)) + ": " + bounded_error(t.block));
      return false;
    }
    if (state != static_cast<std::uint32_t>(gc::hook_state_e::capturing)) {
      return false;
    }

    std::uint32_t generation, handle_kind, sync, width, height;
    std::uint64_t handles[gc::kSlots];
    LUID luid;
    if (!t.read_setup(generation, handles, handle_kind, sync, luid, width, height)) {
      return false;
    }
    if (generation != t.opened_generation && !open_generation(t)) {
      return false;
    }

    // Fresh means the hook published recently, by its own clock
    const auto publish_qpc = t.block->publish_qpc.load(std::memory_order_acquire);
    const auto now = static_cast<std::uint64_t>(qpc_counter());
    if (publish_qpc == 0 || publish_qpc > now) {
      return false;
    }
    return qpc_time_difference(static_cast<int64_t>(now), static_cast<int64_t>(publish_qpc)) < kFrameStaleAfter;
  }

  std::string source_t::hook_stats() const {
    const auto it = _targets.find(_current_pid);
    if (it == _targets.end() || !it->second->attached || !it->second->block) {
      return {};
    }
    const auto *b = it->second->block;
    const auto waits = b->limiter_waits.load();
    return "hook presented=" + std::to_string(b->frames_presented.load()) + " published=" + std::to_string(b->frames_published.load()) +
           " skipped=" + std::to_string(b->frames_skipped.load()) + "; limiter waits=" + std::to_string(waits) +
           " late=" + std::to_string(b->limiter_late.load()) + " resets=" + std::to_string(b->limiter_resets.load()) +
           " mean wait=" + std::to_string(waits ? b->limiter_wait_us.load() / waits : 0) + "us";
  }

  bool source_t::still_foreground(const RECT &capture_rect) const {
    if (!_current_pid) {
      return false;
    }
    const HWND foreground = GetForegroundWindow();
    DWORD pid = 0;
    if (!foreground || !GetWindowThreadProcessId(foreground, &pid) || pid != _current_pid) {
      return false;
    }
    RECT window {}, intersection {};
    if (!GetWindowRect(foreground, &window) || !IntersectRect(&intersection, &window, &capture_rect)) {
      return false;
    }
    const auto capture_area = static_cast<long long>(capture_rect.right - capture_rect.left) * (capture_rect.bottom - capture_rect.top);
    const auto covered = static_cast<long long>(intersection.right - intersection.left) * (intersection.bottom - intersection.top);
    return capture_area > 0 && covered * 100 >= capture_area * 90;
  }

  bool source_t::open_generation(target_t &t) {
    std::uint32_t generation, handle_kind, sync, width, height;
    std::uint64_t handles[gc::kSlots];
    LUID luid;
    target_t::motion_setup_t motion_setup;
    if (!t.read_setup(generation, handles, handle_kind, sync, luid, width, height, &motion_setup)) {
      return false;
    }
    if (width == 0 || height == 0) {
      return false;  // retired setup, nothing to open yet
    }
    if (width > kMaxDimension || height > kMaxDimension) {
      t.give_up("the hook reported an implausible frame size");
      return false;
    }
    // The game's device must be on the adapter we encode on: a shared
    // texture cannot cross GPUs
    if (luid.LowPart != _adapter_luid.LowPart || luid.HighPart != _adapter_luid.HighPart) {
      t.give_up("the game renders on a different GPU than the one encoding");
      return false;
    }

    auto duplicate = [&](std::uint64_t value) -> HANDLE {
      HANDLE out = nullptr;
      if (!value || !DuplicateHandle(t.process, reinterpret_cast<HANDLE>(value), GetCurrentProcess(), &out, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
        return nullptr;
      }
      return out;
    };

    if (handle_kind != static_cast<std::uint32_t>(gc::handle_kind_e::nt) && handle_kind != static_cast<std::uint32_t>(gc::handle_kind_e::legacy)) {
      t.give_up("the hook reported an unknown texture handle kind");
      return false;
    }
    if (sync != static_cast<std::uint32_t>(gc::sync_e::keyed_mutex) && sync != static_cast<std::uint32_t>(gc::sync_e::owner)) {
      t.give_up("the hook reported an unknown slot synchronization");
      return false;
    }
    const bool owner_sync = sync == static_cast<std::uint32_t>(gc::sync_e::owner);

    if (!t.frame_event) {
      t.frame_event.attach(duplicate(t.block->frame_event.load(std::memory_order_relaxed)));
      if (!t.frame_event) {
        t.give_up("cannot duplicate the frame event");
        return false;
      }
    }

    // Open into temporaries; commit only when every slot opened and all
    // describe the same, plain, sampleable 2D texture of the announced size
    winrt::com_ptr<ID3D11Texture2D> textures[gc::kSlots];
    winrt::com_ptr<IDXGIKeyedMutex> mutexes[gc::kSlots];
    D3D11_TEXTURE2D_DESC desc {};
    // A recycled handle during a resize storm fails like a bad texture, so a
    // setup is retried a few times before the hook is taken to be wrong;
    // true once it has been retried enough (then the target gave up)
    auto count_failure = [&](const std::string &why) {
      if (t.open_failures_generation != generation) {
        t.open_failures_generation = generation;
        t.open_failures = 0;
      }
      if (++t.open_failures >= 8) {
        t.give_up(why);
        return true;
      }
      return false;
    };
    for (int i = 0; i < gc::kSlots; ++i) {
      HRESULT hr;
      if (handle_kind == static_cast<std::uint32_t>(gc::handle_kind_e::legacy)) {
        // A global share handle: opened as is, nothing to duplicate
        if (!handles[i]) {
          return false;
        }
        hr = _device->OpenSharedResource(reinterpret_cast<HANDLE>(handles[i]), __uuidof(ID3D11Texture2D), textures[i].put_void());
      } else {
        winrt::handle shared {duplicate(handles[i])};
        if (!shared) {
          BOOST_LOG(warning) << "Game capture: cannot duplicate texture handle " << i << " (" << GetLastError() << ')';
          return false;
        }
        hr = _device->OpenSharedResource1(shared.get(), __uuidof(ID3D11Texture2D), textures[i].put_void());
      }
      if (FAILED(hr)) {
        count_failure("cannot open the hook's shared texture " + std::to_string(i) + " [0x" + std::string(util::hex(hr).to_string_view()) + ']');
        return false;
      }
      if (const HRESULT qi = owner_sync ? S_OK : textures[i]->QueryInterface(__uuidof(IDXGIKeyedMutex), mutexes[i].put_void()); FAILED(qi)) {
        BOOST_LOG(warning) << "Game capture: shared texture " << i << " has no keyed mutex [0x" << util::hex(qi).to_string_view() << ']';
        return false;
      }
      D3D11_TEXTURE2D_DESC this_desc {};
      textures[i]->GetDesc(&this_desc);
      // A keyed-mutex texture must be acquired before any use, so it cannot
      // be read under owner words
      const bool keyed = (this_desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX) != 0;
      if (this_desc.MipLevels != 1 || this_desc.ArraySize != 1 || this_desc.SampleDesc.Count != 1 ||
          !(this_desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) || this_desc.Width == 0 || this_desc.Height == 0 || keyed == owner_sync) {
        count_failure("the hook's texture has an unexpected shape");
        return false;
      }
      if (i == 0) {
        desc = this_desc;
      } else if (this_desc.Width != desc.Width || this_desc.Height != desc.Height || this_desc.Format != desc.Format) {
        BOOST_LOG(warning) << "Game capture: shared textures disagree; retrying";
        return false;
      }
    }
    if (desc.Width != width || desc.Height != height) {
      return false;  // setup moved on while opening; the next call retries
    }

    // The generation's motion textures, if it has them: optional (a failure
    // here only costs the frames their vectors)
    winrt::com_ptr<ID3D11Texture2D> motion_textures[gc::kSlots];
    winrt::com_ptr<IDXGIKeyedMutex> motion_mutexes[gc::kSlots];
    D3D11_TEXTURE2D_DESC motion_desc {};
    if (motion_setup.format != 0) {
      std::string why;
      const bool plausible = (motion_setup.format == DXGI_FORMAT_R16G16_FLOAT || motion_setup.format == DXGI_FORMAT_R32G32_FLOAT) &&
                             motion_setup.width > 0 && motion_setup.height > 0 && motion_setup.width <= kMaxDimension && motion_setup.height <= kMaxDimension &&
                             motion_setup.height % gc::kMotionCandidates == 0;
      if (!plausible) {
        why = "implausible motion textures";
      }
      for (int i = 0; i < gc::kSlots && why.empty(); ++i) {
        HRESULT hr;
        if (handle_kind == static_cast<std::uint32_t>(gc::handle_kind_e::legacy)) {
          hr = motion_setup.handles[i] ? _device->OpenSharedResource(reinterpret_cast<HANDLE>(motion_setup.handles[i]), __uuidof(ID3D11Texture2D), motion_textures[i].put_void()) : E_HANDLE;
        } else {
          winrt::handle shared {duplicate(motion_setup.handles[i])};
          hr = shared ? _device->OpenSharedResource1(shared.get(), __uuidof(ID3D11Texture2D), motion_textures[i].put_void()) : E_HANDLE;
        }
        if (FAILED(hr)) {
          why = "cannot open motion texture " + std::to_string(i) + " [0x" + std::string(util::hex(hr).to_string_view()) + ']';
          break;
        }
        D3D11_TEXTURE2D_DESC d {};
        motion_textures[i]->GetDesc(&d);
        // Guarded like the frames: its own keyed mutex, or the slot's owner word
        const bool keyed = (d.MiscFlags & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX) != 0;
        if (d.Width != motion_setup.width || d.Height != motion_setup.height || d.Format != static_cast<DXGI_FORMAT>(motion_setup.format) ||
            d.MipLevels != 1 || d.ArraySize != 1 || d.SampleDesc.Count != 1 || !(d.BindFlags & D3D11_BIND_SHADER_RESOURCE) || keyed == owner_sync ||
            (keyed && FAILED(motion_textures[i]->QueryInterface(__uuidof(IDXGIKeyedMutex), motion_mutexes[i].put_void())))) {
          why = "a motion texture has an unexpected shape";
          break;
        }
        motion_desc = d;
      }
      if (!why.empty()) {
        for (int i = 0; i < gc::kSlots; ++i) {
          motion_textures[i] = nullptr;
          motion_mutexes[i] = nullptr;
        }
        if (t.motion_logged_generation != generation) {
          t.motion_logged_generation = generation;
          BOOST_LOG(warning) << "Game capture: " << why << "; frames go without motion vectors";
        }
      }
    }
    // The setup must not have changed underneath the handles we opened
    std::uint32_t generation2, handle_kind2, sync2, width2, height2;
    std::uint64_t handles2[gc::kSlots];
    LUID luid2;
    if (!t.read_setup(generation2, handles2, handle_kind2, sync2, luid2, width2, height2) || generation2 != generation) {
      return false;
    }

    if (owner_sync) {
      for (int i = 0; i < gc::kSlots; ++i) {
        if (!t.read_done[i]) {
          const D3D11_QUERY_DESC query_desc {D3D11_QUERY_EVENT, 0};
          if (FAILED(_device->CreateQuery(&query_desc, t.read_done[i].put()))) {
            BOOST_LOG(warning) << "Game capture: cannot create a GPU event query";
            return false;
          }
        }
      }
    }

    for (int i = 0; i < gc::kSlots; ++i) {
      t.textures[i] = std::move(textures[i]);
      t.mutexes[i] = std::move(mutexes[i]);
      t.motion_textures[i] = std::move(motion_textures[i]);
      t.motion_mutexes[i] = std::move(motion_mutexes[i]);
    }
    t.motion_desc = motion_desc;
    t.sync = owner_sync ? gc::sync_e::owner : gc::sync_e::keyed_mutex;
    t.texture_desc = desc;
    t.opened_generation = generation;
    t.open_failures = 0;
    BOOST_LOG(info) << "Game capture: " << t.exe << " frames " << desc.Width << 'x' << desc.Height << " format " << desc.Format << " (generation " << generation << ')'
                    << (t.motion_textures[0] ? ", with DLSS motion vectors " + std::to_string(motion_desc.Width) + 'x' + std::to_string(motion_desc.Height) : std::string());
    return true;
  }

  capture_e source_t::wait(std::chrono::milliseconds timeout) {
    auto *t = current();
    if (!t || !t->attached || !t->frame_event) {
      return capture_e::error;
    }
    // The event is only a wake-up: it can be left over from a frame already
    // consumed, so what decides is an unconsumed publication
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::clamp<long long>(timeout.count(), 0, 1000));
    for (;;) {
      if (t->block->latest.load(std::memory_order_acquire) != t->consumed_latest) {
        return capture_e::ok;
      }
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
      if (left.count() <= 0) {
        return capture_e::timeout;
      }
      const DWORD r = WaitForSingleObject(t->frame_event.get(), static_cast<DWORD>(left.count()));
      if (r == WAIT_TIMEOUT) {
        return t->block->latest.load(std::memory_order_acquire) != t->consumed_latest ? capture_e::ok : capture_e::timeout;
      }
      if (r != WAIT_OBJECT_0) {
        return capture_e::error;
      }
    }
  }

  capture_e source_t::lock(frame_t &frame) {
    auto *t = current();
    if (!t || !t->attached || !t->textures[0]) {
      return capture_e::error;
    }

    for (int attempt = 0; attempt < 3; ++attempt) {
      const auto latest = t->block->latest.load(std::memory_order_acquire);
      const auto slot = static_cast<std::uint32_t>(latest >> 32);
      const auto version = static_cast<std::uint32_t>(latest);
      if (latest == 0 || latest == t->consumed_latest || slot >= static_cast<std::uint32_t>(gc::kSlots)) {
        return capture_e::timeout;
      }
      const bool owner_sync = t->sync == gc::sync_e::owner;
      if (owner_sync) {
        // Never waits: the hook holds a slot's word only while submitting
        auto expected = gc::kOwnerNone;
        if (!t->block->owner[slot].compare_exchange_strong(expected, gc::kOwnerHost, std::memory_order_acq_rel)) {
          return capture_e::timeout;
        }
      } else {
        const HRESULT acquired = t->mutexes[slot]->AcquireSync(0, 10);
        if (acquired == static_cast<HRESULT>(WAIT_ABANDONED)) {
          t->mutexes[slot]->ReleaseSync(0);
          BOOST_LOG(warning) << "Game capture: a shared texture's keyed mutex was abandoned; asking the hook for fresh textures";
          t->block->recreate_request.fetch_add(1, std::memory_order_acq_rel);
          return capture_e::timeout;
        }
        if (acquired != S_OK) {
          return capture_e::timeout;
        }
      }
      // Nothing read yet: giving the slot back needs no GPU wait
      auto release = [&]() {
        if (owner_sync) {
          t->block->owner[slot].store(gc::kOwnerNone, std::memory_order_release);
        } else {
          t->mutexes[slot]->ReleaseSync(0);
        }
      };

      // The slot's own record, now that we hold its pixels: it must still
      // carry the published version, and belong to the textures we opened
      std::uint32_t record_version, generation;
      gc::color_space_e color_space;
      std::uint64_t frame_id, present_qpc, gpu_done_qpc, release_qpc;
      slot_motion_t motion;
      const bool valid = read_slot(t->block->slots[slot], record_version, generation, color_space, frame_id, present_qpc, gpu_done_qpc, release_qpc, motion);
      if (!valid || record_version != version) {
        release();
        if (!valid) {
          return capture_e::timeout;
        }
        continue;  // rewritten meanwhile: a newer publication exists
      }
      if (generation != t->opened_generation) {
        release();
        if (!open_generation(*t)) {
          return capture_e::timeout;
        }
        continue;
      }
      // Without a keyed mutex only the hook's fence orders its copy before
      // our read: a frame it could not time is not used
      if (owner_sync && gpu_done_qpc == 0) {
        release();
        t->consumed_latest = latest;
        return capture_e::timeout;
      }
      if (frame_id <= t->consumed_frame_id) {
        release();
        t->consumed_latest = latest;
        return capture_e::timeout;
      }
      // A frame that sat unconsumed (a long pool wait) is not shown late
      if (qpc_time_difference(qpc_counter(), static_cast<int64_t>(present_qpc)) > kFrameStaleAfter) {
        release();
        t->consumed_latest = latest;
        return capture_e::timeout;
      }

      frame.texture = t->textures[slot].get();
      frame.width = t->texture_desc.Width;
      frame.height = t->texture_desc.Height;
      frame.format = t->texture_desc.Format;
      frame.color_space = color_space;
      frame.frame_id = frame_id;
      frame.present_qpc = present_qpc;
      frame.gpu_done_qpc = gpu_done_qpc;
      frame.release_qpc = release_qpc;
      // Its vectors, if the generation has motion textures and the record
      // describes a region inside them
      frame.motion = nullptr;
      frame.motion_id = 0;
      // Its vectors, if the generation has motion textures and the record
      // describes regions inside their candidate rows
      const std::uint32_t rows = t->motion_desc.Height / gc::kMotionCandidates;
      bool motion_ok = motion.id != 0 && t->motion_textures[slot] && motion.count >= 1 && motion.count <= static_cast<std::uint32_t>(gc::kMotionCandidates) &&
                       motion.out_width > 0 && motion.out_height > 0 && motion.out_width <= kMaxDimension && motion.out_height <= kMaxDimension;
      for (std::uint32_t i = 0; motion_ok && i < motion.count; ++i) {
        motion_ok = motion.width[i] > 0 && motion.height[i] > 0 && motion.width[i] <= t->motion_desc.Width && motion.height[i] <= rows &&
                    std::isfinite(motion.scale_x[i]) && std::isfinite(motion.scale_y[i]) && motion.scale_x[i] != 0 && motion.scale_y[i] != 0 &&
                    std::abs(motion.scale_x[i]) < 1e6f && std::abs(motion.scale_y[i]) < 1e6f;
      }
      // (keyed-mutex sync: the motion texture's own mutex, given back by the
      // hook with the slot's; busy means this frame goes without vectors)
      if (motion_ok && !owner_sync) {
        const HRESULT hr = t->motion_mutexes[slot] ? t->motion_mutexes[slot]->AcquireSync(0, 0) : E_FAIL;
        if (hr == static_cast<HRESULT>(WAIT_ABANDONED)) {
          t->motion_mutexes[slot]->ReleaseSync(0);
        }
        if (hr == S_OK) {
          _locked_motion = true;
        } else {
          motion_ok = false;
        }
      }
      frame.motion_count = 0;
      if (motion_ok) {
        frame.motion = t->motion_textures[slot].get();
        frame.motion_id = motion.id;
        frame.motion_out_width = motion.out_width;
        frame.motion_out_height = motion.out_height;
        frame.motion_rows = rows;
        frame.motion_count = static_cast<int>(motion.count);
        for (std::uint32_t i = 0; i < motion.count; ++i) {
          frame.motion_sets[i] = {motion.width[i], motion.height[i], motion.scale_x[i], motion.scale_y[i]};
        }
      }
      t->consumed_latest = latest;
      t->consumed_frame_id = frame_id;
      _locked_slot = static_cast<int>(slot);
      return capture_e::ok;
    }
    return capture_e::timeout;
  }

  void source_t::unlock() {
    if (_locked_slot < 0) {
      return;
    }
    if (auto *t = current(); t && t->sync == gc::sync_e::owner) {
      // Our conversion draw may still be reading the slot: keep its owner
      // word until the GPU passes this point
      _context->End(t->read_done[_locked_slot].get());
      t->held[_locked_slot] = true;
    } else if (t && t->mutexes[_locked_slot]) {
      t->mutexes[_locked_slot]->ReleaseSync(0);
    }
    if (auto *t = current(); _locked_motion && t && t->motion_mutexes[_locked_slot]) {
      t->motion_mutexes[_locked_slot]->ReleaseSync(0);
    }
    _locked_motion = false;
    _locked_slot = -1;
  }

  void source_t::set_frame_rate(double fps) {
    _limiter_period_ps = fps > 0 ? static_cast<std::uint64_t>(std::llround(1e12 / fps)) : 0;
  }

  void source_t::release_reads(target_t &t, bool wait) {
    const auto deadline = std::chrono::steady_clock::now() + 100ms;
    for (int i = 0; i < gc::kSlots; ++i) {
      if (!t.held[i]) {
        continue;
      }
      for (;;) {
        const HRESULT hr = _context->GetData(t.read_done[i].get(), nullptr, 0, 0);  // flushes if the query is still unsubmitted
        // Released only once our reads are known to be over: the query
        // completed, or the device is gone (nothing of ours runs any more).
        // Otherwise the claim stays: the hook skips that slot meanwhile.
        if (hr == S_OK || (FAILED(hr) && FAILED(_device->GetDeviceRemovedReason()))) {
          t.block->owner[i].store(gc::kOwnerNone, std::memory_order_release);
          t.held[i] = false;
          break;
        }
        if (!wait || std::chrono::steady_clock::now() > deadline) {
          break;
        }
        std::this_thread::yield();
      }
    }
  }

  // ---- conversion --------------------------------------------------------

  namespace {
    // Shader modes, see game_capture_ps.hlsl
    enum : int {
      kModeCopy = 0,  ///< values pass through (format change only)
      kModeLinearScale = 1,  ///< linear SDR (sampled from an _SRGB view) -> scRGB
      kModeSrgbToScrgb = 2,  ///< sRGB-encoded SDR -> scRGB
      kModePqToScrgb = 3,  ///< HDR10 PQ BT.2020 -> scRGB
      kModeLinearToSrgb = 4,  ///< linear (from an _SRGB view) -> sRGB-encoded SDR
    };

    struct params_t {
      int mode;
      float sdr_white_scale;
      float pad[2];
    };

    bool is_srgb_format(DXGI_FORMAT f) {
      return f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    }

    bool is_8bit_unorm(DXGI_FORMAT f) {
      return f == DXGI_FORMAT_R8G8B8A8_UNORM || f == DXGI_FORMAT_B8G8R8A8_UNORM || is_srgb_format(f);
    }
  }  // namespace

  bool converter_t::prepare(ID3D11Device *device) {
    if (!_init_attempted && !init(device)) {
      BOOST_LOG(error) << "Game capture: conversion shaders unavailable";
    }
    return _ready;
  }

  bool converter_t::init(ID3D11Device *device) {
    _init_attempted = true;
    auto vs_blob = compile_vertex_shader(GAME_CAPTURE_SHADERS_DIR "/game_capture_vs.hlsl");
    auto ps_blob = compile_pixel_shader(GAME_CAPTURE_SHADERS_DIR "/game_capture_ps.hlsl");
    if (!vs_blob || !ps_blob) {
      return false;
    }
    if (FAILED(device->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, _vs.put())) ||
        FAILED(device->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, _ps.put()))) {
      return false;
    }
    D3D11_BUFFER_DESC bd {};
    bd.ByteWidth = sizeof(params_t);
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(device->CreateBuffer(&bd, nullptr, _params.put()))) {
      return false;
    }
    D3D11_SAMPLER_DESC sd {};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(device->CreateSamplerState(&sd, _sampler.put()))) {
      return false;
    }
    _ready = true;
    return true;
  }

  bool converter_t::convert(ID3D11Device *device, ID3D11DeviceContext *context, const frame_t &frame, ID3D11Texture2D *target_texture, ID3D11RenderTargetView *target_rtv, DXGI_FORMAT target_format, float sdr_white_scale) {
    using cs = gc::color_space_e;
    const auto color_space = frame.color_space;
    if (color_space == cs::unknown) {
      return false;
    }

    int mode = -1;
    bool plain_copy = false;
    if (target_format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
      // HDR (advanced color) desktop: scRGB, as the compositor would emit
      if (color_space == cs::scrgb) {
        plain_copy = frame.format == target_format;
        mode = kModeCopy;
      } else if (color_space == cs::hdr10) {
        mode = kModePqToScrgb;
      } else {
        mode = is_srgb_format(frame.format) ? kModeLinearScale : kModeSrgbToScrgb;
      }
    } else if (is_8bit_unorm(target_format) || target_format == DXGI_FORMAT_R10G10B10A2_UNORM) {
      // SDR desktop: only SDR games map onto it
      if (color_space != cs::srgb) {
        return false;
      }
      plain_copy = frame.format == target_format;
      mode = is_srgb_format(frame.format) ? kModeLinearToSrgb : kModeCopy;
    } else {
      return false;
    }

    if (plain_copy) {
      context->CopyResource(target_texture, frame.texture);
      return true;
    }

    if (!_init_attempted && !init(device)) {
      BOOST_LOG(error) << "Game capture: conversion shaders unavailable";
    }
    if (!_ready) {
      return false;
    }

    if (_srv_texture != frame.texture) {
      _srv = nullptr;
      _srv_texture = nullptr;
      if (FAILED(device->CreateShaderResourceView(frame.texture, nullptr, _srv.put()))) {
        return false;
      }
      _srv_texture = frame.texture;
    }

    const params_t params {mode, sdr_white_scale, {0, 0}};
    context->UpdateSubresource(_params.get(), 0, nullptr, &params, 0, 0);

    D3D11_VIEWPORT viewport {0, 0, static_cast<float>(frame.width), static_cast<float>(frame.height), 0, 1};
    context->OMSetRenderTargets(1, &target_rtv, nullptr);
    context->RSSetViewports(1, &viewport);
    context->IASetInputLayout(nullptr);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(_vs.get(), nullptr, 0);
    context->PSSetShader(_ps.get(), nullptr, 0);
    ID3D11Buffer *cb = _params.get();
    context->PSSetConstantBuffers(0, 1, &cb);
    ID3D11SamplerState *sampler = _sampler.get();
    context->PSSetSamplers(0, 1, &sampler);
    ID3D11ShaderResourceView *srv = _srv.get();
    context->PSSetShaderResources(0, 1, &srv);
    context->Draw(3, 0);

    // Leave nothing bound that another pass could hazard against
    ID3D11ShaderResourceView *null_srv = nullptr;
    context->PSSetShaderResources(0, 1, &null_srv);
    ID3D11RenderTargetView *null_rtv = nullptr;
    context->OMSetRenderTargets(1, &null_rtv, nullptr);
    return true;
  }

  float sdr_white_scale_for_output(const wchar_t *gdi_device_name) {
    UINT32 path_count = 0, mode_count = 0;
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS) {
      return 1.0f;
    }
    std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
    std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
    if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) != ERROR_SUCCESS) {
      return 1.0f;
    }
    // DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL (not in every SDK's headers)
    struct sdr_white_level_t {
      DISPLAYCONFIG_DEVICE_INFO_HEADER header;
      ULONG sdr_white_level;  ///< 1000 = 80 nits
    };
    for (UINT32 i = 0; i < path_count; ++i) {
      DISPLAYCONFIG_SOURCE_DEVICE_NAME source {};
      source.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
      source.header.size = sizeof(source);
      source.header.adapterId = paths[i].sourceInfo.adapterId;
      source.header.id = paths[i].sourceInfo.id;
      if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS || std::wcscmp(source.viewGdiDeviceName, gdi_device_name) != 0) {
        continue;
      }
      sdr_white_level_t white {};
      white.header.type = static_cast<DISPLAYCONFIG_DEVICE_INFO_TYPE>(11);
      white.header.size = sizeof(white);
      white.header.adapterId = paths[i].targetInfo.adapterId;
      white.header.id = paths[i].targetInfo.id;
      if (DisplayConfigGetDeviceInfo(&white.header) == ERROR_SUCCESS && white.sdr_white_level > 0) {
        return static_cast<float>(white.sdr_white_level) / 1000.0f;
      }
    }
    return 1.0f;
  }

}  // namespace platf::dxgi::game_capture
