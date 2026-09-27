/**
 * @file src/platform/windows/game_capture/game_source.cpp
 * @brief Host side of in-game capture. See game_source.h and protocol.h.
 *
 * The shared block is writable by the game's user, so every field read from
 * it is bounded and validated here; the frame's dimensions and format come
 * from the texture the host itself opened, never from the block.
 */
#include "game_source.h"

#include "src/config.h"
#include "src/logging.h"
#include "src/platform/windows/display.h"
#include "src/platform/windows/misc.h"
#include "src/utility.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <filesystem>
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

    // Never inject into Windows itself or into Vibepollo's own processes
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
      return true;
    }

    // Games under an anti-cheat are never injected: the hook would be flagged
    // or blocked. Detected from the modules the process has loaded and from
    // the anti-cheat runtimes shipped next to the executable.
    bool has_anti_cheat(HANDLE process, const std::string &exe, std::string &which) {
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
        "vac.dll",  // just in case a VAC module is mapped in-process
        "denuvo_anti_cheat",
        "ricochet",
        "mhyprot",  // miHoYo
        "zksvc",
        "anticheat",
      };
      HMODULE modules[1024];
      DWORD needed = 0;
      if (EnumProcessModulesEx(process, modules, sizeof(modules), &needed, LIST_MODULES_ALL)) {
        const auto count = std::min<DWORD>(needed / sizeof(HMODULE), 1024);
        for (DWORD i = 0; i < count; ++i) {
          char name[MAX_PATH];
          if (GetModuleBaseNameA(process, modules[i], name, MAX_PATH) == 0) {
            continue;
          }
          const auto module = lower(name);
          for (const auto *pattern : kModules) {
            if (module.find(pattern) != std::string::npos) {
              which = name;
              return true;
            }
          }
        }
      }
      std::error_code ec;
      const auto dir = std::filesystem::path(exe).parent_path();
      for (const auto *runtime : {"EasyAntiCheat", "EasyAntiCheat_EOS_Setup.exe", "BattlEye", "EasyAntiCheat.exe"}) {
        if (std::filesystem::exists(dir / runtime, ec)) {
          which = runtime;
          return true;
        }
      }
      return false;
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
    // full access, the game's own logon user read/write, and a medium
    // integrity label so the (medium integrity) game may write it at all.
    // Nobody else can touch it.
    PSECURITY_DESCRIPTOR make_block_descriptor(HANDLE process) {
      winrt::handle token;
      if (!OpenProcessToken(process, TOKEN_QUERY, token.put())) {
        return nullptr;
      }
      DWORD size = 0;
      GetTokenInformation(token.get(), TokenUser, nullptr, 0, &size);
      std::vector<std::uint8_t> buffer(size);
      if (!GetTokenInformation(token.get(), TokenUser, buffer.data(), size, &size)) {
        return nullptr;
      }
      LPWSTR sid_string = nullptr;
      if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER *>(buffer.data())->User.Sid, &sid_string)) {
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
  }  // namespace

  struct source_t::target_t {
    DWORD pid = 0;
    std::string exe;
    winrt::handle process;
    winrt::handle mapping;
    gc::shared_block_t *block = nullptr;
    bool given_up = false;
    std::uint32_t logged_state = 0xffffffffu;

    // Attachment (open, check, create the block, inject) runs on a worker
    // so a slow LoadLibrary in the game never stalls capture
    enum class attach_e { pending,
                          running,
                          done,
                          failed };
    std::atomic<attach_e> attach {attach_e::pending};
    std::string attach_error;
    std::thread attach_thread;

    winrt::handle frame_event;
    std::uint32_t opened_generation = 0;
    winrt::com_ptr<ID3D11Texture2D> textures[gc::kSlots];
    winrt::com_ptr<IDXGIKeyedMutex> mutexes[gc::kSlots];
    D3D11_TEXTURE2D_DESC texture_desc {};

    std::uint64_t consumed_frame_id = 0;

    ~target_t() {
      if (attach_thread.joinable()) {
        attach_thread.join();
      }
      if (block) {
        block->capture_enabled.store(0, std::memory_order_release);
        UnmapViewOfFile(block);
      }
    }

    void give_up(const std::string &why) {
      if (!given_up) {
        BOOST_LOG(info) << "Game capture: not capturing " << exe << " (pid " << pid << "): " << why << "; using desktop capture";
      }
      given_up = true;
      if (block) {
        block->capture_enabled.store(0, std::memory_order_release);
      }
    }

    bool exited() const {
      return process && WaitForSingleObject(process.get(), 0) == WAIT_OBJECT_0;
    }

    // Reads the latest published frame consistently (seqlock); validated
    bool read_latest(std::uint32_t &slot, std::uint32_t &slot_version, std::uint32_t &generation, gc::color_space_e &color_space, std::uint64_t &frame_id, std::uint64_t &present_qpc, std::uint64_t &gpu_done_qpc) const {
      for (int attempt = 0; attempt < 8; ++attempt) {
        const auto s1 = block->frame_seq.load(std::memory_order_acquire);
        if (s1 & 1u) {
          continue;
        }
        slot = block->slot.load(std::memory_order_relaxed);
        slot_version = block->slot_version.load(std::memory_order_relaxed);
        generation = block->generation.load(std::memory_order_relaxed);
        const auto cs = block->color_space.load(std::memory_order_relaxed);
        frame_id = block->frame_id.load(std::memory_order_relaxed);
        present_qpc = block->present_qpc.load(std::memory_order_relaxed);
        gpu_done_qpc = block->gpu_done_qpc.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (block->frame_seq.load(std::memory_order_relaxed) != s1) {
          continue;
        }
        if (slot >= static_cast<std::uint32_t>(gc::kSlots) || cs > static_cast<std::uint32_t>(gc::color_space_e::hdr10)) {
          return false;
        }
        color_space = static_cast<gc::color_space_e>(cs);
        return true;
      }
      return false;
    }

    // Reads the setup consistently (seqlock)
    bool read_setup(std::uint32_t &generation, std::uint64_t handles[gc::kSlots], LUID &luid, std::uint32_t &width, std::uint32_t &height) const {
      for (int attempt = 0; attempt < 8; ++attempt) {
        const auto s1 = block->setup_seq.load(std::memory_order_acquire);
        if (s1 & 1u) {
          continue;
        }
        generation = block->setup.generation.load(std::memory_order_relaxed);
        for (int i = 0; i < gc::kSlots; ++i) {
          handles[i] = block->setup.textures[i].load(std::memory_order_relaxed);
        }
        luid.LowPart = block->setup.adapter_luid_low.load(std::memory_order_relaxed);
        luid.HighPart = block->setup.adapter_luid_high.load(std::memory_order_relaxed);
        width = block->setup.width.load(std::memory_order_relaxed);
        height = block->setup.height.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (block->setup_seq.load(std::memory_order_relaxed) == s1) {
          return true;
        }
      }
      return false;
    }
  };

  source_t::source_t(ID3D11Device *device) {
    device->QueryInterface(__uuidof(ID3D11Device1), _device.put_void());
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
  }

  source_t::target_t *source_t::current() {
    const auto it = _targets.find(_current_pid);
    return it == _targets.end() ? nullptr : it->second.get();
  }

  // Exited games are dropped whatever the foreground is, so their shared
  // textures (two 4K FP16 frames are ~127 MiB) do not accumulate
  void source_t::reap_exited() {
    const auto now = std::chrono::steady_clock::now();
    if (now - _last_reap < kReapInterval) {
      return;
    }
    _last_reap = now;
    for (auto it = _targets.begin(); it != _targets.end();) {
      const auto &t = *it->second;
      const bool attaching = t.attach.load() == target_t::attach_e::running;
      if (!attaching && t.exited()) {
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

    const bool eligible = foreground.valid_window && !foreground.shell_window && foreground.fullscreen_on_capture_display && foreground.foreground_pid != 0;
    const DWORD pid = eligible ? foreground.foreground_pid : 0;
    if (pid != _current_pid) {
      // Stop copying in the process we are leaving; its hook stays loaded
      // for when it comes back into focus
      unlock();
      if (auto *previous = current(); previous && previous->block) {
        previous->block->capture_enabled.store(0, std::memory_order_release);
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

    switch (t.attach.load(std::memory_order_acquire)) {
      case target_t::attach_e::pending: {
        if (!is_candidate_process(pid, t.exe)) {
          t.give_up("excluded process");
          return false;
        }
        t.attach.store(target_t::attach_e::running, std::memory_order_release);
        t.attach_thread = std::thread([&t, dll = hook_dll_path()]() {
          auto fail = [&](std::string why) {
            t.attach_error = std::move(why);
            t.attach.store(target_t::attach_e::failed, std::memory_order_release);
          };
          t.process.attach(OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ | PROCESS_DUP_HANDLE | SYNCHRONIZE, FALSE, t.pid));
          if (!t.process) {
            return fail("cannot open the process (" + std::to_string(GetLastError()) + ")");
          }
          BOOL wow64 = FALSE;
          if (IsWow64Process(t.process.get(), &wow64) && wow64) {
            return fail("32-bit process");
          }
          std::string anti_cheat;
          if (has_anti_cheat(t.process.get(), t.exe, anti_cheat)) {
            return fail("anti-cheat present (" + anti_cheat + ")");
          }
          bool existed = false;
          if (!create_block(t.process.get(), t.pid, t.mapping, t.block, existed)) {
            return fail("cannot create the shared block (" + std::to_string(GetLastError()) + ")");
          }
          if (!existed) {
            std::string error;
            if (!inject(t.process.get(), dll, error)) {
              return fail("injection failed: " + error);
            }
            BOOST_LOG(info) << "Game capture: injected the hook into " << t.exe << " (pid " << t.pid << ')';
          } else {
            BOOST_LOG(info) << "Game capture: reattached to the hook already in " << t.exe << " (pid " << t.pid << ')';
          }
          t.attach.store(target_t::attach_e::done, std::memory_order_release);
        });
        return false;
      }
      case target_t::attach_e::running:
        return false;
      case target_t::attach_e::failed:
        if (t.attach_thread.joinable()) {
          t.attach_thread.join();
        }
        t.give_up(t.attach_error);
        return false;
      case target_t::attach_e::done:
        if (t.attach_thread.joinable()) {
          t.attach_thread.join();
        }
        break;
    }

    t.block->capture_enabled.store(1, std::memory_order_release);

    const auto state = t.block->hook_state.load(std::memory_order_acquire);
    if (state != t.logged_state) {
      t.logged_state = state;
      const auto error = bounded_error(t.block);
      BOOST_LOG(info) << "Game capture: " << t.exe << " hook " << state_name(state) << (error.empty() ? std::string() : " (" + error + ")");
    }
    if (state == static_cast<std::uint32_t>(gc::hook_state_e::unsupported) || state == static_cast<std::uint32_t>(gc::hook_state_e::failed)) {
      t.give_up(std::string(state_name(state)) + ": " + bounded_error(t.block));
      return false;
    }
    if (state != static_cast<std::uint32_t>(gc::hook_state_e::capturing)) {
      return false;
    }

    std::uint32_t generation, width, height;
    std::uint64_t handles[gc::kSlots];
    LUID luid;
    if (!t.read_setup(generation, handles, luid, width, height)) {
      return false;
    }
    if (generation != t.opened_generation && !open_generation(t)) {
      return false;
    }

    // Fresh means the hook published recently, by its own clock
    const auto publish_qpc = t.block->publish_qpc.load(std::memory_order_acquire);
    if (publish_qpc == 0) {
      return false;
    }
    const auto age = qpc_time_difference(qpc_counter(), static_cast<int64_t>(publish_qpc));
    return age >= std::chrono::nanoseconds::zero() && age < kFrameStaleAfter;
  }

  bool source_t::still_foreground() const {
    if (!_current_pid) {
      return false;
    }
    const HWND foreground = GetForegroundWindow();
    DWORD pid = 0;
    return foreground && GetWindowThreadProcessId(foreground, &pid) && pid == _current_pid;
  }

  bool source_t::open_generation(target_t &t) {
    std::uint32_t generation, width, height;
    std::uint64_t handles[gc::kSlots];
    LUID luid;
    if (!t.read_setup(generation, handles, luid, width, height)) {
      return false;
    }
    if (width == 0 || height == 0 || width > kMaxDimension || height > kMaxDimension) {
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
      if (!value || !DuplicateHandle(t.process.get(), reinterpret_cast<HANDLE>(value), GetCurrentProcess(), &out, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
        return nullptr;
      }
      return out;
    };

    if (!t.frame_event) {
      t.frame_event.attach(duplicate(t.block->frame_event.load(std::memory_order_relaxed)));
      if (!t.frame_event) {
        t.give_up("cannot duplicate the frame event");
        return false;
      }
    }

    // Open into temporaries; commit only when every slot opened and all
    // describe the same texture of the size the setup announced
    winrt::com_ptr<ID3D11Texture2D> textures[gc::kSlots];
    winrt::com_ptr<IDXGIKeyedMutex> mutexes[gc::kSlots];
    D3D11_TEXTURE2D_DESC desc {};
    for (int i = 0; i < gc::kSlots; ++i) {
      winrt::handle shared {duplicate(handles[i])};
      if (!shared) {
        BOOST_LOG(warning) << "Game capture: cannot duplicate texture handle " << i << " (" << GetLastError() << ')';
        return false;
      }
      const HRESULT hr = _device->OpenSharedResource1(shared.get(), __uuidof(ID3D11Texture2D), textures[i].put_void());
      if (FAILED(hr) || FAILED(textures[i]->QueryInterface(__uuidof(IDXGIKeyedMutex), mutexes[i].put_void()))) {
        BOOST_LOG(warning) << "Game capture: cannot open shared texture " << i << " [0x" << util::hex(hr).to_string_view() << ']';
        return false;
      }
      D3D11_TEXTURE2D_DESC this_desc {};
      textures[i]->GetDesc(&this_desc);
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
    // The setup must not have changed underneath the handles we opened
    std::uint32_t generation2, width2, height2;
    std::uint64_t handles2[gc::kSlots];
    LUID luid2;
    if (!t.read_setup(generation2, handles2, luid2, width2, height2) || generation2 != generation) {
      return false;
    }

    for (int i = 0; i < gc::kSlots; ++i) {
      t.textures[i] = std::move(textures[i]);
      t.mutexes[i] = std::move(mutexes[i]);
    }
    t.texture_desc = desc;
    t.opened_generation = generation;
    t.consumed_frame_id = 0;
    BOOST_LOG(info) << "Game capture: " << t.exe << " frames " << desc.Width << 'x' << desc.Height << " format " << desc.Format << " (generation " << generation << ')';
    return true;
  }

  capture_e source_t::wait(std::chrono::milliseconds timeout) {
    auto *t = current();
    if (!t || !t->block || !t->frame_event) {
      return capture_e::error;
    }
    // A frame published while we were busy has already consumed its event
    if (t->block->frame_id.load(std::memory_order_acquire) != t->consumed_frame_id) {
      return capture_e::ok;
    }
    switch (WaitForSingleObject(t->frame_event.get(), static_cast<DWORD>(std::clamp<long long>(timeout.count(), 0, 1000)))) {
      case WAIT_OBJECT_0:
        return capture_e::ok;
      case WAIT_TIMEOUT:
        return capture_e::timeout;
      default:
        return capture_e::error;
    }
  }

  capture_e source_t::lock(frame_t &frame) {
    auto *t = current();
    if (!t || !t->block || !t->textures[0]) {
      return capture_e::error;
    }

    for (int attempt = 0; attempt < 3; ++attempt) {
      std::uint32_t slot, version, generation;
      gc::color_space_e color_space;
      std::uint64_t frame_id, present_qpc, gpu_done_qpc;
      if (!t->read_latest(slot, version, generation, color_space, frame_id, present_qpc, gpu_done_qpc)) {
        return capture_e::timeout;
      }
      if (generation != t->opened_generation) {
        if (!open_generation(*t) || generation != t->opened_generation) {
          return capture_e::timeout;
        }
      }
      if (frame_id == t->consumed_frame_id) {
        return capture_e::timeout;
      }
      if (t->mutexes[slot]->AcquireSync(0, 10) != S_OK) {
        return capture_e::timeout;
      }

      // The pixels in the slot are the frame we read about only if the
      // published (slot, version) is unchanged now that we hold the mutex
      std::uint32_t slot2, version2, generation2;
      gc::color_space_e color_space2;
      std::uint64_t frame_id2, present_qpc2, gpu_done_qpc2;
      if (!t->read_latest(slot2, version2, generation2, color_space2, frame_id2, present_qpc2, gpu_done_qpc2) ||
          slot2 != slot || version2 != version || generation2 != generation) {
        t->mutexes[slot]->ReleaseSync(0);
        continue;
      }

      frame.texture = t->textures[slot].get();
      frame.width = t->texture_desc.Width;
      frame.height = t->texture_desc.Height;
      frame.format = t->texture_desc.Format;
      frame.color_space = color_space2;
      frame.frame_id = frame_id2;
      frame.present_qpc = present_qpc2;
      frame.gpu_done_qpc = gpu_done_qpc2;
      t->consumed_frame_id = frame_id2;
      _locked_slot = static_cast<int>(slot);
      return capture_e::ok;
    }
    return capture_e::timeout;
  }

  void source_t::unlock() {
    if (_locked_slot < 0) {
      return;
    }
    if (auto *t = current(); t && t->mutexes[_locked_slot]) {
      t->mutexes[_locked_slot]->ReleaseSync(0);
    }
    _locked_slot = -1;
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
    if (frame.color_space == cs::unknown) {
      return false;
    }

    int mode = -1;
    bool plain_copy = false;
    if (target_format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
      // HDR (advanced color) desktop: scRGB, as the compositor would emit
      if (frame.color_space == cs::scrgb) {
        plain_copy = frame.format == target_format;
        mode = kModeCopy;
      } else if (frame.color_space == cs::hdr10) {
        mode = kModePqToScrgb;
      } else {
        mode = is_srgb_format(frame.format) ? kModeLinearScale : kModeSrgbToScrgb;
      }
    } else if (is_8bit_unorm(target_format) || target_format == DXGI_FORMAT_R10G10B10A2_UNORM) {
      // SDR desktop: only SDR games map onto it
      if (frame.color_space != cs::srgb) {
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
