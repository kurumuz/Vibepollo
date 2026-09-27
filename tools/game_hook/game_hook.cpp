/**
 * @file tools/game_hook/game_hook.cpp
 * @brief In-game capture hook (vibepollo_game_hook.dll), injected by Vibepollo
 *        into a focused fullscreen game. See
 *        src/platform/windows/game_capture/protocol.h for the protocol.
 *
 * D3D11 and D3D12 swapchains are captured. The DXGI entry points (Present,
 * Present1, SetColorSpace1) are inline-detoured with MinHook at their
 * function addresses, found from a throwaway swapchain: that catches every
 * swapchain in the process whatever its swap effect or creation API, and
 * overlays that keep private vtables (their cached "original" is the function
 * we patched). D3D12 also needs the queue a swapchain presents from, which
 * DXGI does not expose: ID3D12CommandQueue::ExecuteCommandLists is detoured
 * too, and a frame is captured only when that queue is certain (see the D3D12
 * section).
 *
 * Threading, so that nothing here can stall or crash the game:
 *  - All work on the game's D3D objects happens on the game's own thread
 *    inside its Present call, except the completion thread's read of the
 *    fence's completed value, which holds its own COM reference and is
 *    never done for a device created single-threaded (those get no fence).
 *  - One swapchain is captured at a time; a Present on another thread that
 *    finds the capture busy skips (try-lock), never waits on the host. A
 *    single-threaded device is only ever touched from the thread that first
 *    presented it.
 *  - A free slot's keyed mutex is taken with a zero timeout; if the host still
 *    holds it, the frame is skipped. The slot's shared record is rewritten
 *    while that mutex is held, so a host that holds the mutex and reads an
 *    even record is reading the metadata of exactly the pixels it holds.
 *  - Slot ownership is a per-slot atomic word (state | submission ticket).
 *    Every transition is a CAS on the exact word observed, so a stale
 *    completion, a retirement or a reuse can never act on a slot's new
 *    occupant. The completion thread is the only writer of `latest`, and it
 *    does its whole read/check/publish under the capture lock (a normal
 *    acquire; the render thread only ever try-locks and skips), so no
 *    submission is rewritten or retired underneath it.
 *  - Nothing is drained or waited for on the render thread: retired textures
 *    are released on a later Present, once their copies have completed.
 */
#include "src/platform/windows/game_capture/limiter_logic.h"
#include "src/platform/windows/game_capture/protocol.h"

#include <MinHook.h>

#include <windows.h>
#include <d3d11_4.h>
#include <d3d11on12.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

  namespace gc = game_capture;

  using present_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT);
  using present1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain1 *, UINT, UINT, const DXGI_PRESENT_PARAMETERS *);
  using set_color_space1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain3 *, DXGI_COLOR_SPACE_TYPE);
  using resize_buffers_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT, UINT, DXGI_FORMAT, UINT);
  using resize_buffers1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain3 *, UINT, UINT, UINT, DXGI_FORMAT, UINT, const UINT *, IUnknown *const *);

  // IDXGISwapChain vtable slots (IUnknown 0-2, IDXGIObject 3-6, IDXGIDeviceSubObject 7)
  constexpr int kVtPresent = 8;
  constexpr int kVtResizeBuffers = 13;
  constexpr int kVtPresent1 = 22;  // IDXGISwapChain1
  constexpr int kVtSetColorSpace1 = 38;  // IDXGISwapChain3
  constexpr int kVtResizeBuffers1 = 39;  // IDXGISwapChain3

  // Private data key under which a swapchain remembers its colour space
  // {8f4c2a6e-5b1d-4c7a-9e3f-2d6a8b1c4e70}
  constexpr GUID kColorSpaceKey = {0x8f4c2a6e, 0x5b1d, 0x4c7a, {0x9e, 0x3f, 0x2d, 0x6a, 0x8b, 0x1c, 0x4e, 0x70}};
  // D3D12: the queue a swapchain was seen presenting from (an interface), and
  // whether it was ever seen presenting from more than one (a flag)
  // {3b9d7e21-6c4a-4f0e-8a5d-1e7c9b2f4a63}, {5e2a8c14-9b3d-4d71-b6e0-7f4a2c9d1e85}
  constexpr GUID kQueueKey = {0x3b9d7e21, 0x6c4a, 0x4f0e, {0x8a, 0x5d, 0x1e, 0x7c, 0x9b, 0x2f, 0x4a, 0x63}};
  constexpr GUID kQueueAmbiguousKey = {0x5e2a8c14, 0x9b3d, 0x4d71, {0xb6, 0xe0, 0x7f, 0x4a, 0x2c, 0x9d, 0x1e, 0x85}};

  present_fn g_real_present = nullptr;
  present1_fn g_real_present1 = nullptr;
  set_color_space1_fn g_real_set_color_space1 = nullptr;
  resize_buffers_fn g_real_resize_buffers = nullptr;  // both or neither (D3D12 capture needs them)
  resize_buffers1_fn g_real_resize_buffers1 = nullptr;

  gc::shared_block_t *g_block = nullptr;
  thread_local bool t_in_present = false;
  std::atomic<bool> g_color_tracking_lost {false};  // a SetColorSpace1 could not be recorded

  // ---- logging -----------------------------------------------------------

  FILE *g_log = nullptr;
  bool g_log_failed = false;  // under g_log_lock
  SRWLOCK g_log_lock = SRWLOCK_INIT;

  void log(const char *fmt, ...) {
    AcquireSRWLockExclusive(&g_log_lock);
    if (!g_log && !g_log_failed) {
      // %ls: MinGW's libstdc++ selects ISO printf, where %s is a narrow string
      wchar_t dir[MAX_PATH];
      wchar_t path[MAX_PATH + 64];
      const DWORD n = GetTempPathW(MAX_PATH, dir);
      const int len = (n != 0 && n < MAX_PATH) ? std::swprintf(path, sizeof(path) / sizeof(path[0]), L"%lsvibepollo_game_hook_%lu.log", dir, GetCurrentProcessId()) : -1;
      g_log = len > 0 ? _wfopen(path, L"a") : nullptr;
      if (!g_log) {
        g_log_failed = true;  // once: report it where a debugger can see it
        wchar_t msg[MAX_PATH + 128];
        std::swprintf(msg, sizeof(msg) / sizeof(msg[0]), L"vibepollo_game_hook: cannot open the log (temp path %lu, format %d, errno %d)\n", n, len, errno);
        OutputDebugStringW(msg);
      }
    }
    if (g_log) {
      SYSTEMTIME st;
      GetLocalTime(&st);
      std::fprintf(g_log, "[%02d:%02d:%02d.%03d] ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
      va_list ap;
      va_start(ap, fmt);
      std::vfprintf(g_log, fmt, ap);
      va_end(ap);
      std::fputc('\n', g_log);
      std::fflush(g_log);
    }
    ReleaseSRWLockExclusive(&g_log_lock);
  }

  void set_state(gc::hook_state_e state, const char *error = nullptr) {
    if (!g_block) {
      return;
    }
    if (error) {
      std::snprintf(g_block->last_error, sizeof(g_block->last_error), "%s", error);
      log("%s", error);
    }
    g_block->hook_state.store(static_cast<std::uint32_t>(state), std::memory_order_release);
  }

  std::uint64_t qpc_now() {
    LARGE_INTEGER v;
    QueryPerformanceCounter(&v);
    return static_cast<std::uint64_t>(v.QuadPart);
  }

  std::uint64_t qpc_frequency() {
    static const std::uint64_t f = [] {
      LARGE_INTEGER v;
      QueryPerformanceFrequency(&v);
      return static_cast<std::uint64_t>(v.QuadPart);
    }();
    return f;
  }

  template<class T>
  void safe_release(T *&p) {
    if (p) {
      p->Release();
      p = nullptr;
    }
  }

  // ---- slots ---------------------------------------------------------------

  // Slot word: state in the top two bits, submission ticket below. Tickets
  // are unique (render-thread counter), so a CAS on the observed word is a
  // CAS on the exact submission.
  constexpr std::uint64_t kStateShift = 62;
  constexpr std::uint64_t kTicketMask = (1ull << kStateShift) - 1;
  constexpr std::uint64_t st_free = 0, st_pending = 1, st_published = 2;

  constexpr std::uint64_t make_word(std::uint64_t state, std::uint64_t ticket) {
    return (state << kStateShift) | (ticket & kTicketMask);
  }

  constexpr std::uint64_t word_state(std::uint64_t word) {
    return word >> kStateShift;
  }

  // Per-slot submission data. The render thread writes the fields, then
  // stores the pending word (release); it never writes them again until the
  // word has left pending. The completion thread reads the word, the fields,
  // then the word again: unchanged means the fields belong to that ticket.
  struct slot_t {
    std::atomic<std::uint64_t> word {make_word(st_free, 0)};
    std::uint32_t version = 0;  // the shared record's seq >> 1
    std::uint32_t generation = 0;
    std::uint64_t frame_id = 0;
    std::uint64_t fence_value = 0;  // 0 = published at Present time (no fence)
    HANDLE done_event = nullptr;  // for the life of the DLL
  };

  slot_t g_slots[gc::kSlots];
  HANDLE g_frame_event = nullptr;
  HANDLE g_completion_thread = nullptr;
  std::uint64_t g_next_ticket = 1;  // render thread only
  std::atomic<std::uint32_t> g_current_generation {0};  // 0 = nothing set up; first textures are generation 1

  // Completion-thread state
  int g_published_slot = -1;
  std::uint64_t g_published_word = 0;
  std::uint64_t g_last_published_frame_id = 0;

  SRWLOCK g_capture_lock = SRWLOCK_INIT;  // held by the one thread capturing; Presents try-lock and skip
  std::uint64_t fence_completed();  // the captured device's fence (D3D11 or D3D12); UINT64_MAX without one; capture lock held

  // Runs under the capture lock: nothing here can be retired or rewritten
  // by the render thread meanwhile (it skips the frame instead).
  void complete_slot(int slot) {
    auto &s = g_slots[slot];
    const auto word = s.word.load(std::memory_order_acquire);
    if (word_state(word) != st_pending) {
      return;  // not in flight: a stale or already-consumed event
    }
    const auto version = s.version;
    const auto generation = s.generation;
    const auto frame_id = s.frame_id;
    const auto fence_value = s.fence_value;

    // Is this event the copy's real completion? Only the fence can say; an
    // event set early for this slot (a previous occupant's registration
    // firing late) is ignored and the real one publishes. A removed device
    // reports every value complete: nothing of it is trusted.
    if (fence_value) {
      const auto completed = fence_completed();
      if (completed < fence_value) {
        return;
      }
      if (completed == UINT64_MAX) {
        auto lost = word;
        s.word.compare_exchange_strong(lost, make_word(st_free, word), std::memory_order_acq_rel);
        return;
      }
    }

    const auto done_qpc = qpc_now();
    const bool current = generation == g_current_generation.load(std::memory_order_acquire);
    if (!current || frame_id <= g_last_published_frame_id) {
      auto stale = word;
      s.word.compare_exchange_strong(stale, make_word(st_free, word), std::memory_order_acq_rel);
      return;
    }

    // Claim it: pending -> published on exactly this submission
    auto expected = word;
    if (!s.word.compare_exchange_strong(expected, make_word(st_published, word), std::memory_order_acq_rel)) {
      return;  // retired or reused under us
    }
    auto &b = *g_block;
    b.slots[slot].gpu_done_qpc.store(fence_value ? done_qpc : 0, std::memory_order_release);
    b.latest.store(gc::make_latest(static_cast<std::uint32_t>(slot), version), std::memory_order_release);
    b.publish_qpc.store(done_qpc, std::memory_order_release);
    b.frames_published.fetch_add(1, std::memory_order_relaxed);
    g_last_published_frame_id = frame_id;

    // Only now may the previously published slot be reused: published ->
    // free on exactly the submission we published
    if (g_published_slot >= 0 && g_published_slot != slot) {
      auto previous = g_published_word;
      g_slots[g_published_slot].word.compare_exchange_strong(previous, make_word(st_free, previous), std::memory_order_acq_rel);
    }
    g_published_slot = slot;
    g_published_word = make_word(st_published, word);
    SetEvent(g_frame_event);
  }

  DWORD WINAPI completion_thread_main(void *) {
    HANDLE events[gc::kSlots];
    for (int i = 0; i < gc::kSlots; ++i) {
      events[i] = g_slots[i].done_event;
    }
    for (;;) {
      const DWORD r = WaitForMultipleObjects(gc::kSlots, events, FALSE, INFINITE);
      if (r < WAIT_OBJECT_0 || r >= WAIT_OBJECT_0 + gc::kSlots) {
        return 1;
      }
      AcquireSRWLockExclusive(&g_capture_lock);
      complete_slot(static_cast<int>(r - WAIT_OBJECT_0));
      ReleaseSRWLockExclusive(&g_capture_lock);
    }
  }

  // ---- the captured swapchain ---------------------------------------------

  // A retired texture set, kept until its copies have completed (or a second
  // has passed), released on a later Present. Its shared handles stay open
  // one retirement longer than the textures so a host still opening them
  // never meets a recycled handle value.
  struct retired_t {
    IUnknown *textures[gc::kSlots] = {};  // ID3D11Texture2D or ID3D12Resource
    IDXGIKeyedMutex *mutexes[gc::kSlots] = {};
    HANDLE handles[gc::kSlots] = {};
    ID3D11Fence *fence = nullptr;  // the fence its copies were signalled on (a reference; D3D11)
    ID3D12Fence *fence12 = nullptr;  // ... or D3D12 (then nothing is released before it completes)
    ID3D12CommandAllocator *allocators[gc::kSlots] = {};  // D3D12 command storage of a released capture
    ID3D12GraphicsCommandList *lists[gc::kSlots] = {};
    std::uint64_t highest_fence_value = 0;
    std::uint64_t retired_qpc = 0;
    bool textures_released = false;
    bool legacy = false;  // shared through global handles: the textures themselves keep the values valid
    bool in_use = false;
  };


  struct capture_t {
    IDXGISwapChain *swapchain = nullptr;  // identity only, never dereferenced outside its Present
    ID3D11Device *device = nullptr;
    ID3D11DeviceContext *context = nullptr;
    ID3D11DeviceContext4 *context4 = nullptr;  // null without fence support
    ID3D11Fence *fence = nullptr;
    std::uint64_t fence_value = 0;
    std::uint64_t next_frame_id = 0;
    bool single_threaded = false;
    DWORD owner_thread = 0;
    gc::api_e api = gc::api_e::unknown;

    // D3D12: the game's device and presenting queue, our fence, and one
    // command allocator/list per slot (reused only once its copy completed)
    ID3D12Device *device12 = nullptr;
    ID3D12CommandQueue *queue = nullptr;
    ID3D12Fence *fence12 = nullptr;
    ID3D12CommandAllocator *allocators[gc::kSlots] = {};
    ID3D12GraphicsCommandList *lists[gc::kSlots] = {};
    std::uint64_t list_fence[gc::kSlots] = {};  // fence value of each list's last submission
    ID3D12Resource *textures12[gc::kSlots] = {};

    ID3D11Texture2D *textures[gc::kSlots] = {};
    IDXGIKeyedMutex *mutexes[gc::kSlots] = {};
    HANDLE shared_handles[gc::kSlots] = {};  // NT handles (owned); null with legacy sharing
    std::uint64_t legacy_handles[gc::kSlots] = {};  // global (KMT) share handles: not closable, live as long as the texture
    bool owner_sync = false;  // no keyed mutexes: slots are guarded by shared_block_t::owner
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::uint64_t highest_fence_value_used = 0;  // of this texture set

    // Retired sets and command storage, until their copies completed (and,
    // with handles, the host's grace period passed). Never evicted early:
    // D3D12 frees memory the GPU still uses, so an unfinished entry waits.
    std::vector<retired_t> retired;
  };

  capture_t g_cap;
  std::atomic<IDXGISwapChain *> g_captured_swapchain {nullptr};  // g_cap.swapchain, readable without the capture lock

  std::uint64_t fence_completed() {
    if (g_cap.fence12) {
      return g_cap.fence12->GetCompletedValue();
    }
    return g_cap.fence ? g_cap.fence->GetCompletedValue() : UINT64_MAX;
  }

  bool have_textures() {
    return g_cap.textures[0] || g_cap.textures12[0];
  }

  void publish_setup(HWND hwnd, bool valid) {
    auto &b = *g_block;
    const auto seq = b.setup_seq.load(std::memory_order_relaxed);
    b.setup_seq.store(seq + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    b.setup.api.store(static_cast<std::uint32_t>(g_cap.api), std::memory_order_relaxed);
    b.setup.width.store(valid ? g_cap.width : 0, std::memory_order_relaxed);
    b.setup.height.store(valid ? g_cap.height : 0, std::memory_order_relaxed);
    b.setup.format.store(static_cast<std::uint32_t>(valid ? g_cap.format : DXGI_FORMAT_UNKNOWN), std::memory_order_relaxed);
    b.setup.hwnd.store(reinterpret_cast<std::uint64_t>(hwnd), std::memory_order_relaxed);
    for (int i = 0; i < gc::kSlots; ++i) {
      const auto value = g_cap.shared_handles[i] ? reinterpret_cast<std::uint64_t>(g_cap.shared_handles[i]) : g_cap.legacy_handles[i];
      b.setup.textures[i].store(valid ? value : 0, std::memory_order_relaxed);
    }
    b.setup.sync.store(static_cast<std::uint32_t>(g_cap.owner_sync ? gc::sync_e::owner : gc::sync_e::keyed_mutex), std::memory_order_relaxed);
    b.setup.handle_kind.store(static_cast<std::uint32_t>(valid && g_cap.legacy_handles[0] ? gc::handle_kind_e::legacy : gc::handle_kind_e::nt), std::memory_order_relaxed);
    if (valid && g_cap.device12) {
      const LUID luid = g_cap.device12->GetAdapterLuid();
      b.setup.adapter_luid_low.store(luid.LowPart, std::memory_order_relaxed);
      b.setup.adapter_luid_high.store(luid.HighPart, std::memory_order_relaxed);
    } else if (valid && g_cap.device) {
      IDXGIDevice *dxgi_device = nullptr;
      if (SUCCEEDED(g_cap.device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void **>(&dxgi_device)))) {
        IDXGIAdapter *adapter = nullptr;
        if (SUCCEEDED(dxgi_device->GetAdapter(&adapter))) {
          DXGI_ADAPTER_DESC desc {};
          adapter->GetDesc(&desc);
          b.setup.adapter_luid_low.store(desc.AdapterLuid.LowPart, std::memory_order_relaxed);
          b.setup.adapter_luid_high.store(desc.AdapterLuid.HighPart, std::memory_order_relaxed);
          adapter->Release();
        }
        dxgi_device->Release();
      }
    }
    b.setup.generation.store(g_current_generation.load(std::memory_order_acquire), std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    b.setup_seq.store(seq + 2, std::memory_order_release);
  }

  void close_handles(HANDLE *handles) {
    for (int i = 0; i < gc::kSlots; ++i) {
      if (handles[i]) {
        CloseHandle(handles[i]);
        handles[i] = nullptr;
      }
    }
  }

  void release_retired(retired_t &r) {
    for (int i = 0; i < gc::kSlots; ++i) {
      safe_release(r.mutexes[i]);
      safe_release(r.textures[i]);
      safe_release(r.lists[i]);
      safe_release(r.allocators[i]);
    }
    safe_release(r.fence);
    safe_release(r.fence12);
    r.textures_released = true;
  }

  // Releases retired sets whose copies have completed (or that are a second
  // old), and closes the handles of sets retired before the one before.
  // Called on the render thread; never waits.
  void collect_retired(bool force) {
    const auto now = qpc_now();
    for (auto &r : g_cap.retired) {
      if (!r.in_use) {
        continue;
      }
      if (!r.textures_released) {
        // (UINT64_MAX = device removed: nothing more will complete; release)
        const auto completed = r.fence12 ? r.fence12->GetCompletedValue() : r.fence ? r.fence->GetCompletedValue() : UINT64_MAX;
        const bool done = completed >= r.highest_fence_value;  // (UINT64_MAX: device removed)
        // D3D11 keeps what its GPU work uses alive; D3D12 does not, so a
        // D3D12 set waits for its copies however long they take
        const bool old = !r.fence12 && now - r.retired_qpc > qpc_frequency();
        // A global handle is only as valid as its texture: keep the textures
        // for the whole grace period so a host that read the old setup never
        // opens a recycled value
        const bool hold = r.legacy && now - r.retired_qpc <= 2 * qpc_frequency();
        if (((done || old) && !hold) || force) {
          release_retired(r);
        }
      }
      const bool has_handles = r.legacy || std::any_of(std::begin(r.handles), std::end(r.handles), [](HANDLE h) {
                                 return h != nullptr;
                               });
      if (r.textures_released && (!has_handles || now - r.retired_qpc > 2 * qpc_frequency() || force)) {
        close_handles(r.handles);
        r.in_use = false;
      }
    }
  }

  // A free entry of the retirement list (a new one if none is free)
  retired_t &new_retired() {
    for (auto &r : g_cap.retired) {
      if (!r.in_use) {
        r = retired_t {};
        return r;
      }
    }
    return g_cap.retired.emplace_back();
  }

  // Retires the current texture set: one transition, however it was reached.
  // The set stays alive in `retired` until its copies are done; slots are
  // freed here on exactly the submissions they hold (a completion in flight
  // for one of them fails its CAS and does nothing).
  void retire_generation(HWND hwnd) {
    if (!have_textures()) {
      return;
    }
    g_current_generation.fetch_add(1, std::memory_order_acq_rel);
    publish_setup(hwnd, false);

    retired_t *slot = &new_retired();
    slot->in_use = true;
    slot->retired_qpc = qpc_now();
    slot->highest_fence_value = g_cap.highest_fence_value_used;
    if (g_cap.fence) {
      g_cap.fence->AddRef();
      slot->fence = g_cap.fence;
    }
    if (g_cap.fence12) {
      g_cap.fence12->AddRef();
      slot->fence12 = g_cap.fence12;
    }
    for (int i = 0; i < gc::kSlots; ++i) {
      slot->textures[i] = g_cap.textures[i] ? static_cast<IUnknown *>(g_cap.textures[i]) : static_cast<IUnknown *>(g_cap.textures12[i]);
    }
    std::memcpy(slot->mutexes, g_cap.mutexes, sizeof(g_cap.mutexes));
    std::memcpy(slot->handles, g_cap.shared_handles, sizeof(g_cap.shared_handles));
    slot->legacy = g_cap.legacy_handles[0] != 0;
    std::memset(g_cap.textures, 0, sizeof(g_cap.textures));
    std::memset(g_cap.textures12, 0, sizeof(g_cap.textures12));
    std::memset(g_cap.mutexes, 0, sizeof(g_cap.mutexes));
    std::memset(g_cap.shared_handles, 0, sizeof(g_cap.shared_handles));
    std::memset(g_cap.legacy_handles, 0, sizeof(g_cap.legacy_handles));
    g_cap.owner_sync = false;
    g_cap.width = g_cap.height = 0;
    g_cap.format = DXGI_FORMAT_UNKNOWN;
    g_cap.highest_fence_value_used = 0;

    for (auto &s : g_slots) {
      auto word = s.word.load(std::memory_order_acquire);
      while (word_state(word) != st_free) {
        if (s.word.compare_exchange_weak(word, make_word(st_free, word), std::memory_order_acq_rel)) {
          break;
        }
      }
    }
  }

  // Moves the D3D12 lists and allocators into a retired entry of their own,
  // released once the fence passes `last_used` (collect_retired)
  void retire_command_storage(std::uint64_t last_used) {
    retired_t *slot = &new_retired();
    slot->in_use = true;
    slot->retired_qpc = qpc_now();
    slot->highest_fence_value = last_used;
    g_cap.fence12->AddRef();
    slot->fence12 = g_cap.fence12;
    for (int i = 0; i < gc::kSlots; ++i) {
      slot->lists[i] = g_cap.lists[i];
      slot->allocators[i] = g_cap.allocators[i];
      g_cap.lists[i] = nullptr;
      g_cap.allocators[i] = nullptr;
    }
  }

  void release_capture(HWND hwnd) {
    retire_generation(hwnd);
    collect_retired(false);
    safe_release(g_cap.fence);
    safe_release(g_cap.context4);
    safe_release(g_cap.context);
    safe_release(g_cap.device);
    // D3D12: an allocator's memory must outlive the GPU's use of it, so the
    // lists and allocators are retired with the fence and released only once
    // their last copies completed
    std::uint64_t last_used = 0;
    for (int i = 0; i < gc::kSlots; ++i) {
      last_used = std::max(last_used, g_cap.list_fence[i]);
    }
    if (g_cap.fence12 && last_used > g_cap.fence12->GetCompletedValue()) {
      retire_command_storage(last_used);
    }
    for (int i = 0; i < gc::kSlots; ++i) {
      safe_release(g_cap.lists[i]);
      safe_release(g_cap.allocators[i]);
      g_cap.list_fence[i] = 0;
    }
    safe_release(g_cap.fence12);
    safe_release(g_cap.queue);
    safe_release(g_cap.device12);
    g_cap.api = gc::api_e::unknown;
    g_cap.swapchain = nullptr;
    g_captured_swapchain.store(nullptr, std::memory_order_release);
    g_cap.owner_thread = 0;
    g_cap.single_threaded = false;
  }

  bool ensure_device(IDXGISwapChain *swapchain, ID3D11Device *device, HWND hwnd) {
    if (g_cap.device == device && g_cap.swapchain == swapchain) {
      return true;
    }
    release_capture(hwnd);
    g_cap.api = gc::api_e::d3d11;
    g_cap.swapchain = swapchain;
    g_captured_swapchain.store(swapchain, std::memory_order_release);
    device->AddRef();
    g_cap.device = device;
    device->GetImmediateContext(&g_cap.context);
    g_cap.owner_thread = GetCurrentThreadId();
    g_cap.single_threaded = (device->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED) != 0;

    // A fence tells us when the GPU finished each copy. The completion
    // thread checks its completed value through a reference of its own, so
    // a device created single-threaded gets no fence (frames publish at
    // Present time), as do runtimes without fences.
    if (!g_cap.single_threaded) {
      ID3D11Device5 *device5 = nullptr;
      if (SUCCEEDED(device->QueryInterface(__uuidof(ID3D11Device5), reinterpret_cast<void **>(&device5)))) {
        if (FAILED(g_cap.context->QueryInterface(__uuidof(ID3D11DeviceContext4), reinterpret_cast<void **>(&g_cap.context4))) ||
            FAILED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, __uuidof(ID3D11Fence), reinterpret_cast<void **>(&g_cap.fence)))) {
          safe_release(g_cap.fence);
          safe_release(g_cap.context4);
        }
        device5->Release();
      }
    }
    g_cap.fence_value = 0;
    log("Device %p (%s): frames timestamped at %s", static_cast<void *>(device), g_cap.single_threaded ? "single-threaded" : "multithreaded",
        g_cap.fence ? "GPU completion" : "Present");
    return true;
  }

  DXGI_FORMAT shareable_format(DXGI_FORMAT format) {
    switch (format) {
      case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
      case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
      case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
      case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
      default:
        return format;
    }
  }

  // The documented defaults: an FP16 swapchain is scRGB, 8-bit is sRGB, 10-bit
  // is sRGB unless SetColorSpace1 made it HDR10 (which we may have missed if
  // injected late), so it is unknown until we see that call.
  std::uint32_t default_color_space(DXGI_FORMAT format) {
    switch (format) {
      case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return static_cast<std::uint32_t>(gc::color_space_e::scrgb);
      case DXGI_FORMAT_R8G8B8A8_UNORM:
      case DXGI_FORMAT_B8G8R8A8_UNORM:
      case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
      case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return static_cast<std::uint32_t>(gc::color_space_e::srgb);
      default:
        return static_cast<std::uint32_t>(gc::color_space_e::unknown);
    }
  }

  // The colour space SetColorSpace1 last set on this swapchain, stored as
  // its private data so it lives and dies with the object
  std::uint32_t swapchain_color_space(IDXGISwapChain *swapchain, DXGI_FORMAT format) {
    // A change we could not record may have made a 10-bit chain HDR10 (or
    // back): its stored value can be stale, so it is unknown. Formats with a
    // fixed encoding are unaffected.
    if (g_color_tracking_lost.load(std::memory_order_acquire) && format == DXGI_FORMAT_R10G10B10A2_UNORM) {
      return static_cast<std::uint32_t>(gc::color_space_e::unknown);
    }
    std::uint32_t stored = 0;
    UINT size = sizeof(stored);
    if (SUCCEEDED(swapchain->GetPrivateData(kColorSpaceKey, &size, &stored)) && size == sizeof(stored)) {
      return stored;
    }
    return default_color_space(format);
  }

  bool ensure_textures(const D3D11_TEXTURE2D_DESC &back, HWND hwnd) {
    const DXGI_FORMAT format = shareable_format(back.Format);
    if (g_cap.textures[0] && g_cap.width == back.Width && g_cap.height == back.Height && g_cap.format == format) {
      return true;
    }
    retire_generation(hwnd);

    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = back.Width;
    desc.Height = back.Height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;

    // Preferred: NT handles (duplicated by the host, never globally
    // guessable) with keyed mutexes. Some devices refuse NT handles, so fall
    // back to the legacy global handles; some refuse keyed mutexes entirely
    // (NieR:Automata), so fall back to plain shared textures guarded by the
    // owner words, which needs the fence (publish only after the copy
    // completed). The hook only copies into these, so the render-target
    // bind is optional.
    struct variant_t {
      bool nt;
      bool keyed;
      UINT bind;
      const char *name;
    };
    constexpr variant_t variants[] = {
      {true, true, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, "nt/rtv"},
      {true, true, D3D11_BIND_SHADER_RESOURCE, "nt"},
      {false, true, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, "kmt/rtv"},
      {false, true, D3D11_BIND_SHADER_RESOURCE, "kmt"},
      {true, false, D3D11_BIND_SHADER_RESOURCE, "nt/nokm"},
      {false, false, D3D11_BIND_SHADER_RESOURCE, "kmt/nokm"},
    };
    const bool have_fence = g_cap.fence && g_cap.context4;

    auto release_partial = [] {
      for (int j = 0; j < gc::kSlots; ++j) {
        safe_release(g_cap.mutexes[j]);
        safe_release(g_cap.textures[j]);
      }
      close_handles(g_cap.shared_handles);
      std::memset(g_cap.legacy_handles, 0, sizeof(g_cap.legacy_handles));
    };

    char failures[gc::kErrorLength] = {};
    const variant_t *used = nullptr;
    for (const auto &v : variants) {
      if (!v.keyed && !have_fence) {
        continue;
      }
      desc.BindFlags = v.bind;
      desc.MiscFlags = (v.keyed ? D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX : D3D11_RESOURCE_MISC_SHARED) | (v.nt ? D3D11_RESOURCE_MISC_SHARED_NTHANDLE : 0);
      const char *step = nullptr;
      HRESULT hr = S_OK;
      int failed_slot = 0;
      for (int i = 0; i < gc::kSlots && SUCCEEDED(hr); ++i) {
        failed_slot = i;
        step = "create";
        hr = g_cap.device->CreateTexture2D(&desc, nullptr, &g_cap.textures[i]);
        if (SUCCEEDED(hr) && v.keyed) {
          step = "keyedmutex";
          hr = g_cap.textures[i]->QueryInterface(__uuidof(IDXGIKeyedMutex), reinterpret_cast<void **>(&g_cap.mutexes[i]));
        }
        if (SUCCEEDED(hr) && v.nt) {
          step = "qi-res1";
          IDXGIResource1 *resource = nullptr;
          hr = g_cap.textures[i]->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void **>(&resource));
          if (SUCCEEDED(hr)) {
            step = "nthandle";
            hr = resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &g_cap.shared_handles[i]);
            resource->Release();
          }
        } else if (SUCCEEDED(hr)) {
          step = "qi-res";
          IDXGIResource *resource = nullptr;
          hr = g_cap.textures[i]->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void **>(&resource));
          HANDLE handle = nullptr;
          if (SUCCEEDED(hr)) {
            step = "kmthandle";
            hr = resource->GetSharedHandle(&handle);
            resource->Release();
          }
          if (SUCCEEDED(hr) && !handle) {
            hr = E_FAIL;
          }
          g_cap.legacy_handles[i] = reinterpret_cast<std::uint64_t>(handle);
        }
      }
      if (SUCCEEDED(hr)) {
        used = &v;
        g_cap.owner_sync = !v.keyed;
        break;
      }
      log("Capture texture variant %s (%ux%u fmt %d) failed at %s of slot %d: 0x%08lx", v.name, desc.Width, desc.Height, desc.Format, step, failed_slot, hr);
      const auto len = std::strlen(failures);
      std::snprintf(failures + len, sizeof(failures) - len, "%s%s@%s%d=%lx", len ? " " : "", v.name, step, failed_slot, hr);
      release_partial();
    }
    if (!used) {
      // Narrow down what this device rejects (log only)
      auto probe = [&](const char *name, UINT bind, UINT misc) {
        D3D11_TEXTURE2D_DESC d = desc;
        d.BindFlags = bind;
        d.MiscFlags = misc;
        ID3D11Texture2D *t = nullptr;
        const HRESULT hr = g_cap.device->CreateTexture2D(&d, nullptr, &t);
        safe_release(t);
        log("Probe %s: 0x%08lx", name, hr);
      };
      probe("plain srv", D3D11_BIND_SHADER_RESOURCE, 0);
      probe("plain none", 0, 0);
      probe("shared srv", D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_MISC_SHARED);
      probe("shared srv/rtv", D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, D3D11_RESOURCE_MISC_SHARED);
      UINT support = 0;
      const HRESULT fs = g_cap.device->CheckFormatSupport(format, &support);
      D3D11_FEATURE_DATA_FORMAT_SUPPORT2 support2 {format, 0};
      const HRESULT fs2 = g_cap.device->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &support2, sizeof(support2));
      IUnknown *on12 = nullptr;
      const HRESULT on12_hr = g_cap.device->QueryInterface(__uuidof(ID3D11On12Device), reinterpret_cast<void **>(&on12));
      safe_release(on12);
      log("Format %d support 0x%08x (0x%08lx) support2 0x%08x (0x%08lx, shareable %d); 11on12 0x%08lx; back buffer bind 0x%x misc 0x%x usage %d",
          format, support, fs, support2.OutFormatSupport2, fs2, (support2.OutFormatSupport2 & D3D11_FORMAT_SUPPORT2_SHAREABLE) != 0,
          on12_hr, back.BindFlags, back.MiscFlags, back.Usage);

      char msg[gc::kErrorLength];
      std::snprintf(msg, sizeof(msg), "Textures %ux%u f%d FL%x cf%x: %s", desc.Width, desc.Height, desc.Format, g_cap.device->GetFeatureLevel(), g_cap.device->GetCreationFlags(), failures);
      set_state(gc::hook_state_e::failed, msg);
      return false;
    }
    if (used != &variants[0]) {
      log("Capture textures use %s sharing%s (device FL %x, flags %x)", used->name, g_cap.owner_sync ? " without keyed mutexes (owner words)" : "", g_cap.device->GetFeatureLevel(), g_cap.device->GetCreationFlags());
    }

    g_cap.width = back.Width;
    g_cap.height = back.Height;
    g_cap.format = format;
    if (g_current_generation.load(std::memory_order_acquire) == 0) {
      g_current_generation.store(1, std::memory_order_release);  // 0 means "nothing set up" to the host
    }
    publish_setup(hwnd, true);
    log("Capture textures %ux%u format %d, generation %u", back.Width, back.Height, format, g_current_generation.load());
    return true;
  }

  bool foreground_window(IDXGISwapChain *swapchain, HWND &hwnd) {
    DXGI_SWAP_CHAIN_DESC desc {};
    if (FAILED(swapchain->GetDesc(&desc)) || !desc.OutputWindow) {
      return false;
    }
    hwnd = desc.OutputWindow;
    const HWND foreground = GetForegroundWindow();
    return hwnd == foreground || GetAncestor(hwnd, GA_ROOT) == foreground;
  }

  // Rewrites a slot's shared record while this thread owns the slot (its
  // keyed mutex or owner word), before the copy: a host that holds the slot
  // and reads an even record reads the metadata of exactly its pixels.
  // Returns the record's version.
  std::uint32_t write_slot_record(int slot, std::uint32_t generation, std::uint32_t color_space, std::uint64_t frame_id,
                                  std::uint64_t present_qpc, std::uint64_t release_qpc) {
    auto &record = g_block->slots[slot];
    const auto seq = record.seq.load(std::memory_order_relaxed);
    record.seq.store(seq + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    record.generation.store(generation, std::memory_order_relaxed);
    record.color_space.store(color_space, std::memory_order_relaxed);
    record.frame_id.store(frame_id, std::memory_order_relaxed);
    record.present_qpc.store(present_qpc, std::memory_order_relaxed);
    record.release_qpc.store(release_qpc, std::memory_order_relaxed);
    record.gpu_done_qpc.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    record.seq.store(seq + 2, std::memory_order_release);
    return (seq + 2) >> 1;
  }

  // Hands a submitted copy to the completion thread: the fields first, then
  // the pending word (release). Returns its ticket.
  std::uint64_t mark_pending(int slot, std::uint32_t version, std::uint32_t generation, std::uint64_t frame_id, std::uint64_t fence_value) {
    auto &s = g_slots[slot];
    s.version = version;
    s.generation = generation;
    s.frame_id = frame_id;
    s.fence_value = fence_value;
    const auto ticket = g_next_ticket++;
    s.word.store(make_word(st_pending, ticket), std::memory_order_release);
    return ticket;
  }

  // No completion will come for this submission: free its slot rather than
  // leave it pending forever
  void drop_pending(int slot, std::uint64_t ticket) {
    auto pending = make_word(st_pending, ticket);
    g_slots[slot].word.compare_exchange_strong(pending, make_word(st_free, ticket), std::memory_order_acq_rel);
    g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
  }

  void note_capturing(IDXGISwapChain *swapchain, const char *api) {
    if (g_block->hook_state.load(std::memory_order_relaxed) != static_cast<std::uint32_t>(gc::hook_state_e::capturing)) {
      set_state(gc::hook_state_e::capturing);
      log("Capturing %s frames from swapchain %p", api, static_cast<void *>(swapchain));
    }
  }

  // A free slot whose keyed mutex is available now (the host may still hold
  // the previously published one); -1 if none, -2 if a mutex was abandoned
  // (the shared surface must be recreated)
  int acquire_free_slot(bool (*ready)(int) = nullptr) {
    for (int i = 0; i < gc::kSlots; ++i) {
      if (word_state(g_slots[i].word.load(std::memory_order_acquire)) != st_free) {
        continue;
      }
      if (ready && !ready(i)) {
        continue;  // its command storage is still in use
      }
      if (g_cap.owner_sync) {
        // The host may still be reading the pixels it last locked here
        auto expected = gc::kOwnerNone;
        if (g_block->owner[i].compare_exchange_strong(expected, gc::kOwnerHook, std::memory_order_acq_rel)) {
          return i;
        }
        continue;
      }
      const HRESULT hr = g_cap.mutexes[i]->AcquireSync(0, 0);
      if (hr == S_OK) {
        return i;
      }
      if (hr == static_cast<HRESULT>(WAIT_ABANDONED)) {
        g_cap.mutexes[i]->ReleaseSync(0);
        return -2;
      }
    }
    return -1;
  }

  // ---- D3D12 ---------------------------------------------------------------
  //
  // A D3D12 copy must run on the queue the swapchain presents from: only that
  // orders it after the game's rendering of the back buffer and before the
  // flip. DXGI keeps that queue to itself (IDXGISwapChain::GetDevice hands
  // back the device, not the queue: checked on this machine), but its D3D12
  // Present submits on it, from the calling thread, inside the Present call.
  // So ID3D12CommandQueue::ExecuteCommandLists is detoured, and only while a
  // thread is inside the real Present of a swapchain (t_presenting) does it
  // note the queue it sees. After that Present returns the queue is stored on
  // the swapchain as private data (it lives and dies with the swapchain) if it
  // is a direct queue of the swapchain's own device on a single-node adapter.
  // A swapchain ever seen presenting from more than one queue is marked
  // ambiguous and never captured; one on which no queue shows up within
  // kQueueUnseenMs is reported unsupported. The first Present after injection
  // therefore only learns the queue; capture starts with the next. An
  // observation is thrown away if the Present failed or another swapchain
  // presented inside it. ResizeBuffers1 names the swapchain's queues, so it
  // sets the queue itself (or vetoes the swapchain if they differ); a plain
  // ResizeBuffers forgets it, to be learned again. Either way no copy is
  // submitted on a queue the swapchain no longer presents from.
  //
  // Per frame, on the presenting thread inside Present: the slot's own
  // command list (reused only once its previous copy completed) moves the
  // back buffer PRESENT -> COPY_SOURCE, copies it into the slot's shared
  // texture and moves it back, and is executed on that queue; the queue then
  // signals our fence, whose completion publishes the slot (owner words: D3D12
  // has no keyed mutexes). The shared textures are simultaneous-access: they
  // are promoted to COPY_DEST and decay back to COMMON by themselves, so the
  // host's D3D11 reads see a common-state resource.

  using execute_command_lists_fn = void(STDMETHODCALLTYPE *)(ID3D12CommandQueue *, UINT, ID3D12CommandList *const *);
  execute_command_lists_fn g_real_execute_command_lists = nullptr;

  // ID3D12CommandQueue: IUnknown 0-2, ID3D12Object 3-6, ID3D12DeviceChild 7,
  // UpdateTileMappings 8, CopyTileMappings 9, ExecuteCommandLists 10
  constexpr int kVtExecuteCommandLists = 10;
  constexpr std::uint64_t kQueueUnseenMs = 3000;

  // The presenting thread's observation during its real Present call
  thread_local IDXGISwapChain *t_presenting = nullptr;
  thread_local ID3D12CommandQueue *t_seen_queue = nullptr;  // a reference, taken inside its own ExecuteCommandLists
  thread_local bool t_seen_several = false;
  thread_local bool t_seen_contaminated = false;  // another swapchain presented inside this Present

  // Set when a swapchain veto could not be recorded, or a Signal failed:
  // D3D12 capture stops for the process (what our copies used is leaked)
  std::atomic<bool> g_d3d12_dead {false};

  void STDMETHODCALLTYPE hook_execute_command_lists(ID3D12CommandQueue *queue, UINT count, ID3D12CommandList *const *lists) {
    // Only direct queues can present; copy/compute submissions inside
    // Present (an overlay's uploads, say) are not candidates
    if (t_presenting && queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
      if (!t_seen_queue) {
        queue->AddRef();
        t_seen_queue = queue;
      } else if (t_seen_queue != queue) {
        t_seen_several = true;
      }
    }
    g_real_execute_command_lists(queue, count, lists);
  }

  std::uint64_t ms_to_qpc(std::uint64_t ms) {
    return ms * qpc_frequency() / 1000;
  }

  // Whether the swapchain was vetoed; a veto that cannot be read counts
  bool vetoed(IDXGISwapChain *swapchain) {
    std::uint32_t value = 0;
    UINT size = sizeof(value);
    const HRESULT hr = swapchain->GetPrivateData(kQueueAmbiguousKey, &size, &value);
    if (hr == DXGI_ERROR_NOT_FOUND) {
      return false;
    }
    return FAILED(hr) || size != sizeof(value) || value != 0;
  }

  void veto(IDXGISwapChain *swapchain, const char *why) {
    const std::uint32_t yes = 1;
    if (FAILED(swapchain->SetPrivateData(kQueueAmbiguousKey, sizeof(yes), &yes))) {
      g_d3d12_dead.store(true, std::memory_order_release);
      log("Swapchain %p %s, and the veto could not be recorded: D3D12 capture stopped", static_cast<void *>(swapchain), why);
      return;
    }
    log("Swapchain %p %s: not captured", static_cast<void *>(swapchain), why);
  }

  // Drops the swapchain's queue, to be learned again
  void forget_queue(IDXGISwapChain *swapchain) {
    if (FAILED(swapchain->SetPrivateDataInterface(kQueueKey, nullptr))) {
      veto(swapchain, "changed its buffers and its queue could not be forgotten");
    }
  }

  // Whether `queue` can take our copies for `swapchain`: a queue of the
  // swapchain's own device, on a single-node adapter (our textures live on
  // node 0). Only direct queues are ever passed here.
  bool usable_queue(IDXGISwapChain *swapchain, ID3D12CommandQueue *queue) {
    ID3D12Device *queue_device = nullptr, *swapchain_device = nullptr;
    queue->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&queue_device));
    swapchain->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&swapchain_device));
    const bool usable = queue->GetDesc().NodeMask <= 1 && queue_device && queue_device == swapchain_device;
    safe_release(queue_device);
    safe_release(swapchain_device);
    return usable;
  }

  void remember_queue(IDXGISwapChain *swapchain, ID3D12CommandQueue *queue, const char *how) {
    if (FAILED(swapchain->SetPrivateDataInterface(kQueueKey, queue))) {
      log("Swapchain %p: its D3D12 queue %p could not be remembered", static_cast<void *>(swapchain), static_cast<void *>(queue));
      return;
    }
    log("Swapchain %p presents from D3D12 queue %p (%s)", static_cast<void *>(swapchain), static_cast<void *>(queue), how);
  }

  // The queue this swapchain was seen presenting from (a new reference), or null
  ID3D12CommandQueue *known_queue(IDXGISwapChain *swapchain) {
    IUnknown *stored = nullptr;
    UINT size = sizeof(stored);
    if (FAILED(swapchain->GetPrivateData(kQueueKey, &size, &stored)) || size != sizeof(stored) || !stored) {
      return nullptr;  // missing or unreadable: not captured until learned
    }
    ID3D12CommandQueue *queue = nullptr;
    stored->QueryInterface(__uuidof(ID3D12CommandQueue), reinterpret_cast<void **>(&queue));
    stored->Release();
    return queue;
  }

  // After the real Present of `swapchain` on this thread: record what it
  // submitted on. Only D3D12 swapchains get here with something seen.
  void learn_present_queue(IDXGISwapChain *swapchain) {
    ID3D12CommandQueue *seen = t_seen_queue;
    t_seen_queue = nullptr;
    if (vetoed(swapchain)) {
      safe_release(seen);
      return;
    }
    ID3D12CommandQueue *known = known_queue(swapchain);
    if (t_seen_several || (known && known != seen)) {
      veto(swapchain, "presents from more than one D3D12 queue");
    } else if (!known) {
      if (usable_queue(swapchain, seen)) {
        remember_queue(swapchain, seen, "seen inside Present");
      } else {
        veto(swapchain, "presents from a D3D12 queue of another device or node");
      }
    }
    safe_release(known);
    safe_release(seen);
  }

  // Resizing may move a D3D12 swapchain to other queues: ResizeBuffers1 says
  // which (null: unchanged, as far as the documentation says; we relearn).
  // The swapchain is idle here (DXGI forbids a concurrent Present).
  void note_resize(IDXGISwapChain *swapchain, UINT count, IUnknown *const *queues) {
    ID3D12Device *device = nullptr;
    if (FAILED(swapchain->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&device)))) {
      return;  // not D3D12
    }
    device->Release();
    if (vetoed(swapchain)) {
      return;
    }
    if (!queues) {
      forget_queue(swapchain);
      return;
    }
    if (count == 0) {
      DXGI_SWAP_CHAIN_DESC desc {};
      count = SUCCEEDED(swapchain->GetDesc(&desc)) ? desc.BufferCount : 0;
    }
    ID3D12CommandQueue *queue = nullptr;
    bool one = count > 0;
    for (UINT i = 0; i < count && one; ++i) {
      ID3D12CommandQueue *q = nullptr;
      if (!queues[i] || FAILED(queues[i]->QueryInterface(__uuidof(ID3D12CommandQueue), reinterpret_cast<void **>(&q)))) {
        one = false;
      } else if (!queue) {
        queue = q;
        q = nullptr;
      } else if (q != queue) {
        one = false;
      }
      safe_release(q);
    }
    if (!one || !queue || queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT || !usable_queue(swapchain, queue)) {
      forget_queue(swapchain);
      veto(swapchain, "was resized onto several D3D12 queues, or one we cannot copy on");
    } else {
      remember_queue(swapchain, queue, "named by ResizeBuffers1");
    }
    safe_release(queue);
  }

  HRESULT STDMETHODCALLTYPE hook_resize_buffers(IDXGISwapChain *swapchain, UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags) {
    const HRESULT hr = g_real_resize_buffers(swapchain, count, width, height, format, flags);
    if (SUCCEEDED(hr)) {
      note_resize(swapchain, 0, nullptr);
    }
    return hr;
  }

  HRESULT STDMETHODCALLTYPE hook_resize_buffers1(IDXGISwapChain3 *swapchain, UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags,
                                                 const UINT *node_masks, IUnknown *const *queues) {
    const HRESULT hr = g_real_resize_buffers1(swapchain, count, width, height, format, flags, node_masks, queues);
    if (SUCCEEDED(hr)) {
      note_resize(swapchain, count, queues);
    }
    return hr;
  }

  void execute_on(ID3D12CommandQueue *queue, ID3D12CommandList *list) {
    if (g_real_execute_command_lists) {
      g_real_execute_command_lists(queue, 1, &list);
    } else {
      queue->ExecuteCommandLists(1, &list);
    }
  }

  // A slot's list whose Close or Reset failed cannot be reset again: replace
  // it (its last copy has completed, or it would not have been reset)
  bool recreate_list(int slot) {
    safe_release(g_cap.lists[slot]);
    const UINT node = g_cap.queue->GetDesc().NodeMask;
    HRESULT hr = g_cap.device12->CreateCommandList(node, D3D12_COMMAND_LIST_TYPE_DIRECT, g_cap.allocators[slot], nullptr,
                                                   __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void **>(&g_cap.lists[slot]));
    if (SUCCEEDED(hr)) {
      hr = g_cap.lists[slot]->Close();
    }
    if (FAILED(hr)) {
      log("Recreating the command list of slot %d failed: 0x%08lx", slot, hr);
      safe_release(g_cap.lists[slot]);
      return false;
    }
    return true;
  }

  bool list_ready(int slot) {
    return g_cap.lists[slot] && g_cap.fence12->GetCompletedValue() >= g_cap.list_fence[slot];
  }

  bool ensure_device12(IDXGISwapChain *swapchain, ID3D12Device *device, ID3D12CommandQueue *queue, HWND hwnd) {
    if (g_cap.api == gc::api_e::d3d12 && g_cap.device12 == device && g_cap.queue == queue && g_cap.swapchain == swapchain) {
      return true;
    }
    release_capture(hwnd);
    g_cap.api = gc::api_e::d3d12;
    g_cap.swapchain = swapchain;
    g_captured_swapchain.store(swapchain, std::memory_order_release);
    device->AddRef();
    g_cap.device12 = device;
    queue->AddRef();
    g_cap.queue = queue;
    g_cap.owner_thread = GetCurrentThreadId();
    g_cap.single_threaded = false;  // D3D12 objects are free-threaded
    g_cap.fence_value = 0;

    const UINT node = queue->GetDesc().NodeMask;
    const char *step = "fence";
    HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void **>(&g_cap.fence12));
    for (int i = 0; i < gc::kSlots && SUCCEEDED(hr); ++i) {
      step = "allocator";
      hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void **>(&g_cap.allocators[i]));
      if (SUCCEEDED(hr)) {
        step = "list";
        hr = device->CreateCommandList(node, D3D12_COMMAND_LIST_TYPE_DIRECT, g_cap.allocators[i], nullptr, __uuidof(ID3D12GraphicsCommandList),
                                       reinterpret_cast<void **>(&g_cap.lists[i]));
      }
      if (SUCCEEDED(hr)) {
        hr = g_cap.lists[i]->Close();  // created recording; closed until used
      }
    }
    if (FAILED(hr)) {
      char msg[96];
      std::snprintf(msg, sizeof(msg), "D3D12 setup failed at %s: 0x%08lx", step, hr);
      release_capture(hwnd);
      set_state(gc::hook_state_e::failed, msg);
      return false;
    }
    log("D3D12 device %p, presenting queue %p: frames timestamped at GPU completion", static_cast<void *>(device), static_cast<void *>(queue));
    return true;
  }

  bool ensure_textures12(const D3D12_RESOURCE_DESC &back, HWND hwnd) {
    const DXGI_FORMAT format = shareable_format(back.Format);
    const auto width = static_cast<UINT>(back.Width);
    if (g_cap.textures12[0] && g_cap.width == width && g_cap.height == back.Height && g_cap.format == format) {
      return true;
    }
    retire_generation(hwnd);

    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = back.Width;
    desc.Height = back.Height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    // Simultaneous access: the supported baseline for textures D3D11 opens
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

    const char *step = nullptr;
    HRESULT hr = S_OK;
    for (int i = 0; i < gc::kSlots && SUCCEEDED(hr); ++i) {
      step = "create";
      hr = g_cap.device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   __uuidof(ID3D12Resource), reinterpret_cast<void **>(&g_cap.textures12[i]));
      if (SUCCEEDED(hr)) {
        step = "nthandle";
        hr = g_cap.device12->CreateSharedHandle(g_cap.textures12[i], nullptr, GENERIC_ALL, nullptr, &g_cap.shared_handles[i]);
      }
    }
    if (FAILED(hr)) {
      for (int j = 0; j < gc::kSlots; ++j) {
        safe_release(g_cap.textures12[j]);
      }
      close_handles(g_cap.shared_handles);
      char msg[gc::kErrorLength];
      std::snprintf(msg, sizeof(msg), "D3D12 textures %ux%u f%d failed at %s: 0x%08lx", width, back.Height, format, step, hr);
      set_state(gc::hook_state_e::failed, msg);
      return false;
    }
    g_cap.owner_sync = true;
    g_cap.width = width;
    g_cap.height = back.Height;
    g_cap.format = format;
    if (g_current_generation.load(std::memory_order_acquire) == 0) {
      g_current_generation.store(1, std::memory_order_release);
    }
    publish_setup(hwnd, true);
    log("D3D12 capture textures %ux%u format %d, generation %u", width, back.Height, format, g_current_generation.load());
    return true;
  }

  D3D12_RESOURCE_BARRIER transition(ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
  }

  // Capture lock held. `device` is the swapchain's (borrowed).
  void capture_frame12(IDXGISwapChain *swapchain, ID3D12Device *device, HWND hwnd, std::uint64_t present_qpc, std::uint64_t release_qpc) {
    static IDXGISwapChain *unseen_swapchain = nullptr;
    static std::uint64_t unseen_since = 0;
    static bool reported = false;

    if (g_d3d12_dead.load(std::memory_order_acquire)) {
      if (g_cap.api == gc::api_e::d3d12) {
        release_capture(hwnd);
      }
      return;
    }
    if (!g_real_execute_command_lists || !g_real_resize_buffers1) {
      if (!reported) {
        reported = true;
        set_state(gc::hook_state_e::unsupported, "D3D12: ExecuteCommandLists or ResizeBuffers is not detoured, so the presenting queue cannot be tracked");
      }
      return;
    }
    if (FAILED(device->GetDeviceRemovedReason())) {
      release_capture(hwnd);  // everything of the lost device goes (its fence reads complete)
      return;
    }
    if (vetoed(swapchain)) {
      if (g_cap.swapchain == swapchain) {
        release_capture(hwnd);
      }
      if (!reported) {
        reported = true;
        set_state(gc::hook_state_e::unsupported, "D3D12: the swapchain presents from more than one queue, or not from a direct queue of its device");
      }
      return;
    }
    ID3D12CommandQueue *queue = known_queue(swapchain);
    if (!queue) {
      // Learned from this Present (after the real call); give up if it never shows
      const auto now = qpc_now();
      if (unseen_swapchain != swapchain) {
        unseen_swapchain = swapchain;
        unseen_since = now;
      } else if (!reported && now - unseen_since > ms_to_qpc(kQueueUnseenMs)) {
        reported = true;
        set_state(gc::hook_state_e::unsupported, "D3D12: no queue submission seen inside Present, so the presenting queue is unknown");
      }
      return;
    }
    unseen_swapchain = nullptr;
    const bool ready = ensure_device12(swapchain, device, queue, hwnd);
    queue->Release();
    if (!ready) {
      return;
    }

    IDXGISwapChain3 *swapchain3 = nullptr;
    if (FAILED(swapchain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&swapchain3)))) {
      return;
    }
    const UINT index = swapchain3->GetCurrentBackBufferIndex();
    swapchain3->Release();
    ID3D12Resource *back = nullptr;
    if (FAILED(swapchain->GetBuffer(index, __uuidof(ID3D12Resource), reinterpret_cast<void **>(&back)))) {
      return;
    }
    struct release_back_t {
      ID3D12Resource *&r;
      ~release_back_t() {
        safe_release(r);
      }
    } release_back {back};

    const auto back_desc = back->GetDesc();
    if (back_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || back_desc.SampleDesc.Count != 1 || back_desc.DepthOrArraySize != 1) {
      return;  // not a flip-model back buffer we know how to copy
    }
    if (!ensure_textures12(back_desc, hwnd)) {
      return;
    }
    const std::uint32_t color_space = swapchain_color_space(swapchain, g_cap.format);

    // A free slot whose list's last copy completed; claims its owner word
    const int slot = acquire_free_slot(&list_ready);
    if (slot < 0) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    auto give_back = [&] {
      g_block->owner[slot].store(gc::kOwnerNone, std::memory_order_release);
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
    };

    auto *list = g_cap.lists[slot];
    if (FAILED(g_cap.allocators[slot]->Reset()) || FAILED(list->Reset(g_cap.allocators[slot], nullptr))) {
      log("Resetting the command list of slot %d failed", slot);
      recreate_list(slot);
      give_back();
      return;
    }
    auto *target = g_cap.textures12[slot];
    D3D12_RESOURCE_BARRIER to_copy = transition(back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_RESOURCE_BARRIER to_present = transition(back, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
    list->ResourceBarrier(1, &to_copy);
    list->CopyResource(target, back);  // target: promoted to COPY_DEST, decays to COMMON
    list->ResourceBarrier(1, &to_present);
    if (const HRESULT closed = list->Close(); FAILED(closed)) {
      log("Closing the command list of slot %d failed: 0x%08lx", slot, closed);
      recreate_list(slot);
      give_back();
      return;
    }

    const auto generation = g_current_generation.load(std::memory_order_acquire);
    const auto frame_id = ++g_cap.next_frame_id;
    const auto version = write_slot_record(slot, generation, color_space, frame_id, present_qpc, release_qpc);

    execute_on(g_cap.queue, list);
    // The copy is still in flight, but the host takes only a published slot,
    // and this one publishes after the fence below completes
    g_block->owner[slot].store(gc::kOwnerNone, std::memory_order_release);

    const auto fence_value = ++g_cap.fence_value;
    if (const HRESULT signalled = g_cap.queue->Signal(g_cap.fence12, fence_value); FAILED(signalled)) {
      // The copy was submitted but nothing will say when it finished. Its
      // list, allocator and texture retire on this value, which only a
      // removed device ever reports complete: short of that they leak.
      g_cap.list_fence[slot] = fence_value;
      g_cap.highest_fence_value_used = fence_value;
      g_d3d12_dead.store(true, std::memory_order_release);
      release_capture(hwnd);
      char msg[80];
      std::snprintf(msg, sizeof(msg), "D3D12 Signal failed: 0x%08lx; capture stopped", signalled);
      set_state(gc::hook_state_e::failed, msg);
      return;
    }
    g_cap.list_fence[slot] = fence_value;
    g_cap.highest_fence_value_used = fence_value;
    const auto ticket = mark_pending(slot, version, generation, frame_id, fence_value);
    if (FAILED(g_cap.fence12->SetEventOnCompletion(fence_value, g_slots[slot].done_event))) {
      drop_pending(slot, ticket);  // the copy still completes; only the publication is lost
      return;
    }
    note_capturing(swapchain, "D3D12");
  }

  HWND setup_hwnd() {
    return reinterpret_cast<HWND>(g_block->setup.hwnd.load(std::memory_order_relaxed));
  }

  // The captured swapchain's device was lost (its Present said so): let go
  // of it now, since no further capture may come to notice
  void drop_lost_capture(IDXGISwapChain *swapchain) {
    if (g_captured_swapchain.load(std::memory_order_acquire) != swapchain) {
      return;
    }
    AcquireSRWLockExclusive(&g_capture_lock);
    if (g_cap.swapchain == swapchain) {
      log("The captured swapchain's device was lost: capture released");
      release_capture(setup_hwnd());
      collect_retired(false);
    }
    ReleaseSRWLockExclusive(&g_capture_lock);
  }

  // While not capturing: retired entries are still collected, and a lost
  // D3D12 device is still let go of
  void housekeeping() {
    if (!TryAcquireSRWLockExclusive(&g_capture_lock)) {
      return;
    }
    if (g_cap.device12 && FAILED(g_cap.device12->GetDeviceRemovedReason())) {
      release_capture(setup_hwnd());
    }
    collect_retired(false);
    ReleaseSRWLockExclusive(&g_capture_lock);
  }

  void capture_frame(IDXGISwapChain *swapchain, std::uint64_t present_qpc, std::uint64_t release_qpc) {
    g_block->frames_presented.fetch_add(1, std::memory_order_relaxed);
    if (!g_block->capture_enabled.load(std::memory_order_acquire)) {
      housekeeping();
      return;
    }
    HWND hwnd = nullptr;
    if (!foreground_window(swapchain, hwnd)) {
      housekeeping();
      return;
    }

    // One capture at a time; a concurrent Present elsewhere just skips
    if (!TryAcquireSRWLockExclusive(&g_capture_lock)) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    struct unlock_t {
      ~unlock_t() {
        ReleaseSRWLockExclusive(&g_capture_lock);
      }
    } unlock;

    // A single-threaded device is touched only from the thread that first
    // presented it: any other thread skips before looking at any device
    if (g_cap.device && g_cap.single_threaded && GetCurrentThreadId() != g_cap.owner_thread) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    collect_retired(false);

    static std::uint32_t seen_recreate = 0;
    const auto recreate = g_block->recreate_request.load(std::memory_order_acquire);
    if (recreate != seen_recreate) {
      seen_recreate = recreate;
      log("The host asked for fresh capture textures");
      retire_generation(hwnd);
    }

    ID3D11Device *device = nullptr;
    if (FAILED(swapchain->GetDevice(__uuidof(ID3D11Device), reinterpret_cast<void **>(&device)))) {
      ID3D12Device *device12 = nullptr;
      if (FAILED(swapchain->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&device12)))) {
        static bool reported = false;
        if (!reported) {
          reported = true;
          set_state(gc::hook_state_e::unsupported, "The swapchain is neither D3D11 nor D3D12");
        }
        return;
      }
      capture_frame12(swapchain, device12, hwnd, present_qpc, release_qpc);
      device12->Release();
      return;
    }
    ensure_device(swapchain, device, hwnd);
    device->Release();

    ID3D11Texture2D *back = nullptr;
    if (FAILED(swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&back)))) {
      return;
    }
    D3D11_TEXTURE2D_DESC back_desc {};
    back->GetDesc(&back_desc);
    if (!ensure_textures(back_desc, hwnd)) {
      back->Release();
      return;
    }
    const std::uint32_t color_space = swapchain_color_space(swapchain, g_cap.format);

    const int slot = acquire_free_slot();
    if (slot == -2) {
      log("A capture texture's keyed mutex was abandoned; recreating the textures");
      retire_generation(hwnd);  // the next Present creates a fresh set
      back->Release();
      return;
    }
    if (slot < 0) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      back->Release();
      return;
    }

    // Holding the slot's pixel mutex: rewrite its record, then the pixels.
    // The host's acquire is GPU-ordered after the release below, so a host
    // holding the mutex and reading an even record reads this frame's.
    const auto generation = g_current_generation.load(std::memory_order_acquire);
    const auto frame_id = ++g_cap.next_frame_id;
    const auto version = write_slot_record(slot, generation, color_space, frame_id, present_qpc, release_qpc);

    if (back_desc.SampleDesc.Count > 1) {
      g_cap.context->ResolveSubresource(g_cap.textures[slot], 0, back, 0, g_cap.format);
    } else {
      g_cap.context->CopyResource(g_cap.textures[slot], back);
    }
    if (g_cap.owner_sync) {
      // The copy is still in flight, but the host takes only a published
      // slot, and this one publishes after its fence completes
      g_block->owner[slot].store(gc::kOwnerNone, std::memory_order_release);
    } else {
      g_cap.mutexes[slot]->ReleaseSync(0);
    }
    back->Release();

    auto &s = g_slots[slot];
    const auto ticket = mark_pending(slot, version, generation, frame_id, (g_cap.fence && g_cap.context4) ? ++g_cap.fence_value : 0);

    // Completion always goes through the completion thread (single
    // publisher). Without a fence, or if registering its event fails, the
    // event is set here and the frame publishes at Present time.
    bool signalled = false;
    if (s.fence_value) {
      g_cap.highest_fence_value_used = s.fence_value;
      signalled = SUCCEEDED(g_cap.context4->Signal(g_cap.fence, s.fence_value)) &&
                  SUCCEEDED(g_cap.fence->SetEventOnCompletion(s.fence_value, s.done_event));
      if (signalled && g_cap.owner_sync) {
        // Without a keyed mutex nothing else submits the copy: a shared
        // resource's writer must flush (OpenSharedResource)
        g_cap.context->Flush();
      }
    }
    if (!signalled) {
      if (s.fence_value) {
        drop_pending(slot, ticket);
      } else {
        SetEvent(s.done_event);
      }
    }
    note_capturing(swapchain, "D3D11");
  }


  // ---- frame limiter -------------------------------------------------------
  //
  // Front edge: the wait comes AFTER the real Present returns, so the game
  // starts its next frame, and samples input, at the release time, and its
  // next Present (where we capture) follows as soon as that frame is
  // rendered. A limiter that waits before Present (RTSS async) holds a
  // finished frame instead: its content ages by the whole wait.
  //
  // One Present paces at a time: it takes `owner` before deciding anything
  // and keeps it through its wait, and every other Present meanwhile (another
  // thread, another swapchain) runs unpaced and untouched. All limiter state
  // below belongs to the owner. One swapchain is paced: the one being
  // captured when it presents, else the first to present, re-chosen once it
  // has been idle for half a second; a change of swapchain starts a new grid.
  // Presents that must not block (DXGI_PRESENT_DO_NOT_WAIT) are never paced.

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
  #define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

  struct limiter_t {
    std::atomic<DWORD> owner {0};  // thread id of the pacing Present; 0 = none

    // Owner only
    IDXGISwapChain *swapchain = nullptr;  // identity only
    std::uint64_t last_present_qpc = 0;
    std::uint64_t release_qpc = 0;  // the release that started the frame now being rendered on `swapchain`
    HANDLE timer = nullptr;  // one timer: only the owner waits
    gc::limiter_logic_t logic;  // the release grid (limiter_logic.h), in QPC ticks
  };

  limiter_t g_limiter;

  // The period to pace at, in QPC ticks; 0 while the limiter is off (no
  // period, or the host stopped refreshing its heartbeat)
  double limiter_period_qpc(std::uint64_t now) {
    const auto period_ps = g_block->limiter_period_ps.load(std::memory_order_acquire);
    if (period_ps < gc::kMinLimiterPeriodPs || period_ps > gc::kMaxLimiterPeriodPs) {
      return 0;
    }
    const auto heartbeat = g_block->host_heartbeat_qpc.load(std::memory_order_acquire);
    const auto timeout = gc::kHeartbeatTimeoutMs * qpc_frequency() / 1000;
    if (heartbeat == 0 || now > heartbeat + timeout || heartbeat > now + timeout) {
      return 0;
    }
    return static_cast<double>(period_ps) * 1e-12 * static_cast<double>(qpc_frequency());
  }

  bool limiter_take() {
    DWORD expected = 0;
    return g_limiter.owner.compare_exchange_strong(expected, GetCurrentThreadId(), std::memory_order_acquire);
  }

  void limiter_give_back() {
    g_limiter.owner.store(0, std::memory_order_release);
  }

  void limiter_forget() {
    g_limiter.swapchain = nullptr;
    g_limiter.release_qpc = 0;
    g_limiter.logic.forget();
  }

  // Owner only: whether this swapchain is the paced one
  bool limiter_select(IDXGISwapChain *swapchain, std::uint64_t now) {
    if (g_limiter.swapchain != swapchain) {
      const bool captured = g_captured_swapchain.load(std::memory_order_acquire) == swapchain;
      const bool idle = !g_limiter.swapchain || now < g_limiter.last_present_qpc || now - g_limiter.last_present_qpc > qpc_frequency() / 2;
      if (!captured && !idle) {
        return false;
      }
      limiter_forget();  // a new grid: nothing of the previous swapchain's carries over
      g_limiter.swapchain = swapchain;
    }
    g_limiter.last_present_qpc = now;
    return true;
  }

  // Owner only. A high-resolution waitable timer for all but the last
  // millisecond (its wakeups land within ~0.5 ms), then a spin to the exact
  // point.
  void sleep_until(std::uint64_t target, double period) {
    const auto freq = qpc_frequency();
    const auto spin = freq / 1000;
    const auto now = qpc_now();
    if (target > now + spin) {
      if (!g_limiter.timer) {
        g_limiter.timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (!g_limiter.timer) {
          g_limiter.timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);  // before Windows 10 1803
        }
      }
      const auto ticks = target - spin - now;
      const auto bound_ms = static_cast<DWORD>(2 * period * 1000 / static_cast<double>(freq)) + 2;
      LARGE_INTEGER due;
      due.QuadPart = -static_cast<LONGLONG>(ticks * 10'000'000ull / freq);  // relative, 100 ns units
      if (g_limiter.timer && due.QuadPart < 0 && SetWaitableTimer(g_limiter.timer, &due, 0, nullptr, nullptr, FALSE)) {
        WaitForSingleObject(g_limiter.timer, bound_ms);
      } else {
        Sleep(std::min<DWORD>(static_cast<DWORD>(ticks * 1000 / freq), bound_ms));
      }
    }
    while (qpc_now() < target) {
      YieldProcessor();
    }
  }

  // Owner only, after a successful real Present of the paced swapchain: wait
  // for the next release point (limiter_logic.h decides where it is)
  void limiter_wait(double period) {
    auto &l = g_limiter;
    const auto now = qpc_now();
    const double nowd = static_cast<double>(now);
    const auto plan = l.logic.plan(nowd, period);
    if (plan.reset) {
      g_block->limiter_resets.fetch_add(1, std::memory_order_relaxed);
    } else if (plan.late) {
      g_block->limiter_late.fetch_add(1, std::memory_order_relaxed);
    }
    const auto target_qpc = static_cast<std::uint64_t>(plan.release);
    if (target_qpc > now) {
      sleep_until(target_qpc, period);
    }
    const auto released = qpc_now();
    g_block->limiter_waits.fetch_add(1, std::memory_order_relaxed);
    g_block->limiter_wait_us.fetch_add((released - now) * 1'000'000ull / qpc_frequency(), std::memory_order_relaxed);
    l.release_qpc = released;
  }

  // What one outermost Present does around the real call
  template<class F>
  HRESULT present_with_capture(IDXGISwapChain *swapchain, UINT sync_interval, UINT flags, F &&real) {
    if (flags & DXGI_PRESENT_TEST) {
      return real(sync_interval);
    }
    const auto now = qpc_now();
    const double period = g_block && !(flags & DXGI_PRESENT_DO_NOT_WAIT) ? limiter_period_qpc(now) : 0;

    bool paced = false;
    std::uint64_t release = 0;
    if (period > 0) {
      if (limiter_take()) {
        paced = limiter_select(swapchain, now);
        if (paced) {
          release = g_limiter.release_qpc;
        } else {
          limiter_give_back();
        }
      }
    } else if (g_block && !(flags & DXGI_PRESENT_DO_NOT_WAIT) && limiter_take()) {
      limiter_forget();  // off: the next start begins a fresh grid
      limiter_give_back();
    }

    // (scoped: a downstream hook's exception must not keep the token)
    struct give_back_t {
      bool paced;

      ~give_back_t() {
        if (paced) {
          limiter_give_back();
        }
      }
    } give_back {paced};

    capture_frame(swapchain, now, release);

    // Paced, the game must not also wait for the host display's vblank. For
    // a D3D12 swapchain the queue DXGI submits on during this call is the
    // presenting queue (see the D3D12 section). The observation is scoped
    // so that nothing of it outlives the call, however it ends.
    struct observation_t {
      explicit observation_t(IDXGISwapChain *swapchain) {
        t_presenting = swapchain;
      }

      ~observation_t() {
        t_presenting = nullptr;
        safe_release(t_seen_queue);
        t_seen_several = false;
        t_seen_contaminated = false;
      }
    };

    HRESULT hr;
    {
      observation_t observing(swapchain);
      hr = real(paced ? 0 : sync_interval);
      t_presenting = nullptr;
      if (t_seen_queue && SUCCEEDED(hr) && !t_seen_contaminated) {
        learn_present_queue(swapchain);
      }
    }
    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
      drop_lost_capture(swapchain);
    }
    if (paced && SUCCEEDED(hr)) {
      limiter_wait(period);
    }
    return hr;
  }

  // ---- detours -------------------------------------------------------------

  // DXGI may implement one Present entry point through the other; only the
  // outermost call on a thread captures. A nested Present of *another*
  // swapchain spoils the outer one's queue observation.
  struct in_present_t {
    in_present_t() {
      t_in_present = true;
    }

    ~in_present_t() {
      t_in_present = false;
    }
  };

  void note_nested(IDXGISwapChain *swapchain) {
    if (!t_presenting || t_presenting == swapchain) {
      return;
    }
    // (DXGI may call itself through another interface of the same object)
    IUnknown *outer = nullptr, *inner = nullptr;
    t_presenting->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void **>(&outer));
    swapchain->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void **>(&inner));
    if (!outer || outer != inner) {
      t_seen_contaminated = true;
    }
    safe_release(outer);
    safe_release(inner);
  }

  HRESULT STDMETHODCALLTYPE hook_present(IDXGISwapChain *swapchain, UINT sync_interval, UINT flags) {
    if (t_in_present) {
      note_nested(swapchain);
      return g_real_present(swapchain, sync_interval, flags);
    }
    in_present_t in_present;
    return present_with_capture(swapchain, sync_interval, flags, [&](UINT interval) {
      return g_real_present(swapchain, interval, flags);
    });
  }

  HRESULT STDMETHODCALLTYPE hook_present1(IDXGISwapChain1 *swapchain, UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS *params) {
    if (t_in_present) {
      note_nested(swapchain);
      return g_real_present1(swapchain, sync_interval, flags, params);
    }
    in_present_t in_present;
    return present_with_capture(swapchain, sync_interval, flags, [&](UINT interval) {
      return g_real_present1(swapchain, interval, flags, params);
    });
  }

  HRESULT STDMETHODCALLTYPE hook_set_color_space1(IDXGISwapChain3 *swapchain, DXGI_COLOR_SPACE_TYPE color_space) {
    const HRESULT hr = g_real_set_color_space1(swapchain, color_space);
    if (SUCCEEDED(hr)) {
      auto cs = gc::color_space_e::unknown;
      if (color_space == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) {
        cs = gc::color_space_e::srgb;
      } else if (color_space == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709) {
        cs = gc::color_space_e::scrgb;
      } else if (color_space == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
        cs = gc::color_space_e::hdr10;
      }
      const auto value = static_cast<std::uint32_t>(cs);
      if (FAILED(swapchain->SetPrivateData(kColorSpaceKey, sizeof(value), &value))) {
        g_color_tracking_lost.store(true, std::memory_order_release);
        log("Swapchain %p color space %d could not be recorded", static_cast<void *>(swapchain), static_cast<int>(color_space));
      } else {
        log("Swapchain %p color space %d", static_cast<void *>(swapchain), static_cast<int>(color_space));
      }
    }
    return hr;
  }

  // ID3D12CommandQueue::ExecuteCommandLists, found from a throwaway queue on
  // a D3D12 device (the runtime hands back the process's existing device
  // for the adapter). Only when the game has loaded d3d12.dll; its runtime
  // (the Agility SDK's D3D12Core.dll if it ships one) is the one we reach.
  void install_d3d12_hook() {
    HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
    if (!d3d12) {
      return;
    }
    using create_device_fn = HRESULT(WINAPI *)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
    auto create_device = reinterpret_cast<create_device_fn>(reinterpret_cast<void *>(GetProcAddress(d3d12, "D3D12CreateDevice")));
    if (!create_device) {
      return;
    }
    ID3D12Device *device = nullptr;
    ID3D12CommandQueue *queue = nullptr;
    HRESULT hr = create_device(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void **>(&device));
    if (SUCCEEDED(hr)) {
      D3D12_COMMAND_QUEUE_DESC desc {};
      desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      hr = device->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void **>(&queue));
    }
    if (SUCCEEDED(hr)) {
      void **vtable = *reinterpret_cast<void ***>(queue);
      if (MH_CreateHook(vtable[kVtExecuteCommandLists], reinterpret_cast<void *>(&hook_execute_command_lists),
                        reinterpret_cast<void **>(&g_real_execute_command_lists)) != MH_OK) {
        g_real_execute_command_lists = nullptr;
        log("ExecuteCommandLists could not be detoured; D3D12 swapchains cannot be captured");
      } else {
        log("Detoured ID3D12CommandQueue::ExecuteCommandLists");
      }
    } else {
      log("No D3D12 device for the queue detour: 0x%08lx", hr);
    }
    safe_release(queue);
    safe_release(device);
  }

  // The entry points are found from a throwaway swapchain and inline-detoured
  // at their addresses, which every swapchain of the process shares.
  bool install_hooks() {
    WNDCLASSW wc {};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"VibepolloGameHookDummy";
    RegisterClassW(&wc);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"", WS_OVERLAPPEDWINDOW, 0, 0, 16, 16, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) {
      set_state(gc::hook_state_e::failed, "Could not create the dummy window");
      return false;
    }

    bool ok = false;
    ID3D11Device *device = nullptr;
    ID3D11DeviceContext *context = nullptr;
    IDXGIFactory2 *factory = nullptr;
    IDXGISwapChain1 *swapchain = nullptr;
    D3D_FEATURE_LEVEL level;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, &level, &context);
    if (SUCCEEDED(hr)) {
      hr = CreateDXGIFactory1(__uuidof(IDXGIFactory2), reinterpret_cast<void **>(&factory));
    }
    if (SUCCEEDED(hr)) {
      DXGI_SWAP_CHAIN_DESC1 desc {};
      desc.Width = 16;
      desc.Height = 16;
      desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
      desc.SampleDesc.Count = 1;
      desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
      desc.BufferCount = 2;
      desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
      hr = factory->CreateSwapChainForHwnd(device, hwnd, &desc, nullptr, nullptr, &swapchain);
    }
    if (SUCCEEDED(hr)) {
      void **vtable = *reinterpret_cast<void ***>(swapchain);
      ok = MH_Initialize() == MH_OK &&
           MH_CreateHook(vtable[kVtPresent], reinterpret_cast<void *>(&hook_present), reinterpret_cast<void **>(&g_real_present)) == MH_OK &&
           MH_CreateHook(vtable[kVtPresent1], reinterpret_cast<void *>(&hook_present1), reinterpret_cast<void **>(&g_real_present1)) == MH_OK;
      IDXGISwapChain3 *swapchain3 = nullptr;
      if (ok && SUCCEEDED(swapchain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&swapchain3)))) {
        void **vtable3 = *reinterpret_cast<void ***>(swapchain3);
        if (MH_CreateHook(vtable3[kVtSetColorSpace1], reinterpret_cast<void *>(&hook_set_color_space1), reinterpret_cast<void **>(&g_real_set_color_space1)) != MH_OK) {
          log("SetColorSpace1 could not be detoured; 10-bit swapchains stay unknown");
        }
        // D3D12 capture needs both (a resize may change the presenting queue)
        if (MH_CreateHook(vtable[kVtResizeBuffers], reinterpret_cast<void *>(&hook_resize_buffers), reinterpret_cast<void **>(&g_real_resize_buffers)) != MH_OK) {
          g_real_resize_buffers = nullptr;
          log("ResizeBuffers could not be detoured: D3D12 swapchains are not captured");
        } else if (MH_CreateHook(vtable3[kVtResizeBuffers1], reinterpret_cast<void *>(&hook_resize_buffers1), reinterpret_cast<void **>(&g_real_resize_buffers1)) != MH_OK) {
          g_real_resize_buffers1 = nullptr;
          log("ResizeBuffers1 could not be detoured: D3D12 swapchains are not captured");
        }
        swapchain3->Release();
      }
      if (ok) {
        install_d3d12_hook();
        ok = MH_EnableHook(MH_ALL_HOOKS) == MH_OK;
      }
      if (!ok) {
        MH_DisableHook(MH_ALL_HOOKS);
        set_state(gc::hook_state_e::failed, "Installing the DXGI detours failed");
      }
    } else {
      char msg[96];
      std::snprintf(msg, sizeof(msg), "Could not create the dummy swapchain: 0x%08lx", hr);
      set_state(gc::hook_state_e::failed, msg);
    }
    safe_release(swapchain);
    safe_release(factory);
    safe_release(context);
    safe_release(device);
    DestroyWindow(hwnd);
    return ok;
  }

  DWORD WINAPI init_thread_main(void *) {
    wchar_t name[96];
    gc::shared_block_name(name, 96, GetCurrentProcessId());
    HANDLE mapping = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name);
    if (!mapping) {
      log("No shared block %ls (error %lu); not capturing", name, GetLastError());
      return 1;
    }
    g_block = static_cast<gc::shared_block_t *>(MapViewOfFile(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(gc::shared_block_t)));
    if (!g_block || g_block->magic != gc::kMagic || g_block->version != gc::kVersion) {
      log("Shared block missing or version mismatch; not capturing");
      g_block = nullptr;
      return 1;
    }

    g_frame_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    bool events_ok = g_frame_event != nullptr;
    for (auto &s : g_slots) {
      s.done_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
      events_ok = events_ok && s.done_event != nullptr;
    }
    if (!events_ok) {
      set_state(gc::hook_state_e::failed, "Could not create events");
      return 1;
    }
    g_block->frame_event.store(reinterpret_cast<std::uint64_t>(g_frame_event), std::memory_order_release);
    g_completion_thread = CreateThread(nullptr, 0, completion_thread_main, nullptr, 0, nullptr);
    if (!g_completion_thread) {
      set_state(gc::hook_state_e::failed, "Could not create the completion thread");
      return 1;
    }

    if (!install_hooks()) {
      return 1;
    }
    set_state(gc::hook_state_e::hooked);
    log("Detoured DXGI Present/Present1 in pid %lu", GetCurrentProcessId());
    return 0;
  }

}  // namespace

// The DLL stays loaded for the life of the process: the detours point into
// it, and there is no safe moment to take them back mid-frame. Setup runs on
// its own thread, which DllMain does not wait for (no loader-lock deadlock).
BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    DisableThreadLibraryCalls(instance);
    HANDLE thread = CreateThread(nullptr, 0, init_thread_main, nullptr, 0, nullptr);
    if (thread) {
      CloseHandle(thread);
    }
  }
  return TRUE;
}
