/**
 * @file tools/game_hook/game_hook.cpp
 * @brief In-game capture hook (vibepollo_game_hook.dll), injected by Vibepollo
 *        into a focused fullscreen game. See
 *        src/platform/windows/game_capture/protocol.h for the protocol.
 *
 * D3D11 and D3D12 swapchains are captured. The DXGI entry points (Present,
 * Present1, ResizeBuffers(1)) are inline-detoured with MinHook at their
 * function addresses, found from a throwaway swapchain: that catches every
 * swapchain in the process whatever its swap effect or creation API, and
 * overlays that keep private vtables (their cached "original" is the function
 * we patched). D3D12 also needs the queue a swapchain presents from, which
 * DXGI does not expose: ID3D12CommandQueue::ExecuteCommandLists is detoured
 * too, and a frame is captured only when that queue is certain (see the D3D12
 * section). D3D11 frames are copied at dxgi!CDXGISwapChain::PresentImpl,
 * the internal function both public Present entries reach after every hook
 * on them (an overlay's) has run: see the PresentImpl section. A frame's
 * colour space is read from its swapchain through DXGI's private getter
 * (see swapchain_color_space).
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
#include "tools/vk_layer/vk_layer_api.h"

#include <MinHook.h>

extern "C" {
#include "hde/hde64.h"
}

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
#include <cwchar>
#include <string>
#include <vector>

// DXGI's private swapchain interface (see swapchain_color_space). Declared
// outside the anonymous namespace on purpose: there GCC sees that no class
// in the program implements it and compiles calls through it as unreachable
// (a call to nowhere, which crashed Stellar Blade).
struct IDXGISwapChainTest : IUnknown {
  virtual bool STDMETHODCALLTYPE HasProxyFrontBufferSurface() = 0;
  virtual HRESULT STDMETHODCALLTYPE GetFrameStatisticsTest(void *) = 0;
  virtual void STDMETHODCALLTYPE EmulateXBOXBehavior(BOOL) = 0;
  virtual DXGI_COLOR_SPACE_TYPE STDMETHODCALLTYPE GetColorSpace1() = 0;
};

namespace {

  namespace gc = game_capture;

  using present_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT);
  using present1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain1 *, UINT, UINT, const DXGI_PRESENT_PARAMETERS *);
  using resize_buffers_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT, UINT, DXGI_FORMAT, UINT);
  using resize_buffers1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain3 *, UINT, UINT, UINT, DXGI_FORMAT, UINT, const UINT *, IUnknown *const *);

  // IDXGISwapChain vtable slots (IUnknown 0-2, IDXGIObject 3-6, IDXGIDeviceSubObject 7)
  constexpr int kVtPresent = 8;
  constexpr int kVtResizeBuffers = 13;
  constexpr int kVtPresent1 = 22;  // IDXGISwapChain1
  constexpr int kVtResizeBuffers1 = 39;  // IDXGISwapChain3

  // D3D12: the queue a swapchain was seen presenting from (an interface), and
  // whether it was ever seen presenting from more than one (a flag)
  // {3b9d7e21-6c4a-4f0e-8a5d-1e7c9b2f4a63}, {5e2a8c14-9b3d-4d71-b6e0-7f4a2c9d1e85}
  constexpr GUID kQueueKey = {0x3b9d7e21, 0x6c4a, 0x4f0e, {0x8a, 0x5d, 0x1e, 0x7c, 0x9b, 0x2f, 0x4a, 0x63}};
  constexpr GUID kQueueAmbiguousKey = {0x5e2a8c14, 0x9b3d, 0x4d71, {0xb6, 0xe0, 0x7f, 0x4a, 0x2c, 0x9d, 0x1e, 0x85}};
  // ... and which of its back buffers were seen presented from it (a bitmask
  // of buffer indices; all ones when ResizeBuffers1 named the queues)
  // {7c1f4b9a-2e6d-4a83-9f15-3b8e6d2a7c40}
  constexpr GUID kQueueSeenKey = {0x7c1f4b9a, 0x2e6d, 0x4a83, {0x9f, 0x15, 0x3b, 0x8e, 0x6d, 0x2a, 0x7c, 0x40}};
  // ... and how its frames are copied (submit_mode_e, a count)
  // {2d8e5f17-4c3b-4e9a-a061-9b7d3c5e2f84}
  constexpr GUID kSubmitModeKey = {0x2d8e5f17, 0x4c3b, 0x4e9a, {0xa0, 0x61, 0x9b, 0x7d, 0x3c, 0x5e, 0x2f, 0x84}};

  present_fn g_real_present = nullptr;
  present1_fn g_real_present1 = nullptr;
  resize_buffers_fn g_real_resize_buffers = nullptr;  // both or neither (D3D12 capture needs them)
  resize_buffers1_fn g_real_resize_buffers1 = nullptr;

  gc::shared_block_t *g_block = nullptr;
  thread_local bool t_in_present = false;

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
    if (!g_block || g_block->hook_state.load(std::memory_order_acquire) == static_cast<std::uint32_t>(gc::hook_state_e::fatal)) {
      return;  // (fatal stays: the host must see it)
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
    IUnknown *motion[gc::kSlots] = {};  // the generation's motion textures (same kinds)
    IDXGIKeyedMutex *mutexes[gc::kSlots] = {};
    IDXGIKeyedMutex *motion_mutexes[gc::kSlots] = {};
    HANDLE handles[gc::kSlots] = {};
    HANDLE motion_handles[gc::kSlots] = {};
    ID3D11Fence *fence = nullptr;  // the fence its copies were signalled on (a reference; D3D11)
    ID3D12Fence *fence12 = nullptr;  // ... or D3D12 (then nothing is released before it completes)
    ID3D12CommandAllocator *allocators[gc::kSlots] = {};  // D3D12 command storage of a released capture
    ID3D12GraphicsCommandList *lists[gc::kSlots] = {};
    std::uint64_t highest_fence_value = 0;
    std::uint64_t retired_qpc = 0;
    bool textures_released = false;
    bool legacy = false;  // shared through global handles: the textures themselves keep the values valid
    bool in_use = false;
    DWORD owner_thread = 0;  // a single-threaded D3D11 device's thread, the only one that may release it; 0 = any
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

    // Motion textures of the generation (see the motion section), shared like
    // the textures and guarded with them; motion_format UNKNOWN = none
    ID3D11Texture2D *motion11[gc::kSlots] = {};
    IDXGIKeyedMutex *motion_mutexes[gc::kSlots] = {};  // D3D11 with keyed-mutex sync: each motion texture's own
    ID3D12Resource *motion12[gc::kSlots] = {};
    HANDLE motion_handles[gc::kSlots] = {};
    std::uint64_t motion_legacy[gc::kSlots] = {};
    DXGI_FORMAT motion_format = DXGI_FORMAT_UNKNOWN;
    UINT motion_width = 0;
    UINT motion_height = 0;
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

  // The motion textures a generation should have (see the motion section)
  struct motion_want_t {
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;  // UNKNOWN: none wanted
    UINT width = 0;
    UINT height = 0;

    bool operator==(const motion_want_t &o) const {
      return format == o.format && width == o.width && height == o.height;
    }
  };

  void motion_unread(std::uint32_t ring_serial, int entry);  // (motion section)
  void motion_on_present(IDXGISwapChain *swapchain);

  // The vectors a captured frame carries: one evaluation's ring texture (a
  // reference, see the motion section) and where they lie in it
  struct motion_pick_t {
    std::uint64_t id = 0;  // 0 = none; else the frame's motion sequence number
    ID3D11Texture2D *src11 = nullptr;
    ID3D12Resource *src12 = nullptr;
    UINT width = 0;  // the region, from the ring texture's top-left
    UINT height = 0;
    UINT out_width = 0;  // the DLSS output it spans
    UINT out_height = 0;
    float scale_x = 1;
    float scale_y = 1;
    motion_want_t textures;  // the ring's format and size: what the generation's motion textures must be
    std::uint32_t ring = 0;  // the ring's serial and entry
    int entry = -1;
    bool reading = false;  // D3D12: the entry is reserved for a capture copy (handed on to it, or given back here)

    void release() {
      if (reading) {
        motion_unread(ring, entry);
        reading = false;
      }
      safe_release(src11);
      safe_release(src12);
      id = 0;
    }
  };

  motion_want_t g_motion_refused;  // a set the device would not create: not asked for again (capture lock)

  // Whether the current generation serves `want` (one that has motion
  // textures keeps them when none is wanted)
  bool motion_satisfied(const motion_want_t &want) {
    return want.format == DXGI_FORMAT_UNKNOWN || want == g_motion_refused ||
           (g_cap.motion_format == want.format && g_cap.motion_width == want.width && g_cap.motion_height == want.height);
  }

  std::atomic<std::uint64_t> g_vk_completed {0};  // Vulkan: the last copy the fence waiter saw finish (UINT64_MAX: device lost)

  std::uint64_t fence_completed() {
    if (g_cap.api == gc::api_e::vulkan) {
      return g_vk_completed.load(std::memory_order_acquire);
    }
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
    const bool motion = valid && g_cap.motion_format != DXGI_FORMAT_UNKNOWN;
    for (int i = 0; i < gc::kSlots; ++i) {
      const auto value = g_cap.motion_handles[i] ? reinterpret_cast<std::uint64_t>(g_cap.motion_handles[i]) : g_cap.motion_legacy[i];
      b.setup.motion_textures[i].store(motion ? value : 0, std::memory_order_relaxed);
    }
    b.setup.motion_format.store(motion ? static_cast<std::uint32_t>(g_cap.motion_format) : 0, std::memory_order_relaxed);
    b.setup.motion_width.store(motion ? g_cap.motion_width : 0, std::memory_order_relaxed);
    b.setup.motion_height.store(motion ? g_cap.motion_height : 0, std::memory_order_relaxed);
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
      safe_release(r.motion_mutexes[i]);
      safe_release(r.textures[i]);
      safe_release(r.motion[i]);
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
    const DWORD thread = GetCurrentThreadId();
    for (auto &r : g_cap.retired) {
      if (!r.in_use || (r.owner_thread && r.owner_thread != thread)) {
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
      const auto some = [](const HANDLE (&handles)[gc::kSlots]) {
        return std::any_of(std::begin(handles), std::end(handles), [](HANDLE h) {
          return h != nullptr;
        });
      };
      const bool has_handles = r.legacy || some(r.handles) || some(r.motion_handles);
      if (r.textures_released && (!has_handles || now - r.retired_qpc > 2 * qpc_frequency() || force)) {
        close_handles(r.handles);
        close_handles(r.motion_handles);
        r.in_use = false;
      }
    }
  }

  // Retiring must not allocate (it runs inside Present, and what it keeps
  // must not be dropped), so free entries are reserved up front: enough for
  // the most one capture pass can retire (a recreate, a device switch and a
  // failed Signal: two texture sets and two command stores, plus one spare)
  constexpr std::size_t kRetireReserve = 5;

  bool reserve_retirement() {
    const auto free = static_cast<std::size_t>(std::count_if(g_cap.retired.begin(), g_cap.retired.end(), [](const retired_t &r) {
      return !r.in_use;
    }));
    if (free + (g_cap.retired.capacity() - g_cap.retired.size()) >= kRetireReserve) {
      return true;
    }
    try {
      g_cap.retired.reserve(g_cap.retired.size() + kRetireReserve);
      return true;
    } catch (...) {
      return false;
    }
  }

  // A free entry of the retirement list (a new one if none is free; there
  // is room: reserve_retirement)
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
    slot->owner_thread = g_cap.single_threaded ? g_cap.owner_thread : 0;
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
    for (int i = 0; i < gc::kSlots; ++i) {
      slot->motion[i] = g_cap.motion11[i] ? static_cast<IUnknown *>(g_cap.motion11[i]) : static_cast<IUnknown *>(g_cap.motion12[i]);
    }
    std::memcpy(slot->mutexes, g_cap.mutexes, sizeof(g_cap.mutexes));
    std::memcpy(slot->motion_mutexes, g_cap.motion_mutexes, sizeof(g_cap.motion_mutexes));
    std::memcpy(slot->handles, g_cap.shared_handles, sizeof(g_cap.shared_handles));
    std::memcpy(slot->motion_handles, g_cap.motion_handles, sizeof(g_cap.motion_handles));
    slot->legacy = g_cap.legacy_handles[0] != 0;
    std::memset(g_cap.textures, 0, sizeof(g_cap.textures));
    std::memset(g_cap.textures12, 0, sizeof(g_cap.textures12));
    std::memset(g_cap.mutexes, 0, sizeof(g_cap.mutexes));
    std::memset(g_cap.shared_handles, 0, sizeof(g_cap.shared_handles));
    std::memset(g_cap.legacy_handles, 0, sizeof(g_cap.legacy_handles));
    std::memset(g_cap.motion11, 0, sizeof(g_cap.motion11));
    std::memset(g_cap.motion_mutexes, 0, sizeof(g_cap.motion_mutexes));
    std::memset(g_cap.motion12, 0, sizeof(g_cap.motion12));
    std::memset(g_cap.motion_handles, 0, sizeof(g_cap.motion_handles));
    std::memset(g_cap.motion_legacy, 0, sizeof(g_cap.motion_legacy));
    g_cap.motion_format = DXGI_FORMAT_UNKNOWN;
    g_cap.motion_width = g_cap.motion_height = 0;
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
    g_motion_refused = motion_want_t {};  // (another device may take that shape)
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

  // A swapchain's colour space, read from the swapchain itself. DXGI has no
  // public getter, and the hook usually arrives after the game called
  // SetColorSpace1, but DXGI's swapchains answer through an undocumented
  // interface (ReShade and Special K rely on it; declaration from ReShade).
  // Checked on dxgi.dll 10.0.26100.7309, D3D11 and D3D12: it reports each
  // format's default before any call (0 = sRGB for 8/10-bit, 1 = scRGB for
  // FP16), every successful SetColorSpace1 (an 8-bit chain can be PQ too),
  // and is unchanged by failed ones. Only DXGI's own swapchains reach our
  // detours (install_hooks checks), so one without it means Windows changed
  // under us: fatal, reported to the host, never guessed around.
  // (IDXGISwapChainTest is declared at file scope, above the namespace)
  // {8C803E30-9E41-4DDF-B206-46F28E90E405}
  constexpr GUID kIidSwapChainTest = {0x8c803e30, 0x9e41, 0x4ddf, {0xb2, 0x06, 0x46, 0xf2, 0x8e, 0x90, 0xe4, 0x05}};

  // The colour space as DXGI reports it; false (and the hook fatal) if the
  // swapchain cannot say
  bool dxgi_color_space(IDXGISwapChain *swapchain, DXGI_COLOR_SPACE_TYPE &color_space) {
    IDXGISwapChainTest *test = nullptr;
    const HRESULT hr = swapchain->QueryInterface(kIidSwapChainTest, reinterpret_cast<void **>(&test));
    if (FAILED(hr) || !test) {
      char msg[gc::kErrorLength];
      std::snprintf(msg, sizeof(msg), "DXGI's swapchain no longer answers IDXGISwapChainTest (colour-space getter) [0x%08lx]: this Windows build needs a Vibepollo update", hr);
      set_state(gc::hook_state_e::fatal, msg);
      return false;
    }
    color_space = test->GetColorSpace1();
    test->Release();
    return true;
  }

  std::uint32_t swapchain_color_space(IDXGISwapChain *swapchain) {
    DXGI_COLOR_SPACE_TYPE dxgi = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    if (!dxgi_color_space(swapchain, dxgi)) {
      return static_cast<std::uint32_t>(gc::color_space_e::unknown);
    }
    auto cs = gc::color_space_e::unknown;  // (a colour space we do not convert: the host uses desktop capture)
    if (dxgi == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) {
      cs = gc::color_space_e::srgb;
    } else if (dxgi == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709) {
      cs = gc::color_space_e::scrgb;
    } else if (dxgi == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
      cs = gc::color_space_e::hdr10;
    }
    static std::atomic<int> logged {-1};
    if (logged.exchange(static_cast<int>(dxgi), std::memory_order_relaxed) != static_cast<int>(dxgi)) {
      log("Swapchain %p colour space %d%s", static_cast<void *>(swapchain), static_cast<int>(dxgi), cs == gc::color_space_e::unknown ? " (not converted)" : "");
    }
    return static_cast<std::uint32_t>(cs);
  }

  // Which sharing a texture set may use: any (D3D11), or owner words only
  // with an NT or a global handle (Vulkan imports one of those kinds)
  enum class sharing_e {
    any,
    owner_nt,
    owner_kmt,
  };

  // `terminal`: whether a failure is reported to the host as final (not
  // when the caller still has another sharing kind to try)
  // A generation's motion textures on the D3D11 device, shared like its
  // textures (`nt`) and, with keyed-mutex sync, each with a keyed mutex of
  // its own, taken and given back with the slot's (under owner words the
  // slot's word and the fence cover them). False (nothing created) if the
  // device refuses them.
  bool create_motion11(const motion_want_t &want, bool nt, bool keyed) {
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = want.width;
    desc.Height = want.height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = want.format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = (keyed ? D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX : D3D11_RESOURCE_MISC_SHARED) | (nt ? D3D11_RESOURCE_MISC_SHARED_NTHANDLE : 0);
    HRESULT hr = S_OK;
    for (int i = 0; i < gc::kSlots && SUCCEEDED(hr); ++i) {
      hr = g_cap.device->CreateTexture2D(&desc, nullptr, &g_cap.motion11[i]);
      if (SUCCEEDED(hr) && keyed) {
        hr = g_cap.motion11[i]->QueryInterface(__uuidof(IDXGIKeyedMutex), reinterpret_cast<void **>(&g_cap.motion_mutexes[i]));
      }
      if (SUCCEEDED(hr) && nt) {
        IDXGIResource1 *resource = nullptr;
        hr = g_cap.motion11[i]->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void **>(&resource));
        if (SUCCEEDED(hr)) {
          hr = resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &g_cap.motion_handles[i]);
          resource->Release();
        }
      } else if (SUCCEEDED(hr)) {
        IDXGIResource *resource = nullptr;
        hr = g_cap.motion11[i]->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void **>(&resource));
        HANDLE handle = nullptr;
        if (SUCCEEDED(hr)) {
          hr = resource->GetSharedHandle(&handle);
          resource->Release();
        }
        if (SUCCEEDED(hr) && !handle) {
          hr = E_FAIL;
        }
        g_cap.motion_legacy[i] = reinterpret_cast<std::uint64_t>(handle);
      }
    }
    if (FAILED(hr)) {
      for (int i = 0; i < gc::kSlots; ++i) {
        safe_release(g_cap.motion_mutexes[i]);
        safe_release(g_cap.motion11[i]);
      }
      close_handles(g_cap.motion_handles);
      std::memset(g_cap.motion_legacy, 0, sizeof(g_cap.motion_legacy));
      log("Motion textures %ux%u format %d could not be created (0x%08lx): frames go without motion vectors", want.width, want.height, want.format, hr);
      return false;
    }
    g_cap.motion_format = want.format;
    g_cap.motion_width = want.width;
    g_cap.motion_height = want.height;
    return true;
  }

  bool ensure_textures(const D3D11_TEXTURE2D_DESC &back, HWND hwnd, sharing_e sharing = sharing_e::any, bool terminal = true, const motion_want_t &motion = {}) {
    const DXGI_FORMAT format = shareable_format(back.Format);
    if (g_cap.textures[0] && g_cap.width == back.Width && g_cap.height == back.Height && g_cap.format == format && motion_satisfied(motion)) {
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
      if (sharing != sharing_e::any) {
        // (Vulkan: its own fence says when a copy is done)
        if (v.keyed || v.nt != (sharing == sharing_e::owner_nt)) {
          continue;
        }
      } else if (!v.keyed && !have_fence) {
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
      if (terminal) {
        set_state(gc::hook_state_e::failed, msg);
      } else {
        log("%s (another sharing kind is tried next)", msg);
      }
      return false;
    }
    if (used != &variants[0]) {
      log("Capture textures use %s sharing%s (device FL %x, flags %x)", used->name, g_cap.owner_sync ? " without keyed mutexes (owner words)" : "", g_cap.device->GetFeatureLevel(), g_cap.device->GetCreationFlags());
    }

    g_cap.width = back.Width;
    g_cap.height = back.Height;
    g_cap.format = format;
    if (motion.format != DXGI_FORMAT_UNKNOWN && !(motion == g_motion_refused) && !create_motion11(motion, used->nt, used->keyed)) {
      g_motion_refused = motion;
    }
    if (g_current_generation.load(std::memory_order_acquire) == 0) {
      g_current_generation.store(1, std::memory_order_release);  // 0 means "nothing set up" to the host
    }
    publish_setup(hwnd, true);
    log("Capture textures %ux%u format %d, generation %u%s", back.Width, back.Height, format, g_current_generation.load(),
        g_cap.motion_format != DXGI_FORMAT_UNKNOWN ? " (with motion textures)" : "");
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
  // `motion`: the vectors copied into the slot's motion texture with its
  // pixels (null: none)
  std::uint32_t write_slot_record(int slot, std::uint32_t generation, std::uint32_t color_space, std::uint64_t frame_id,
                                  std::uint64_t present_qpc, std::uint64_t release_qpc, const motion_pick_t *motion = nullptr) {
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
    const auto bits = [](float f) {
      std::uint32_t u;
      std::memcpy(&u, &f, sizeof(u));
      return u;
    };
    record.motion_id.store(motion ? motion->id : 0, std::memory_order_relaxed);
    record.motion_width.store(motion ? motion->width : 0, std::memory_order_relaxed);
    record.motion_height.store(motion ? motion->height : 0, std::memory_order_relaxed);
    record.motion_out_width.store(motion ? motion->out_width : 0, std::memory_order_relaxed);
    record.motion_out_height.store(motion ? motion->out_height : 0, std::memory_order_relaxed);
    record.motion_scale_x.store(motion ? bits(motion->scale_x) : 0, std::memory_order_relaxed);
    record.motion_scale_y.store(motion ? bits(motion->scale_y) : 0, std::memory_order_relaxed);
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
  // kQueueUnseenMs is reported unsupported. A swapchain may present each back
  // buffer from its own queue (ResizeBuffers1), which may have been set up
  // before we were injected, so a learned queue is trusted only once it was
  // seen in as many Presents as there are buffers; capture starts after. An
  // observation is thrown away if the Present failed or another swapchain
  // presented inside it. ResizeBuffers1 names the swapchain's queues, so it
  // sets the queue itself (or vetoes the swapchain if they differ); a plain
  // ResizeBuffers forgets it, to be learned again. Either way no copy is
  // submitted on a queue the swapchain no longer presents from.
  //
  // Overlays (Steam's, say) draw into the back buffer from their own Present
  // hook, which sits behind ours when we are injected after them, so a copy
  // taken before the real Present misses them. DXGI's own presentation work
  // is the last submission on the presenting queue inside Present, made from
  // dxgi.dll (checked on this machine: an overlay's draw, then DXGI's, one
  // each per Present). So the copy is armed before the real Present and
  // submitted from our ExecuteCommandLists detour just ahead of DXGI's
  // submission, when the frame is final. Only DXGI's submissions teach the
  // queue, too, so an overlay's own queue is no ambiguity. A swapchain whose
  // Presents stop showing DXGI's submission falls back to copying before the
  // real Present (overlays then missed), after kSubmitMisses in a row.
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
  thread_local ID3D12CommandQueue *t_seen_dxgi_queue = nullptr;  // ... of the submissions made by dxgi.dll itself (a reference)
  thread_local bool t_seen_dxgi_several = false;
  thread_local UINT t_present_index = UINT_MAX;  // the back buffer this Present shows, read before the real call

  // A recorded D3D12 copy (capture_frame12), not yet submitted. Its slot's
  // owner word is held by the hook until it is submitted or given back.
  struct prepared12_t {
    bool valid = false;
    int slot = -1;
    ID3D12GraphicsCommandList *list = nullptr;  // g_cap's (borrowed): checked unchanged before use
    ID3D12CommandQueue *queue = nullptr;  // g_cap.queue when recorded
    ID3D12Fence *fence = nullptr;  // g_cap.fence12 when recorded
    std::uint32_t version = 0;
    std::uint32_t generation = 0;
    std::uint64_t frame_id = 0;
    // The frame's DLSS vectors in the list: the ring entry read (reserved
    // until submitted or given back)
    std::uint32_t motion_ring = 0;
    int motion_entry = -1;
  };

  // A copy recorded before the real Present, waiting for DXGI's own
  // submission inside it. Everything that touches the swapchain or DXGI state
  // happened before the Present; inside DXGI's call only the recorded list is
  // executed and the fence signalled.
  struct armed_capture_t {
    bool armed = false;  // a copy was prepared this Present (stays set when it is cancelled: it still counts)
    bool boundary_seen = false;  // DXGI submitted on the copy's queue during this Present, uncontaminated
    IDXGISwapChain *swapchain = nullptr;  // borrowed: we are inside its Present
    HWND hwnd = nullptr;
    prepared12_t prepared;  // valid until submitted or given back
  };

  thread_local armed_capture_t t_armed;

  enum class submit_mode_e : std::uint32_t {
    before_present = 0,  ///< copy before the real Present (overlays drawn after ours are missed)
    at_dxgi_submit = 1,  ///< D3D12: copy just ahead of DXGI's own submission (overlays included)
    abandoned = 2,  ///< the late copy point stopped being reached: before_present for good
    at_present_impl = 3,  ///< D3D11: copy at dxgi!CDXGISwapChain::PresentImpl (overlays included)
  };

  constexpr std::uint32_t kSubmitMisses = 3;

  HMODULE g_dxgi_module = nullptr;  // Windows' dxgi.dll (set by install_hooks)
  bool g_vk_layer_present = false;  // Vibepollo's Vulkan layer is loaded here (set before install_hooks)

  bool called_from_dxgi(void *return_address) {
    HMODULE module = nullptr;
    return g_dxgi_module &&
           GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<LPCWSTR>(return_address), &module) &&
           module == g_dxgi_module;
  }

  void run_armed_capture(ID3D12CommandQueue *queue);
  void disarm_capture();
  bool motion_submissions_pending();  // (motion section)
  void motion_note_submission(ID3D12CommandQueue *queue, UINT count, ID3D12CommandList *const *lists);
  void motion_note_reset(const void *list);
  void cancel_prepared11_for(IDXGISwapChain *swapchain);  // (PresentImpl section)
  void give_back_armed11();

  // Set when a swapchain veto could not be recorded, or a Signal failed:
  // D3D12 capture stops for the process (what our copies used is leaked)
  std::atomic<bool> g_d3d12_dead {false};

  void STDMETHODCALLTYPE hook_execute_command_lists(ID3D12CommandQueue *queue, UINT count, ID3D12CommandList *const *lists) {
    // Only direct queues can present; copy/compute submissions inside
    // Present (an overlay's uploads, say) are not candidates
    if (t_presenting && queue->GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
      const bool dxgi = called_from_dxgi(__builtin_return_address(0));
      auto &seen = dxgi ? t_seen_dxgi_queue : t_seen_queue;
      auto &several = dxgi ? t_seen_dxgi_several : t_seen_several;
      if (!seen) {
        queue->AddRef();
        seen = queue;
      } else if (seen != queue) {
        several = true;
      }
      if (dxgi && t_armed.armed && queue == t_armed.prepared.queue) {
        run_armed_capture(queue);  // our copy goes on the queue just ahead of DXGI's work
      }
    }
    g_real_execute_command_lists(queue, count, lists);
    if (motion_submissions_pending()) {
      motion_note_submission(queue, count, lists);  // (any queue, any thread: a DLSS evaluation's copy may be in these)
    }
  }

  // ID3D12GraphicsCommandList::Reset (IUnknown 0-2, ID3D12Object 3-6,
  // ID3D12DeviceChild 7, ID3D12CommandList::GetType 8, Close 9, Reset 10):
  // a recording that holds a DLSS evaluation's copy and is reset unsubmitted
  // will never run it
  constexpr int kVtListReset = 10;
  using list_reset_fn = HRESULT(STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, ID3D12CommandAllocator *, ID3D12PipelineState *);
  list_reset_fn g_real_list_reset = nullptr;

  HRESULT STDMETHODCALLTYPE hook_list_reset(ID3D12GraphicsCommandList *list, ID3D12CommandAllocator *allocator, ID3D12PipelineState *state) {
    const HRESULT hr = g_real_list_reset(list, allocator, state);
    if (SUCCEEDED(hr) && motion_submissions_pending()) {
      motion_note_reset(list);
    }
    return hr;
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

  // Back buffers confirmed presented from the stored queue (bit per index; 0:
  // none stored or unreadable)
  std::uint32_t queue_seen(IDXGISwapChain *swapchain) {
    std::uint32_t value = 0;
    UINT size = sizeof(value);
    return SUCCEEDED(swapchain->GetPrivateData(kQueueSeenKey, &size, &value)) && size == sizeof(value) ? value : 0;
  }

  bool set_queue_seen(IDXGISwapChain *swapchain, std::uint32_t value) {
    return SUCCEEDED(swapchain->SetPrivateData(kQueueSeenKey, sizeof(value), &value));
  }

  // Drops the swapchain's queue, to be learned again
  void forget_queue(IDXGISwapChain *swapchain) {
    if (!set_queue_seen(swapchain, 0) || FAILED(swapchain->SetPrivateDataInterface(kQueueKey, nullptr))) {
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

  // Stores `queue` as seen `seen` times; false (and nothing stored) if the
  // swapchain would not take it
  bool remember_queue(IDXGISwapChain *swapchain, ID3D12CommandQueue *queue, std::uint32_t seen, const char *how) {
    if (!set_queue_seen(swapchain, 0) || FAILED(swapchain->SetPrivateDataInterface(kQueueKey, queue))) {
      return false;
    }
    if (!set_queue_seen(swapchain, seen)) {
      return false;
    }
    log("Swapchain %p presents from D3D12 queue %p (%s)", static_cast<void *>(swapchain), static_cast<void *>(queue), how);
    return true;
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
  std::uint32_t swapchain_count(IDXGISwapChain *swapchain, const GUID &key) {
    std::uint32_t value = 0;
    UINT size = sizeof(value);
    return SUCCEEDED(swapchain->GetPrivateData(key, &size, &value)) && size == sizeof(value) ? value : 0;
  }

  // Set when a submit-mode write failed: the miss count cannot be trusted, so
  // no swapchain copies at DXGI's submission any more
  std::atomic<bool> g_submit_mode_broken {false};

  void set_swapchain_count(IDXGISwapChain *swapchain, const GUID &key, std::uint32_t value) {
    if (FAILED(swapchain->SetPrivateData(key, sizeof(value), &value)) && &key == &kSubmitModeKey) {
      g_submit_mode_broken.store(true, std::memory_order_release);
      log("Swapchain %p: its copy mode could not be recorded; copying before Present from now on", static_cast<void *>(swapchain));
    }
  }

  submit_mode_e submit_mode(IDXGISwapChain *swapchain) {
    if (g_submit_mode_broken.load(std::memory_order_acquire)) {
      return submit_mode_e::abandoned;
    }
    return static_cast<submit_mode_e>(swapchain_count(swapchain, kSubmitModeKey) & 0xff);
  }

  void learn_present_queue(IDXGISwapChain *swapchain) {
    // Only DXGI's own submissions (made from dxgi.dll inside Present) teach
    // the queue: another direct submission there is an overlay's, and an
    // overlay may use a queue of its own. A Present showing none teaches
    // nothing (after kQueueUnseenMs the swapchain is reported unsupported).
    safe_release(t_seen_queue);
    const bool from_dxgi = t_seen_dxgi_queue != nullptr;
    ID3D12CommandQueue *seen = t_seen_dxgi_queue;
    t_seen_several = t_seen_dxgi_several;
    t_seen_dxgi_queue = nullptr;
    if (vetoed(swapchain)) {
      safe_release(seen);
      return;
    }
    ID3D12CommandQueue *known = known_queue(swapchain);
    if (t_seen_several || (known && known != seen)) {
      veto(swapchain, "presents from more than one D3D12 queue");
    } else if (known) {
      // This Present's buffer is confirmed: DXGI itself submitted it on the
      // queue (a swapchain may present each buffer from its own queue)
      const auto mask = queue_seen(swapchain);
      const auto bit = t_present_index < 32 ? 1u << t_present_index : 0u;
      if (bit && !(mask & bit) && !set_queue_seen(swapchain, mask | bit)) {
        forget_queue(swapchain);
      }
      if (from_dxgi && submit_mode(swapchain) == submit_mode_e::before_present) {
        set_swapchain_count(swapchain, kSubmitModeKey, static_cast<std::uint32_t>(submit_mode_e::at_dxgi_submit));
        log("Swapchain %p: DXGI submits on its queue from dxgi.dll; frames are copied just ahead of that (overlays included)", static_cast<void *>(swapchain));
      }
    } else if (!usable_queue(swapchain, seen)) {
      veto(swapchain, "presents from a D3D12 queue of another device or node");
    } else if (!remember_queue(swapchain, seen, t_present_index < 32 ? 1u << t_present_index : 0u, "DXGI's own submission inside Present")) {
      forget_queue(swapchain);
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
    // (done even for a vetoed swapchain: a veto that could not be read now
    // must not leave the old queue behind for a later read)
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
    } else if (!remember_queue(swapchain, queue, UINT32_MAX, "named by ResizeBuffers1")) {
      forget_queue(swapchain);
      veto(swapchain, "was resized onto a D3D12 queue that could not be recorded");
    }
    safe_release(queue);
  }

  // A prepared copy references the swapchain's current buffers: a resize
  // (an overlay may resize from its own Present hook, inside ours) voids it
  void cancel_prepared_for(IDXGISwapChain *swapchain);

  HRESULT STDMETHODCALLTYPE hook_resize_buffers(IDXGISwapChain *swapchain, UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags) {
    cancel_prepared_for(swapchain);
    const HRESULT hr = g_real_resize_buffers(swapchain, count, width, height, format, flags);
    if (SUCCEEDED(hr)) {
      note_resize(swapchain, 0, nullptr);
    }
    return hr;
  }

  HRESULT STDMETHODCALLTYPE hook_resize_buffers1(IDXGISwapChain3 *swapchain, UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags,
                                                 const UINT *node_masks, IUnknown *const *queues) {
    cancel_prepared_for(swapchain);
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

  // D3D12 motion textures of a generation: shared, simultaneous-access (the
  // copy promotes them and they decay to COMMON, like the frame textures).
  // False (nothing created) if the device refuses them.
  bool create_motion12(const motion_want_t &want) {
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = want.width;
    desc.Height = want.height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = want.format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;  // (see ensure_textures12)
    HRESULT hr = S_OK;
    for (int i = 0; i < gc::kSlots && SUCCEEDED(hr); ++i) {
      hr = g_cap.device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   __uuidof(ID3D12Resource), reinterpret_cast<void **>(&g_cap.motion12[i]));
      if (SUCCEEDED(hr)) {
        hr = g_cap.device12->CreateSharedHandle(g_cap.motion12[i], nullptr, GENERIC_ALL, nullptr, &g_cap.motion_handles[i]);
      }
    }
    if (FAILED(hr)) {
      for (int i = 0; i < gc::kSlots; ++i) {
        safe_release(g_cap.motion12[i]);
      }
      close_handles(g_cap.motion_handles);
      log("D3D12 motion textures %ux%u format %d could not be created (0x%08lx): frames go without motion vectors", want.width, want.height, want.format, hr);
      return false;
    }
    g_cap.motion_format = want.format;
    g_cap.motion_width = want.width;
    g_cap.motion_height = want.height;
    return true;
  }

  bool ensure_textures12(const D3D12_RESOURCE_DESC &back, HWND hwnd, const motion_want_t &motion = {}) {
    const DXGI_FORMAT format = shareable_format(back.Format);
    const auto width = static_cast<UINT>(back.Width);
    if (g_cap.textures12[0] && g_cap.width == width && g_cap.height == back.Height && g_cap.format == format && motion_satisfied(motion)) {
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
    // D3D11 opens a D3D12 shared texture only if it allows render targets
    // (OpenSharedResource1 is E_INVALIDARG otherwise, whatever the format:
    // tested on genwin); simultaneous access spares us barriers on it
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

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
    if (motion.format != DXGI_FORMAT_UNKNOWN && !(motion == g_motion_refused) && !create_motion12(motion)) {
      g_motion_refused = motion;
    }
    if (g_current_generation.load(std::memory_order_acquire) == 0) {
      g_current_generation.store(1, std::memory_order_release);
    }
    publish_setup(hwnd, true);
    log("D3D12 capture textures %ux%u format %d, generation %u%s", width, back.Height, format, g_current_generation.load(),
        g_cap.motion_format != DXGI_FORMAT_UNKNOWN ? " (with motion textures)" : "");
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

  // ---- DLSS motion vectors -------------------------------------------------
  //
  // Games that use DLSS hand it, every frame, screen-space motion vectors
  // ("the location the pixel occupied in the previous frame" = pixel +
  // vector, in pixels of the region they cover, times MV.Scale). They reach
  // the driver's NGX (_nvngx.dll) through its EvaluateFeature entry points,
  // whichever SDK or Streamline sits in between, so those are detoured
  // (once _nvngx.dll is loaded, by a watcher thread; the module is pinned).
  // CreateFeature is detoured too: the feature's creation flags and sizes
  // say how its vectors are laid out.
  //
  // After a successful evaluation of an upscaling feature (super sampling or
  // ray reconstruction) whose layout we know (creation flags known, vectors
  // not jittered, no output subrectangle), its motion-vector region is copied,
  // on the game's own command list (D3D12: after the evaluation's work; the
  // vectors are in NON_PIXEL_SHADER_RESOURCE there, which DLSS requires; a
  // simultaneous-access vector texture is declined, its state history is not
  // ours to know) or immediate context (D3D11; not on a single-threaded
  // device, whose objects we could end up releasing on another thread), into
  // a ring of hook-owned textures.
  //
  // Which evaluation belongs to which presented frame: every Present of the
  // captured swapchain consumes one main-view evaluation (the largest output:
  // a game may upscale a second view too), the oldest submitted, whether or
  // not the frame is captured. Evaluations are numbered when they are
  // executed. An engine that submits the next frame's work (its evaluation
  // included) before presenting this one has two pending at each Present,
  // and this frame's is the older; one that does not has one: the backlog
  // stays what the engine keeps in flight, with nothing learned or guessed.
  // A Present with none pending consumes none (frame generation's extra
  // frames, menus), and one that finds more than three drops them all. Each
  // Present advances the motion sequence frames report, so the host sees a
  // gap wherever a frame with vectors is not the one it had before.
  //
  // D3D12 lifetime and order: a ring entry is written when the game executes
  // the evaluation's list; the ExecuteCommandLists detour, after the real
  // call, signals that queue's own fence of the ring (one fence per queue:
  // each is one ordered timeline) and records the value as the entry's
  // producer. The capture copy reads the entry on the presenting queue, so
  // an evaluation executed on another queue is taken only once its producer
  // value has completed (we never make the presenting queue wait for another
  // one: that could deadlock a game whose other queue waits for this one).
  // The capture fence value after the copy is the entry's consumer. An entry
  // is reused only once its producer and consumer completed (none free: the
  // evaluation goes without a copy), and reusing it forgets every record that
  // pointed at it; a replaced ring is released only when all its entries are
  // free. A recording reset unsubmitted (the Reset detour, after a successful
  // Reset) frees its entry; one we lose track of otherwise keeps its ring
  // alive for good. The bookkeeping is lossless: the motion lock is held only
  // for short, allocation-free stretches (textures are created outside it)
  // and taken by everyone.
  // Known limit: a closed list the game executes a second time replays its
  // copy into an entry that may by then be another evaluation's (games
  // re-record their DLSS lists every frame).
  //
  // D3D11: the immediate context orders everything by call, so an entry can
  // be neither overwritten nor read early; the runtime keeps what the GPU
  // uses alive.
  //
  // Motion capture runs only while the host streams (capture enabled, motion
  // wanted, its heartbeat fresh).

  constexpr int kMotionRing = 4;
  constexpr int kMotionEvals = 8;  // evaluations remembered (D3D12: until submitted)
  constexpr int kMotionOldRings = 4;
  constexpr int kMotionFeatures = 16;
  constexpr int kMotionQueues = 8;  // D3D12 queues the game executes evaluations on
  constexpr std::uint32_t kNgxFeatureSuperSampling = 1;
  constexpr std::uint32_t kNgxFeatureRayReconstruction = 13;
  constexpr int kNgxFlagMvLowRes = 1 << 1;  // DLSS.Feature.Create.Flags: vectors at render resolution
  constexpr int kNgxFlagMvJittered = 1 << 2;  // ... that include the camera jitter

  // A D3D12 queue the game executed evaluations on, with a fence of ours
  // for it (one ordered timeline per queue), shared by the rings; a slot is
  // reclaimed once its fence completed and no entry refers to it
  struct motion_queue_t {
    ID3D12CommandQueue *queue = nullptr;  // a reference
    ID3D12Fence *fence = nullptr;
    std::uint64_t value = 0;  // the last value signalled (under the motion lock, so values follow enqueue order)
  };

  // One D3D12 ring texture's use
  struct motion_entry_t {
    std::uint64_t writer = 0;  // the evaluation whose copy writes it (0 = none)
    int queue = -1;  // the producer queue (index into g_motion_queues; -1 = not submitted yet)
    std::uint64_t produced = 0;  // ... and its fence value after that submission
    std::uint64_t consumed = 0;  // the capture fence value after the last capture copy that read it
    ID3D12Fence *consumer = nullptr;  // that capture fence (a reference)
    int readers = 0;  // capture copies prepared from it, not yet submitted or given back
  };

  struct motion_ring_t {
    gc::api_e api = gc::api_e::unknown;
    IUnknown *device = nullptr;  // a reference (ID3D11Device / ID3D12Device)
    motion_want_t desc;  // format and size of each texture
    ID3D11Texture2D *tex11[kMotionRing] = {};
    ID3D12Resource *tex12[kMotionRing] = {};
    motion_entry_t entries[kMotionRing];
    std::uint32_t serial = 0;  // identifies the ring (0 = none)
    bool tainted = false;  // a recording into it was lost track of: kept for good

    void release() {
      for (int i = 0; i < kMotionRing; ++i) {
        safe_release(tex11[i]);
        safe_release(tex12[i]);
        safe_release(entries[i].consumer);
      }
      safe_release(device);
      *this = motion_ring_t {};
    }
  };

  struct motion_eval_t {
    std::uint64_t id = 0;  // 0 = empty
    const void *list = nullptr;  // D3D12: the command list it was recorded into (identity)
    bool submitted = false;
    std::uint64_t order = 0;  // its execution's number (submission order; 0 = not yet)
    std::uint32_t ring = 0;  // the serial of the ring it writes
    int entry = 0;
    UINT width = 0;
    UINT height = 0;
    UINT out_width = 0;
    UINT out_height = 0;
    float scale_x = 1;
    float scale_y = 1;
  };

  // What CreateFeature told us about a feature
  struct motion_feature_t {
    const void *handle = nullptr;
    std::uint32_t id = 0;
    bool have_flags = false;
    int flags = 0;
    std::uint32_t width = 0;  // render size (0 = unknown)
    std::uint32_t height = 0;
    std::uint32_t out_width = 0;  // output size (0 = unknown)
    std::uint32_t out_height = 0;
  };

  // Counters for the periodic log line (motion lock)
  struct motion_stats_t {
    std::uint64_t pending[5] = {};  // Presents that found 0, 1, 2, 3, 4+ evaluations pending
    std::uint64_t picked = 0;  // captured frames that took their vectors
    std::uint64_t dropped = 0;  // evaluations dropped by a Present that found more than three
    std::uint64_t unfinished = 0;  // a frame's, executed on another queue and not finished yet: no vectors
    std::uint64_t no_entry = 0;  // evaluations that found no free ring entry
    std::uint64_t evaluations = 0;
    std::uint64_t logged_qpc = 0;
  };

  SRWLOCK g_motion_lock = SRWLOCK_INIT;
  motion_ring_t g_motion_ring;  // the current device's
  motion_ring_t g_motion_old[kMotionOldRings];  // replaced rings, until idle
  motion_eval_t g_motion_evals[kMotionEvals];
  motion_feature_t g_motion_features[kMotionFeatures];
  motion_stats_t g_motion_stats;
  motion_queue_t g_motion_queues[kMotionQueues];
  motion_ring_t g_motion_dead[kMotionOldRings];  // idle replaced rings, released outside the lock
  std::uint64_t g_motion_next_id = 1;
  std::uint64_t g_motion_order = 0;  // the last execution number handed out
  std::uint64_t g_motion_consumed = 0;  // evaluations executed up to this number are consumed by Presents
  std::uint64_t g_motion_seq = 0;  // +1 per Present of the captured swapchain: what captured frames report
  std::uint64_t g_motion_frame_eval = 0;  // the evaluation the current Present consumed (0 = none) ...
  std::uint64_t g_motion_frame_seq = 0;  // ... and that Present's number
  std::uint32_t g_motion_ring_serial = 0;
  std::atomic<int> g_motion_unsubmitted {0};  // D3D12 evaluations recorded, not yet seen submitted (the detours' fast path)

  bool motion_wanted() {
    if (!g_block || g_block->capture_enabled.load(std::memory_order_acquire) == 0 || g_block->motion_enabled.load(std::memory_order_acquire) == 0) {
      return false;
    }
    const auto heartbeat = g_block->host_heartbeat_qpc.load(std::memory_order_acquire);
    const auto now = qpc_now();
    return heartbeat && now >= heartbeat && now - heartbeat <= ms_to_qpc(gc::kHeartbeatTimeoutMs);
  }

  // The motion lock: every holder keeps it for a short, allocation-free
  // stretch (no texture creation, no GPU waits), so everyone may block on it
  struct motion_lock_t {
    motion_lock_t() {
      AcquireSRWLockExclusive(&g_motion_lock);
    }

    ~motion_lock_t() {
      ReleaseSRWLockExclusive(&g_motion_lock);
    }

    motion_lock_t(const motion_lock_t &) = delete;
    motion_lock_t &operator=(const motion_lock_t &) = delete;
  };

  // NVSDK_NGX_Parameter is an MSVC-compiled interface: its overloaded
  // virtuals sit in reverse declaration order, the eight Set overloads in
  // slots 0-7, the Get overloads in 8-15 (void**, ID3D12Resource**,
  // ID3D11Resource**, int*, unsigned*, double*, float*, unsigned long long*).
  // A Get is a plain x64 call with `this` first.
  enum ngx_get_e : int {
    kNgxGetVoid = 8,
    kNgxGetD3d12 = 9,
    kNgxGetD3d11 = 10,
    kNgxGetInt = 11,
    kNgxGetUInt = 12,
    kNgxGetFloat = 14,
  };

  bool ngx_ok(std::uint32_t result) {
    return (result & 0xfff00000u) != 0xbad00000u;
  }

  template<class T>
  bool ngx_get(const void *params, ngx_get_e slot, const char *key, T &out) {
    using get_fn = std::uint32_t (*)(const void *, const char *, T *);
    void *const *vtable = *static_cast<void *const *const *>(params);
    T value {};
    if (!ngx_ok(reinterpret_cast<get_fn>(vtable[slot])(params, key, &value))) {
      return false;
    }
    out = value;
    return true;
  }

  // An unsigned parameter by its name or its encoded name
  bool ngx_uint(const void *params, const char *key, const char *encoded, std::uint32_t &out) {
    return ngx_get(params, kNgxGetUInt, key, out) || (encoded && ngx_get(params, kNgxGetUInt, encoded, out));
  }

  // A resource parameter, by its name or its encoded name, typed or not
  void *ngx_resource(const void *params, ngx_get_e typed, const char *key, const char *encoded) {
    void *r = nullptr;
    if ((ngx_get(params, typed, key, r) && r) || (ngx_get(params, typed, encoded, r) && r) ||
        (ngx_get(params, kNgxGetVoid, key, r) && r) || (ngx_get(params, kNgxGetVoid, encoded, r) && r)) {
      return r;
    }
    return nullptr;
  }

  // Float formats DLSS takes vectors in, as the typed format of our copies
  DXGI_FORMAT motion_format(DXGI_FORMAT f) {
    switch (f) {
      case DXGI_FORMAT_R16G16_FLOAT:
      case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;
      case DXGI_FORMAT_R32G32_FLOAT:
      case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;
      default:
        return DXGI_FORMAT_UNKNOWN;
    }
  }

  void log_once(std::atomic<bool> &flag, const char *what) {
    if (!flag.exchange(true, std::memory_order_relaxed)) {
      log("DLSS motion vectors: %s", what);
    }
  }

  // What an evaluation's parameters say, read before the motion lock
  struct motion_params_t {
    bool have_flags = false;
    int flags = 0;
    std::uint32_t base_x = 0, base_y = 0;  // of the vectors in their texture
    std::uint32_t subrect_w = 0, subrect_h = 0;  // the render subrectangle (0 = not given)
    std::uint32_t width = 0, height = 0;  // render size (0 = not given)
    std::uint32_t out_width = 0, out_height = 0;  // output size (0 = not given)
    std::uint32_t out_base_x = 0, out_base_y = 0;
    float scale_x = 1, scale_y = 1;
    bool color = false;  // an upscaler's colour input is there
  };

  void read_motion_params(const void *params, ngx_get_e typed, motion_params_t &p) {
    p.have_flags = ngx_get(params, kNgxGetInt, "DLSS.Feature.Create.Flags", p.flags);
    ngx_uint(params, "DLSS.Input.MV.Subrect.Base.X", nullptr, p.base_x);
    ngx_uint(params, "DLSS.Input.MV.Subrect.Base.Y", nullptr, p.base_y);
    if (!ngx_uint(params, "DLSS.Render.Subrect.Dimensions.Width", nullptr, p.subrect_w) ||
        !ngx_uint(params, "DLSS.Render.Subrect.Dimensions.Height", nullptr, p.subrect_h)) {
      p.subrect_w = p.subrect_h = 0;
    }
    if (!ngx_uint(params, "Width", "#\x10", p.width) || !ngx_uint(params, "Height", "#\x11", p.height)) {
      p.width = p.height = 0;
    }
    if (!ngx_uint(params, "OutWidth", "#\x12", p.out_width) || !ngx_uint(params, "OutHeight", "#\x13", p.out_height)) {
      p.out_width = p.out_height = 0;
    }
    ngx_uint(params, "DLSS.Output.Subrect.Base.X", nullptr, p.out_base_x);
    ngx_uint(params, "DLSS.Output.Subrect.Base.Y", nullptr, p.out_base_y);
    if (!(ngx_get(params, kNgxGetFloat, "MV.Scale.X", p.scale_x) || ngx_get(params, kNgxGetFloat, "#\x2c", p.scale_x))) {
      p.scale_x = 1;
    }
    if (!(ngx_get(params, kNgxGetFloat, "MV.Scale.Y", p.scale_y) || ngx_get(params, kNgxGetFloat, "#\x2d", p.scale_y))) {
      p.scale_y = 1;
    }
    p.color = ngx_resource(params, typed, "Color", "#\x1e") != nullptr;
  }

  // Motion lock held
  motion_feature_t *motion_feature(const void *handle) {
    for (auto &f : g_motion_features) {
      if (f.handle && f.handle == handle) {
        return &f;
      }
    }
    return nullptr;
  }

  // Where the vectors lie and what they span
  struct motion_region_t {
    UINT x = 0, y = 0, width = 0, height = 0;  // in the vector texture
    UINT out_width = 0, out_height = 0;
    float scale_x = 1, scale_y = 1;
  };

  std::atomic<bool> g_logged_no_flags {false}, g_logged_jittered {false}, g_logged_subrect {false}, g_logged_no_size {false},
    g_logged_simultaneous {false}, g_logged_single_threaded {false}, g_logged_flags_disagree {false};

  // Motion lock held. The region of an upscaler evaluation's vectors, or
  // false when the evaluation is not an upscaler's or its layout is not one
  // we can be sure of (declined, never guessed)
  bool motion_region(const void *handle, const motion_params_t &p, UINT mv_width, UINT mv_height, UINT out_res_width, UINT out_res_height, motion_region_t &r) {
    const motion_feature_t *f = motion_feature(handle);
    // An upscaler: by the feature its handle was created for, or, for one
    // created before we came, by its inputs (upscalers take colour)
    if (f ? (f->id != kNgxFeatureSuperSampling && f->id != kNgxFeatureRayReconstruction) : !p.color) {
      return false;
    }
    // The feature's own creation flags first (a parameter object may have
    // been reused to create another feature since); evaluation parameters
    // that say otherwise make the layout uncertain
    const bool have_flags = (f && f->have_flags) || p.have_flags;
    const int flags = f && f->have_flags ? f->flags : p.flags;
    if (!have_flags) {
      log_once(g_logged_no_flags, "an evaluation without its creation flags (vector resolution unknown) is not used");
      return false;
    }
    if (f && f->have_flags && p.have_flags && ((f->flags ^ p.flags) & (kNgxFlagMvLowRes | kNgxFlagMvJittered))) {
      log_once(g_logged_flags_disagree, "an evaluation's flags contradict its feature's creation flags; not used");
      return false;
    }
    if (flags & kNgxFlagMvJittered) {
      log_once(g_logged_jittered, "the game's vectors include the camera jitter; not used");
      return false;
    }
    if (p.out_base_x || p.out_base_y) {
      log_once(g_logged_subrect, "DLSS writes a subrectangle of its output; not used");
      return false;
    }
    const UINT out_w = f && f->out_width ? f->out_width : p.out_width ? p.out_width : out_res_width;
    const UINT out_h = f && f->out_height ? f->out_height : p.out_height ? p.out_height : out_res_height;
    UINT w = out_w, h = out_h;  // (full-resolution vectors span the output)
    if (flags & kNgxFlagMvLowRes) {
      w = p.subrect_w ? p.subrect_w : p.width ? p.width : f ? f->width : 0;
      h = p.subrect_h ? p.subrect_h : p.height ? p.height : f ? f->height : 0;
    }
    if (!w || !h || !out_w || !out_h || p.base_x >= mv_width || p.base_y >= mv_height || w > mv_width - p.base_x || h > mv_height - p.base_y) {
      log_once(g_logged_no_size, "an evaluation whose vector region is unknown or outside its texture is not used");
      return false;
    }
    if (!std::isfinite(p.scale_x) || !std::isfinite(p.scale_y) || p.scale_x == 0 || p.scale_y == 0) {
      return false;
    }
    r.x = p.base_x;
    r.y = p.base_y;
    r.width = w;
    r.height = h;
    r.out_width = out_w;
    r.out_height = out_h;
    r.scale_x = p.scale_x;
    r.scale_y = p.scale_y;
    return true;
  }

  void motion_note_feature(const void *handle, std::uint32_t feature, const void *params) {
    if (!handle) {
      return;
    }
    motion_feature_t f;
    f.handle = handle;
    f.id = feature;
    if (params) {
      f.have_flags = ngx_get(params, kNgxGetInt, "DLSS.Feature.Create.Flags", f.flags);
      if (!ngx_uint(params, "Width", "#\x10", f.width) || !ngx_uint(params, "Height", "#\x11", f.height)) {
        f.width = f.height = 0;
      }
      if (!ngx_uint(params, "OutWidth", "#\x12", f.out_width) || !ngx_uint(params, "OutHeight", "#\x13", f.out_height)) {
        f.out_width = f.out_height = 0;
      }
    }
    motion_lock_t lock;
    motion_feature_t *slot = motion_feature(handle);
    for (int i = 0; i < kMotionFeatures && !slot; ++i) {
      if (!g_motion_features[i].handle) {
        slot = &g_motion_features[i];
      }
    }
    if (slot) {
      *slot = f;
    }
  }

  void motion_forget_feature(const void *handle) {
    motion_lock_t lock;
    if (auto *f = motion_feature(handle)) {
      *f = motion_feature_t {};
    }
  }

  // Motion lock held
  motion_ring_t *motion_ring(std::uint32_t serial) {
    if (!serial) {
      return nullptr;
    }
    if (g_motion_ring.serial == serial) {
      return &g_motion_ring;
    }
    for (auto &r : g_motion_old) {
      if (r.serial == serial) {
        return &r;
      }
    }
    return nullptr;
  }

  // Motion lock held. Whether the entry's producer finished (a removed device
  // reports every fence value complete)
  bool motion_produced(const motion_ring_t &, const motion_entry_t &e) {
    return e.queue >= 0 && g_motion_queues[e.queue].fence && g_motion_queues[e.queue].fence->GetCompletedValue() >= e.produced;
  }

  // Motion lock held. Whether nothing writes or reads the entry any more
  bool motion_entry_free(const motion_ring_t &ring, int i) {
    const auto &e = ring.entries[i];
    if (ring.api != gc::api_e::d3d12) {
      return true;
    }
    if (e.readers || (e.writer && !motion_produced(ring, e))) {
      return false;
    }
    return !e.consumer || e.consumer->GetCompletedValue() >= e.consumed;
  }

  bool motion_ring_idle(const motion_ring_t &ring) {
    if (ring.tainted) {
      return false;
    }
    for (int i = 0; i < kMotionRing; ++i) {
      if (!motion_entry_free(ring, i)) {
        return false;
      }
    }
    return true;
  }

  // Motion lock held. Moves replaced rings nothing uses any more to the dead
  // list (released by motion_release_dead, outside the lock); true if a
  // place is free for one more
  bool motion_collect_old() {
    bool room = false;
    for (auto &r : g_motion_old) {
      if (r.serial && motion_ring_idle(r)) {
        for (auto &d : g_motion_dead) {
          if (!d.serial && !d.device) {
            d = r;
            r = motion_ring_t {};
            break;
          }
        }
      }
      room = room || !r.serial;
    }
    return room;
  }

  // No lock held: releases what motion_collect_old set aside
  void motion_release_dead() {
    motion_ring_t dead[kMotionOldRings];
    {
      AcquireSRWLockExclusive(&g_motion_lock);
      for (int i = 0; i < kMotionOldRings; ++i) {
        dead[i] = g_motion_dead[i];
        g_motion_dead[i] = motion_ring_t {};
      }
      ReleaseSRWLockExclusive(&g_motion_lock);
    }
    for (auto &d : dead) {
      if (d.device) {
        d.release();
      }
    }
  }

  // Motion lock held. A record leaves the table unsubmitted: `discarded`
  // (its recording was reset: the copy will never run) frees its entry;
  // otherwise its ring may still be written some day and is kept for good
  void motion_drop_unsubmitted(motion_eval_t &e, bool discarded) {
    if (auto *ring = motion_ring(e.ring)) {
      if (!discarded) {
        ring->tainted = true;
      } else if (ring->entries[e.entry].writer == e.id) {
        ring->entries[e.entry].writer = 0;
      }
    }
    g_motion_unsubmitted.fetch_sub(1, std::memory_order_relaxed);
    e = motion_eval_t {};
  }

  // Motion lock held
  bool motion_ring_matches(gc::api_e api, IUnknown *device, const motion_want_t &desc) {
    return g_motion_ring.api == api && g_motion_ring.device == device && g_motion_ring.desc == desc;
  }

  // A new ring's textures (no lock held: this allocates)
  bool motion_create_ring(gc::api_e api, IUnknown *device, const motion_want_t &desc, motion_ring_t &ring) {
    ring = motion_ring_t {};
    ring.api = api;
    ring.desc = desc;
    HRESULT hr = S_OK;
    if (api == gc::api_e::d3d12) {
      auto *device12 = static_cast<ID3D12Device *>(device);
      D3D12_HEAP_PROPERTIES heap {};
      heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      D3D12_RESOURCE_DESC rd {};
      rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      rd.Width = desc.width;
      rd.Height = desc.height;
      rd.DepthOrArraySize = 1;
      rd.MipLevels = 1;
      rd.Format = desc.format;
      rd.SampleDesc.Count = 1;
      rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
      for (int i = 0; i < kMotionRing && SUCCEEDED(hr); ++i) {
        hr = device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
                                               __uuidof(ID3D12Resource), reinterpret_cast<void **>(&ring.tex12[i]));
      }
    } else {
      auto *device11 = static_cast<ID3D11Device *>(device);
      D3D11_TEXTURE2D_DESC td {};
      td.Width = desc.width;
      td.Height = desc.height;
      td.MipLevels = 1;
      td.ArraySize = 1;
      td.Format = desc.format;
      td.SampleDesc.Count = 1;
      td.Usage = D3D11_USAGE_DEFAULT;
      td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      for (int i = 0; i < kMotionRing && SUCCEEDED(hr); ++i) {
        hr = device11->CreateTexture2D(&td, nullptr, &ring.tex11[i]);
      }
    }
    if (FAILED(hr)) {
      ring.release();
      log("DLSS motion vectors: the ring (%ux%u format %d) could not be created: 0x%08lx", desc.width, desc.height, desc.format, hr);
      return false;
    }
    device->AddRef();
    ring.device = device;
    return true;
  }

  // A ring on `device` for vectors of `desc` (made now if the current one is
  // of another device or shape); false if none can be had now
  bool motion_ensure_ring(gc::api_e api, IUnknown *device, const motion_want_t &desc) {
    {
      motion_lock_t lock;
      if (motion_ring_matches(api, device, desc)) {
        return true;
      }
    }
    motion_ring_t fresh;
    if (!motion_create_ring(api, device, desc, fresh)) {
      return false;
    }
    struct release_dead_t {
      ~release_dead_t() {
        motion_release_dead();
      }
    } release_dead;  // (after the lock below is let go)
    motion_lock_t lock;
    if (motion_ring_matches(api, device, desc)) {
      fresh.release();  // (another thread got there first)
      return true;
    }
    const bool room = motion_collect_old();
    if (g_motion_ring.serial) {
      if (!room) {
        fresh.release();  // (the replaced rings are all still in use)
        return false;
      }
      for (auto &r : g_motion_old) {
        if (!r.serial) {
          r = g_motion_ring;
          g_motion_ring = motion_ring_t {};
          break;
        }
      }
    }
    fresh.serial = ++g_motion_ring_serial;
    g_motion_ring = fresh;
    g_motion_consumed = g_motion_order;  // (no frame takes an older ring's vectors)
    log("DLSS motion vectors: copying %s vectors (%ux%u format %d)", api == gc::api_e::d3d12 ? "D3D12" : "D3D11", desc.width, desc.height, desc.format);
    return true;
  }

  // Motion lock held. The record and ring entry of a new evaluation on the
  // current ring (checked by the caller); null if no entry is free (the
  // evaluation goes without a copy)
  motion_eval_t *motion_new_eval(const motion_region_t &r, const void *list) {
    auto &ring = g_motion_ring;
    const bool d3d12 = ring.api == gc::api_e::d3d12;
    const auto id = g_motion_next_id;
    int entry = static_cast<int>(id % kMotionRing);
    ++g_motion_stats.evaluations;
    if (d3d12) {
      entry = -1;
      for (int i = 0; i < kMotionRing && entry < 0; ++i) {
        const int k = static_cast<int>((id + i) % kMotionRing);
        if (motion_entry_free(ring, k)) {
          entry = k;
        }
      }
      if (entry < 0) {
        ++g_motion_stats.no_entry;
        return nullptr;
      }
      auto &en = ring.entries[entry];
      safe_release(en.consumer);
      en = motion_entry_t {};
      en.writer = id;
    }
    // Records of the entry's previous contents describe it no more
    for (auto &old : g_motion_evals) {
      if (old.id && old.ring == ring.serial && old.entry == entry) {
        if (!old.submitted) {
          g_motion_unsubmitted.fetch_sub(1, std::memory_order_relaxed);
        }
        old = motion_eval_t {};
      }
    }
    ++g_motion_next_id;
    auto &e = g_motion_evals[id % kMotionEvals];
    if (e.id && !e.submitted) {
      motion_drop_unsubmitted(e, false);
    }
    e = motion_eval_t {};
    e.id = id;
    e.list = list;
    e.submitted = !d3d12;  // (D3D11: the immediate context orders it before the next Present)
    e.ring = ring.serial;
    e.entry = entry;
    e.width = r.width;
    e.height = r.height;
    e.out_width = r.out_width;
    e.out_height = r.out_height;
    e.scale_x = r.scale_x;
    e.scale_y = r.scale_y;
    if (d3d12) {
      g_motion_unsubmitted.fetch_add(1, std::memory_order_relaxed);
    } else {
      e.order = ++g_motion_order;
    }
    return &e;
  }

  void log_first_eval(const char *api, const motion_region_t &r, UINT mv_w, UINT mv_h, DXGI_FORMAT format) {
    static std::atomic<bool> logged {false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
      log("DLSS evaluation seen (%s): vectors %ux%u format %d, region %u,%u %ux%u, output %ux%u, scale %g,%g", api, mv_w, mv_h, format,
          r.x, r.y, r.width, r.height, r.out_width, r.out_height, r.scale_x, r.scale_y);
    }
  }

  void motion_after_eval12(ID3D12GraphicsCommandList *list, const void *handle, const void *params) {
    if (!g_real_execute_command_lists || !g_real_list_reset) {
      return;  // (submissions and resets cannot be followed)
    }
    const auto type = list->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) {
      return;
    }
    auto *mv = static_cast<ID3D12Resource *>(ngx_resource(params, kNgxGetD3d12, "MotionVectors", "#\x27"));
    auto *out = static_cast<ID3D12Resource *>(ngx_resource(params, kNgxGetD3d12, "Output", "#\x22"));
    if (!mv || !out) {
      return;
    }
    const auto mvd = mv->GetDesc();
    const auto od = out->GetDesc();
    const DXGI_FORMAT format = motion_format(mvd.Format);
    if (mvd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || mvd.SampleDesc.Count != 1 || format == DXGI_FORMAT_UNKNOWN ||
        mvd.Width > 16384 || mvd.Height > 16384) {
      return;
    }
    if (mvd.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) {
      log_once(g_logged_simultaneous, "the game's vector texture is simultaneous-access (its state is unknown to us); not used");
      return;
    }
    motion_params_t p;
    read_motion_params(params, kNgxGetD3d12, p);
    ID3D12Device *device = nullptr;
    if (FAILED(list->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&device)))) {
      return;
    }
    struct release_t {
      ID3D12Device *&d;

      ~release_t() {
        safe_release(d);
      }
    } release {device};

    motion_region_t r;
    {
      motion_lock_t lock;
      if (!motion_region(handle, p, static_cast<UINT>(mvd.Width), mvd.Height, static_cast<UINT>(od.Width), od.Height, r)) {
        return;
      }
    }
    log_first_eval("D3D12", r, static_cast<UINT>(mvd.Width), mvd.Height, mvd.Format);
    const motion_want_t desc {format, static_cast<UINT>(mvd.Width), mvd.Height};
    if (!motion_ensure_ring(gc::api_e::d3d12, device, desc)) {
      return;
    }
    motion_lock_t lock;
    if (!motion_ring_matches(gc::api_e::d3d12, device, desc)) {
      return;  // (replaced meanwhile)
    }
    const motion_eval_t *e = motion_new_eval(r, list);
    if (!e) {
      return;
    }
    auto *ring = g_motion_ring.tex12[e->entry];

    // The vectors are in NON_PIXEL_SHADER_RESOURCE at the evaluation (DLSS
    // requires it); the ring entry rests in COPY_SOURCE. Only subresource 0
    // (the one DLSS reads) changes state.
    D3D12_RESOURCE_BARRIER before[2] = {
      transition(ring, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
      transition(mv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE),
    };
    D3D12_RESOURCE_BARRIER after[2] = {
      transition(ring, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
      transition(mv, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
    };
    before[1].Transition.Subresource = after[1].Transition.Subresource = 0;
    D3D12_TEXTURE_COPY_LOCATION dst {};
    dst.pResource = ring;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION src {};
    src.pResource = mv;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    const D3D12_BOX box {r.x, r.y, 0, r.x + r.width, r.y + r.height, 1};
    list->ResourceBarrier(2, before);
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    list->ResourceBarrier(2, after);
  }

  void motion_after_eval11(ID3D11DeviceContext *context, const void *handle, const void *params) {
    if (context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
      return;  // (a deferred context's work runs whenever the game executes it)
    }
    auto *mv_resource = static_cast<ID3D11Resource *>(ngx_resource(params, kNgxGetD3d11, "MotionVectors", "#\x27"));
    auto *out_resource = static_cast<ID3D11Resource *>(ngx_resource(params, kNgxGetD3d11, "Output", "#\x22"));
    if (!mv_resource || !out_resource) {
      return;
    }
    ID3D11Texture2D *mv = nullptr, *out = nullptr;
    ID3D11Device *device = nullptr;
    struct release_t {
      ID3D11Texture2D *&a, *&b;
      ID3D11Device *&d;

      ~release_t() {
        safe_release(a);
        safe_release(b);
        safe_release(d);
      }
    } release {mv, out, device};
    context->GetDevice(&device);
    if (!device || (device->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED)) {
      log_once(g_logged_single_threaded, "the game's D3D11 device is single-threaded; not used");
      return;
    }
    if (FAILED(mv_resource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&mv))) ||
        FAILED(out_resource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&out)))) {
      return;
    }
    D3D11_TEXTURE2D_DESC mvd {}, od {};
    mv->GetDesc(&mvd);
    out->GetDesc(&od);
    const DXGI_FORMAT format = motion_format(mvd.Format);
    if (mvd.SampleDesc.Count != 1 || format == DXGI_FORMAT_UNKNOWN || mvd.Width > 16384 || mvd.Height > 16384) {
      return;
    }
    motion_params_t p;
    read_motion_params(params, kNgxGetD3d11, p);

    motion_region_t r;
    {
      motion_lock_t lock;
      if (!motion_region(handle, p, mvd.Width, mvd.Height, od.Width, od.Height, r)) {
        return;
      }
    }
    log_first_eval("D3D11", r, mvd.Width, mvd.Height, mvd.Format);
    const motion_want_t desc {format, mvd.Width, mvd.Height};
    if (!motion_ensure_ring(gc::api_e::d3d11, device, desc)) {
      return;
    }
    motion_lock_t lock;
    if (!motion_ring_matches(gc::api_e::d3d11, device, desc)) {
      return;
    }
    const motion_eval_t *e = motion_new_eval(r, nullptr);
    if (!e) {
      return;
    }
    const D3D11_BOX box {r.x, r.y, 0, r.x + r.width, r.y + r.height, 1};
    context->CopySubresourceRegion(g_motion_ring.tex11[e->entry], 0, 0, 0, 0, mv, 0, &box);
  }

  bool motion_submissions_pending() {
    return g_motion_unsubmitted.load(std::memory_order_relaxed) > 0;
  }

  // Motion lock held. The queue's slot in g_motion_queues, or -1
  int motion_queue_slot(ID3D12CommandQueue *queue) {
    for (int i = 0; i < kMotionQueues; ++i) {
      if (g_motion_queues[i].queue == queue) {
        return i;
      }
    }
    return -1;
  }

  // Motion lock held. Whether a ring entry refers to queue slot `i`
  bool motion_queue_referenced(int i) {
    auto refers = [i](const motion_ring_t &r) {
      for (const auto &e : r.entries) {
        if (e.queue == i) {
          return true;
        }
      }
      return false;
    };
    if (refers(g_motion_ring)) {
      return true;
    }
    for (const auto &r : g_motion_old) {
      if (r.serial && refers(r)) {
        return true;
      }
    }
    return false;
  }

  // Motion lock held. Puts `fence` (made outside the lock) in a slot for
  // `queue`: a free one, or one whose fence completed and that no entry
  // refers to (its queue and fence come back in `old_*` for release outside
  // the lock). -1 if none: the caller keeps `fence`.
  int motion_queue_install(ID3D12CommandQueue *queue, ID3D12Fence *fence, ID3D12CommandQueue *&old_queue, ID3D12Fence *&old_fence) {
    int slot = -1;
    for (int i = 0; i < kMotionQueues && slot < 0; ++i) {
      if (!g_motion_queues[i].queue) {
        slot = i;
      }
    }
    for (int i = 0; i < kMotionQueues && slot < 0; ++i) {
      const auto &q = g_motion_queues[i];
      if (q.fence->GetCompletedValue() >= q.value && !motion_queue_referenced(i)) {
        slot = i;
        old_queue = q.queue;
        old_fence = q.fence;
      }
    }
    if (slot < 0) {
      return -1;
    }
    queue->AddRef();
    g_motion_queues[slot] = {queue, fence, 0};
    return slot;
  }

  // Motion lock held. Whether these lists hold an evaluation's copy not yet
  // seen submitted
  bool motion_lists_hold_eval(UINT count, ID3D12CommandList *const *lists) {
    for (const auto &e : g_motion_evals) {
      if (!e.id || e.submitted) {
        continue;
      }
      for (UINT i = 0; i < count; ++i) {
        if (static_cast<const void *>(lists[i]) == e.list) {  // (single inheritance: the list's own pointer)
          return true;
        }
      }
    }
    return false;
  }

  // D3D12, from the ExecuteCommandLists detour after the real call:
  // evaluations recorded into these lists were executed on `queue`; the
  // queue's fence is signalled there once (the producer value) and each gets
  // its execution number. The fence is made outside the lock the first time
  // a queue shows up; the Signal is made under it, so each queue's values
  // follow the order the signals were queued in.
  void motion_note_submission(ID3D12CommandQueue *queue, UINT count, ID3D12CommandList *const *lists) {
    bool need_fence = false;
    {
      motion_lock_t lock;
      if (!motion_lists_hold_eval(count, lists)) {
        return;
      }
      need_fence = motion_queue_slot(queue) < 0;
    }
    ID3D12Fence *fresh = nullptr;
    if (need_fence) {
      ID3D12Device *device = nullptr;
      if (SUCCEEDED(queue->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&device)))) {
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void **>(&fresh));
        device->Release();
      }
    }
    ID3D12CommandQueue *old_queue = nullptr;
    ID3D12Fence *old_fence = nullptr;
    {
      motion_lock_t lock;
      int slot = motion_queue_slot(queue);
      if (slot < 0 && fresh) {
        slot = motion_queue_install(queue, fresh, old_queue, old_fence);
        if (slot >= 0) {
          fresh = nullptr;
        }
      }
      std::uint64_t value = 0;
      for (auto &e : g_motion_evals) {
        if (!e.id || e.submitted) {
          continue;
        }
        bool here = false;
        for (UINT i = 0; i < count && !here; ++i) {
          here = static_cast<const void *>(lists[i]) == e.list;
        }
        if (!here) {
          continue;
        }
        auto *ring = motion_ring(e.ring);
        if (!ring || slot < 0) {
          motion_drop_unsubmitted(e, false);  // (nothing will say when it ran: its ring stays)
          continue;
        }
        if (!value) {
          value = g_motion_queues[slot].value + 1;
          if (FAILED(queue->Signal(g_motion_queues[slot].fence, value))) {
            value = 0;
            motion_drop_unsubmitted(e, false);
            continue;
          }
          g_motion_queues[slot].value = value;
        }
        auto &en = ring->entries[e.entry];
        en.queue = slot;
        en.produced = value;
        e.submitted = true;
        e.order = ++g_motion_order;
        g_motion_unsubmitted.fetch_sub(1, std::memory_order_relaxed);
      }
    }
    safe_release(fresh);
    safe_release(old_fence);
    safe_release(old_queue);
  }

  // D3D12, from the command-list Reset detour after a successful Reset:
  // recordings into this list that were never submitted are gone
  void motion_note_reset(const void *list) {
    motion_lock_t lock;
    for (auto &e : g_motion_evals) {
      if (e.id && !e.submitted && e.list == list) {
        motion_drop_unsubmitted(e, true);
      }
    }
  }

  // Motion lock held. The periodic line, when due (written by the caller
  // after unlocking); resets the counts
  bool motion_stats_line(char (&line)[512]) {
    auto &s = g_motion_stats;
    const auto now = qpc_now();
    if (!s.logged_qpc) {
      s.logged_qpc = now;
      return false;
    }
    if (now - s.logged_qpc < 10 * qpc_frequency()) {
      return false;
    }
    std::snprintf(line, sizeof(line),
                  "DLSS motion vectors, last 10 s: %llu evaluations (%llu without a free entry); Presents found 0/1/2/3/4+ pending %llu/%llu/%llu/%llu/%llu; "
                  "frames given vectors %llu, %llu evaluations dropped (backlog), %llu unfinished on another queue",
                  static_cast<unsigned long long>(s.evaluations), static_cast<unsigned long long>(s.no_entry), static_cast<unsigned long long>(s.pending[0]),
                  static_cast<unsigned long long>(s.pending[1]), static_cast<unsigned long long>(s.pending[2]), static_cast<unsigned long long>(s.pending[3]),
                  static_cast<unsigned long long>(s.pending[4]), static_cast<unsigned long long>(s.picked), static_cast<unsigned long long>(s.dropped),
                  static_cast<unsigned long long>(s.unfinished));
    s = motion_stats_t {};
    s.logged_qpc = now;
    return true;
  }

  // Every Present of the captured swapchain (captured or not): consumes this
  // frame's evaluation (see the pairing rules above) and advances the
  // motion sequence
  void motion_on_present(IDXGISwapChain *swapchain) {
    if (!motion_wanted()) {
      return;
    }
    if (auto *captured = g_captured_swapchain.load(std::memory_order_acquire); captured && captured != swapchain) {
      return;  // (another swapchain's Present)
    }
    char line[512];
    bool log_now = false;
    {
      motion_lock_t lock;
      ++g_motion_seq;
      g_motion_frame_eval = 0;
      motion_collect_old();
      const auto &ring = g_motion_ring;
      auto pending_eval = [&](const motion_eval_t &e) {
        return e.id && e.submitted && e.ring == ring.serial && e.order > g_motion_consumed;
      };
      std::uint64_t main_area = 0;
      for (const auto &e : g_motion_evals) {
        if (pending_eval(e)) {
          main_area = std::max(main_area, static_cast<std::uint64_t>(e.out_width) * e.out_height);
        }
      }
      const motion_eval_t *pending[kMotionEvals];
      int n = 0;
      std::uint64_t newest = 0;
      for (const auto &e : g_motion_evals) {
        if (pending_eval(e)) {
          newest = std::max(newest, e.order);
          if (static_cast<std::uint64_t>(e.out_width) * e.out_height == main_area) {
            pending[n++] = &e;
          }
        }
      }
      std::sort(pending, pending + n, [](const motion_eval_t *a, const motion_eval_t *b) {
        return a->order < b->order;
      });
      ++g_motion_stats.pending[std::min(n, 4)];
      if (n > 3) {
        g_motion_consumed = newest;  // (a backlog no frame can be matched against: dropped)
        g_motion_stats.dropped += n;
      } else if (n > 0) {
        g_motion_consumed = pending[0]->order;  // (older secondary views go with it)
        g_motion_frame_eval = pending[0]->id;
        g_motion_frame_seq = g_motion_seq;
      }
      log_now = motion_stats_line(line);
    }
    motion_release_dead();
    if (log_now) {
      log("%s", line);
    }
  }

  // Capture lock held, before the capture copy is recorded: the vectors the
  // current Present consumed, if made on `device` (D3D12: read on
  // `present_queue`, without waiting for another queue; the entry is
  // reserved for the copy until motion_consumed or motion_unread). False if
  // there are none; `want` then still names the ring's shape, so the
  // generation keeps (or gets) motion textures.
  bool motion_pick(gc::api_e api, IUnknown *device, ID3D12CommandQueue *present_queue, motion_pick_t &pick, motion_want_t &want) {
    pick = motion_pick_t {};
    want = motion_want_t {};
    if (!motion_wanted()) {
      return false;
    }
    motion_lock_t lock;
    auto &ring = g_motion_ring;
    if (ring.api != api || ring.device != device) {
      return false;
    }
    want = ring.desc;
    const auto id = g_motion_frame_eval;
    g_motion_frame_eval = 0;
    if (!id || g_motion_frame_seq != g_motion_seq) {
      return false;
    }
    const auto &e = g_motion_evals[id % kMotionEvals];
    if (e.id != id || e.ring != ring.serial) {
      return false;
    }
    // The entry must still hold this evaluation, and a D3D12 copy must be
    // readable on the presenting queue without waiting for another queue
    if (api == gc::api_e::d3d12) {
      const auto &en = ring.entries[e.entry];
      if (en.writer != e.id || en.queue < 0) {
        return false;
      }
      if (g_motion_queues[en.queue].queue != present_queue && !motion_produced(ring, en)) {
        ++g_motion_stats.unfinished;
        return false;
      }
    }
    ++g_motion_stats.picked;
    pick.id = g_motion_seq;
    pick.width = e.width;
    pick.height = e.height;
    pick.out_width = e.out_width;
    pick.out_height = e.out_height;
    pick.scale_x = e.scale_x;
    pick.scale_y = e.scale_y;
    pick.textures = ring.desc;
    pick.ring = ring.serial;
    pick.entry = e.entry;
    if (api == gc::api_e::d3d12) {
      pick.src12 = ring.tex12[e.entry];
      pick.src12->AddRef();
      ++ring.entries[e.entry].readers;
      pick.reading = true;
    } else {
      pick.src11 = ring.tex11[e.entry];
      pick.src11->AddRef();
    }
    return true;
  }

  // D3D12: a capture copy prepared from an entry was submitted, its reads
  // done when `fence` reaches `value`
  void motion_consumed(std::uint32_t ring_serial, int entry, ID3D12Fence *fence, std::uint64_t value) {
    motion_lock_t lock;
    if (auto *ring = motion_ring(ring_serial); ring && entry >= 0) {
      auto &en = ring->entries[entry];
      if (en.consumer != fence) {
        safe_release(en.consumer);
        fence->AddRef();
        en.consumer = fence;
      }
      en.consumed = std::max(en.consumed, value);
      en.readers = std::max(0, en.readers - 1);
    }
  }

  // D3D12: ... or given back unsubmitted
  void motion_unread(std::uint32_t ring_serial, int entry) {
    motion_lock_t lock;
    if (auto *ring = motion_ring(ring_serial); ring && entry >= 0) {
      ring->entries[entry].readers = std::max(0, ring->entries[entry].readers - 1);
    }
  }

  // Whether the current generation's motion textures can take `pick`
  bool motion_fits(const motion_pick_t &pick) {
    return pick.id && g_cap.motion_format == pick.textures.format && g_cap.motion_width == pick.textures.width && g_cap.motion_height == pick.textures.height;
  }

  // ---- the NGX detours

  using ngx_eval12_fn = std::uint32_t (*)(ID3D12GraphicsCommandList *, const void *, const void *, void *);
  using ngx_eval11_fn = std::uint32_t (*)(ID3D11DeviceContext *, const void *, const void *, void *);
  using ngx_create_fn = std::uint32_t (*)(void *, std::uint32_t, void *, void **);
  using ngx_release_fn = std::uint32_t (*)(void *);

  ngx_eval12_fn g_real_ngx_eval12 = nullptr;
  ngx_eval12_fn g_real_ngx_eval12_c = nullptr;
  ngx_eval11_fn g_real_ngx_eval11 = nullptr;
  ngx_eval11_fn g_real_ngx_eval11_c = nullptr;
  ngx_create_fn g_real_ngx_create12 = nullptr;
  ngx_create_fn g_real_ngx_create11 = nullptr;
  ngx_release_fn g_real_ngx_release12 = nullptr;
  ngx_release_fn g_real_ngx_release11 = nullptr;

  template<ngx_eval12_fn *Real>
  std::uint32_t hook_ngx_eval12(ID3D12GraphicsCommandList *list, const void *handle, const void *params, void *callback) {
    const auto result = (*Real)(list, handle, params, callback);
    if (ngx_ok(result) && list && params && motion_wanted()) {
      const DWORD error = GetLastError();
      motion_after_eval12(list, handle, params);
      SetLastError(error);
    }
    return result;
  }

  template<ngx_eval11_fn *Real>
  std::uint32_t hook_ngx_eval11(ID3D11DeviceContext *context, const void *handle, const void *params, void *callback) {
    const auto result = (*Real)(context, handle, params, callback);
    if (ngx_ok(result) && context && params && motion_wanted()) {
      const DWORD error = GetLastError();
      motion_after_eval11(context, handle, params);
      SetLastError(error);
    }
    return result;
  }

  template<ngx_create_fn *Real>
  std::uint32_t hook_ngx_create(void *list_or_context, std::uint32_t feature, void *params, void **out_handle) {
    const auto result = (*Real)(list_or_context, feature, params, out_handle);
    if (ngx_ok(result) && out_handle && *out_handle) {
      const DWORD error = GetLastError();
      motion_note_feature(*out_handle, feature, params);
      SetLastError(error);
    }
    return result;
  }

  template<ngx_release_fn *Real>
  std::uint32_t hook_ngx_release(void *handle) {
    motion_forget_feature(handle);
    return (*Real)(handle);
  }

  // Detours whichever of the entry points _nvngx.dll exports; true once any
  // evaluation entry is detoured
  bool install_ngx_hooks(HMODULE ngx) {
    struct entry_t {
      const char *name;
      void *detour;
      void **real;
    };
    const entry_t entries[] = {
      {"NVSDK_NGX_D3D12_EvaluateFeature", reinterpret_cast<void *>(&hook_ngx_eval12<&g_real_ngx_eval12>), reinterpret_cast<void **>(&g_real_ngx_eval12)},
      {"NVSDK_NGX_D3D12_EvaluateFeature_C", reinterpret_cast<void *>(&hook_ngx_eval12<&g_real_ngx_eval12_c>), reinterpret_cast<void **>(&g_real_ngx_eval12_c)},
      {"NVSDK_NGX_D3D11_EvaluateFeature", reinterpret_cast<void *>(&hook_ngx_eval11<&g_real_ngx_eval11>), reinterpret_cast<void **>(&g_real_ngx_eval11)},
      {"NVSDK_NGX_D3D11_EvaluateFeature_C", reinterpret_cast<void *>(&hook_ngx_eval11<&g_real_ngx_eval11_c>), reinterpret_cast<void **>(&g_real_ngx_eval11_c)},
      {"NVSDK_NGX_D3D12_CreateFeature", reinterpret_cast<void *>(&hook_ngx_create<&g_real_ngx_create12>), reinterpret_cast<void **>(&g_real_ngx_create12)},
      {"NVSDK_NGX_D3D11_CreateFeature", reinterpret_cast<void *>(&hook_ngx_create<&g_real_ngx_create11>), reinterpret_cast<void **>(&g_real_ngx_create11)},
      {"NVSDK_NGX_D3D12_ReleaseFeature", reinterpret_cast<void *>(&hook_ngx_release<&g_real_ngx_release12>), reinterpret_cast<void **>(&g_real_ngx_release12)},
      {"NVSDK_NGX_D3D11_ReleaseFeature", reinterpret_cast<void *>(&hook_ngx_release<&g_real_ngx_release11>), reinterpret_cast<void **>(&g_real_ngx_release11)},
    };
    const auto init = MH_Initialize();
    if (init != MH_OK && init != MH_ERROR_ALREADY_INITIALIZED) {
      log("DLSS motion vectors: MinHook unavailable (%d)", static_cast<int>(init));
      return false;
    }
    int evaluations = 0;
    for (const auto &e : entries) {
      void *target = reinterpret_cast<void *>(GetProcAddress(ngx, e.name));
      if (!target) {
        continue;
      }
      if (MH_CreateHook(target, e.detour, e.real) != MH_OK || MH_EnableHook(target) != MH_OK) {
        *e.real = nullptr;
        log("DLSS motion vectors: %s could not be detoured", e.name);
        continue;
      }
      if (std::strstr(e.name, "Evaluate")) {
        ++evaluations;
      }
    }
    log("DLSS motion vectors: %d NGX evaluation entry points detoured", evaluations);
    return evaluations > 0;
  }

  // Waits for the driver's NGX to be loaded (games load it when they set up
  // DLSS, maybe long after we came) while motion vectors are wanted, then
  // detours it and exits. The module is pinned: our detours live in it.
  DWORD WINAPI ngx_watch_main(void *) {
    for (;;) {
      if (motion_wanted()) {
        HMODULE ngx = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, L"_nvngx.dll", &ngx) && ngx) {
          install_ngx_hooks(ngx);
          return 0;
        }
      }
      Sleep(500);
    }
  }

  // Capture lock held. `device` is the swapchain's (borrowed).
  void submit12(const prepared12_t &p, IDXGISwapChain *swapchain, HWND hwnd, bool inside_dxgi = false);

  // Records the copy of the current back buffer and, without `prepare`,
  // submits it at once; with it, leaves the recorded copy there (valid on
  // success) to be submitted later by submit12 or given back
  void capture_frame12(IDXGISwapChain *swapchain, ID3D12Device *device, HWND hwnd, std::uint64_t present_qpc, std::uint64_t release_qpc, prepared12_t *prepare = nullptr) {
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
    if (queue) {
      DXGI_SWAP_CHAIN_DESC desc {};
      // Only a back buffer DXGI was seen presenting from this queue is copied
      // on it (the others may belong to a queue we cannot see)
      IDXGISwapChain3 *swapchain3 = nullptr;
      UINT current = UINT_MAX;
      if (SUCCEEDED(swapchain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&swapchain3)))) {
        current = swapchain3->GetCurrentBackBufferIndex();
        swapchain3->Release();
      }
      const auto mask = queue_seen(swapchain);
      if (FAILED(swapchain->GetDesc(&desc)) || current >= 32 || !(mask & (1u << current))) {
        safe_release(queue);  // not yet confirmed for this buffer
      }
    }
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
    const UINT index = swapchain3->GetCurrentBackBufferIndex();  // (before the real Present: the buffer being presented)
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
    // The DLSS vectors this frame carries: the newest evaluation the game
    // submitted before this Present (see the motion section)
    motion_pick_t pick;
    motion_want_t want;
    motion_pick(gc::api_e::d3d12, device, g_cap.queue, pick, want);
    struct release_pick_t {
      motion_pick_t &p;

      ~release_pick_t() {
        p.release();
      }
    } release_pick {pick};
    if (!ensure_textures12(back_desc, hwnd, want)) {
      return;
    }
    const std::uint32_t color_space = swapchain_color_space(swapchain);

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
    const bool motion = motion_fits(pick) && g_cap.motion12[slot];
    if (motion) {
      // The ring entry rests in COPY_SOURCE; the motion texture is promoted
      // and decays like the frame texture
      D3D12_TEXTURE_COPY_LOCATION dst {};
      dst.pResource = g_cap.motion12[slot];
      dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      D3D12_TEXTURE_COPY_LOCATION src {};
      src.pResource = pick.src12;
      src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      const D3D12_BOX box {0, 0, 0, pick.width, pick.height, 1};
      list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    }
    if (const HRESULT closed = list->Close(); FAILED(closed)) {
      log("Closing the command list of slot %d failed: 0x%08lx", slot, closed);
      recreate_list(slot);
      give_back();
      return;
    }

    prepared12_t p;
    p.valid = true;
    p.slot = slot;
    p.list = list;
    p.queue = g_cap.queue;
    p.fence = g_cap.fence12;
    p.generation = g_current_generation.load(std::memory_order_acquire);
    p.frame_id = ++g_cap.next_frame_id;
    p.version = write_slot_record(slot, p.generation, color_space, p.frame_id, present_qpc, release_qpc, motion ? &pick : nullptr);
    if (motion) {
      // The entry's reservation goes with the recorded copy
      p.motion_ring = pick.ring;
      p.motion_entry = pick.entry;
      pick.reading = false;
    }
    if (prepare) {
      *prepare = p;  // submitted from inside the real Present (run_armed_capture)
      return;
    }
    submit12(p, swapchain, hwnd);
  }

  // Executes a recorded copy on its queue and signals our fence; the slot
  // publishes on completion. Capture lock held; `p` recorded against the
  // current g_cap (checked by the caller).
  // (inside_dxgi: called from within DXGI's own submission, where nothing may
  // be created or released; cleanup then waits for the next capture pass)
  void submit12(const prepared12_t &p, IDXGISwapChain *swapchain, HWND hwnd, bool inside_dxgi) {
    const int slot = p.slot;
    auto *list = p.list;
    const auto generation = p.generation;
    const auto frame_id = p.frame_id;
    const auto version = p.version;

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
      if (p.motion_entry >= 0) {
        motion_consumed(p.motion_ring, p.motion_entry, g_cap.fence12, fence_value);  // (never completes: the entry stays taken)
      }
      g_d3d12_dead.store(true, std::memory_order_release);
      if (!inside_dxgi) {
        release_capture(hwnd);  // (otherwise the next capture pass does, on seeing g_d3d12_dead)
      }
      char msg[80];
      std::snprintf(msg, sizeof(msg), "D3D12 Signal failed: 0x%08lx; capture stopped", signalled);
      set_state(gc::hook_state_e::failed, msg);
      return;
    }
    g_cap.list_fence[slot] = fence_value;
    g_cap.highest_fence_value_used = fence_value;
    if (p.motion_entry >= 0) {
      motion_consumed(p.motion_ring, p.motion_entry, g_cap.fence12, fence_value);
    }
    const auto ticket = mark_pending(slot, version, generation, frame_id, fence_value);
    if (FAILED(g_cap.fence12->SetEventOnCompletion(fence_value, g_slots[slot].done_event))) {
      drop_pending(slot, ticket);  // the copy still completes; only the publication is lost
      return;
    }
    note_capturing(swapchain, "D3D12");
  }

  // A recorded copy that will not be submitted: its slot goes back
  void give_back_prepared(prepared12_t &p) {
    if (p.valid && p.motion_entry >= 0) {
      motion_unread(p.motion_ring, p.motion_entry);
      p.motion_entry = -1;
    }
    if (p.valid) {
      g_block->owner[p.slot].store(gc::kOwnerNone, std::memory_order_release);
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      p.valid = false;
    }
  }

  void disarm_capture() {
    give_back_prepared(t_armed.prepared);
    t_armed = armed_capture_t {};
  }

  void cancel_prepared_for(IDXGISwapChain *swapchain) {
    if (t_presenting) {
      t_seen_contaminated = true;  // a Present that resizes teaches nothing about its (old) buffers
    }
    cancel_prepared11_for(swapchain);
    IUnknown *a = nullptr, *b = nullptr;
    if (!t_armed.prepared.valid || !t_armed.swapchain) {
      return;
    }
    t_armed.swapchain->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void **>(&a));
    swapchain->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void **>(&b));
    if (a && a == b) {
      give_back_prepared(t_armed.prepared);
      log("Swapchain %p resized inside its Present: the frame's copy was dropped", static_cast<void *>(swapchain));
    }
    safe_release(a);
    safe_release(b);
  }

  // From the ExecuteCommandLists detour: dxgi.dll submits on the armed copy's
  // queue inside the real Present. The copy goes ahead of it, unless another
  // swapchain's Present intervened or the capture changed since recording.
  void run_armed_capture(ID3D12CommandQueue *queue) {
    auto &p = t_armed.prepared;
    if (t_seen_contaminated) {
      give_back_prepared(p);  // (another swapchain's submission: no boundary of ours)
      return;
    }
    t_armed.boundary_seen = true;
    if (!p.valid) {
      return;
    }
    // Never wait inside DXGI: a capture elsewhere holding the lock costs
    // this frame, not a stall
    struct lock_t {
      bool held = TryAcquireSRWLockExclusive(&g_capture_lock);

      ~lock_t() {
        if (held) {
          ReleaseSRWLockExclusive(&g_capture_lock);
        }
      }
    } lock;
    if (!lock.held) {
      give_back_prepared(p);
      return;
    }
    const bool current = g_cap.api == gc::api_e::d3d12 && g_cap.swapchain == t_armed.swapchain && g_cap.queue == queue &&
                         g_cap.fence12 == p.fence && g_cap.lists[p.slot] == p.list &&
                         g_current_generation.load(std::memory_order_acquire) == p.generation && reserve_retirement();
    if (!current) {
      give_back_prepared(p);
      return;
    }
    const prepared12_t submitting = p;
    p.valid = false;  // (consumed, whatever submit12 does)
    submit12(submitting, t_armed.swapchain, t_armed.hwnd, true);
  }

  // After the real Present. A Present that showed DXGI's submission on the
  // copy's queue resets the miss count; an armed one that did not is a miss,
  // and enough in a row send the swapchain back to copying before Present.
  // Presents where nothing was armed (not captured, no free slot, ...) are
  // no evidence either way.
  // Per swapchain, after a real Present that armed a late copy: reaching the
  // late copy point resets its miss count; not reaching it is a miss, and
  // kSubmitMisses in a row send the swapchain back to copying before Present
  void count_boundary(IDXGISwapChain *swapchain, submit_mode_e late_mode, bool seen, const char *what) {
    const auto value = swapchain_count(swapchain, kSubmitModeKey);
    if ((value & 0xff) != static_cast<std::uint32_t>(late_mode)) {
      return;
    }
    if (seen) {
      if ((value >> 8) != 0) {
        set_swapchain_count(swapchain, kSubmitModeKey, static_cast<std::uint32_t>(late_mode));
      }
      return;
    }
    const auto misses = (value >> 8) + 1;
    if (misses >= kSubmitMisses) {
      set_swapchain_count(swapchain, kSubmitModeKey, static_cast<std::uint32_t>(submit_mode_e::abandoned));
      log("Swapchain %p: %s in %u Presents in a row; copying before Present from now on (overlays drawn after our hook are missed)",
          static_cast<void *>(swapchain), what, misses);
    } else {
      set_swapchain_count(swapchain, kSubmitModeKey, (misses << 8) | static_cast<std::uint32_t>(late_mode));
    }
  }

  // After the real Present (D3D12). Presents where nothing was armed (not
  // captured, no free slot, ...) are no evidence either way.
  void settle_armed_capture(IDXGISwapChain *swapchain, bool presented) {
    if (!t_armed.armed) {
      return;
    }
    const bool seen = t_armed.boundary_seen;
    disarm_capture();  // (gives back a copy that was never submitted)
    if (presented) {
      count_boundary(swapchain, submit_mode_e::at_dxgi_submit, seen, "DXGI's submission was missing");
    }
  }

  HWND setup_hwnd() {
    return reinterpret_cast<HWND>(g_block->setup.hwnd.load(std::memory_order_relaxed));
  }

  struct capture_lock_t {
    bool held;

    explicit capture_lock_t(bool block) {
      if (block) {
        AcquireSRWLockExclusive(&g_capture_lock);
        held = true;
      } else {
        held = TryAcquireSRWLockExclusive(&g_capture_lock);
      }
    }

    ~capture_lock_t() {
      if (held) {
        ReleaseSRWLockExclusive(&g_capture_lock);
      }
    }
  };

  // Whether this thread may touch the captured device (a single-threaded
  // D3D11 device only from the thread that first presented it)
  bool on_capture_thread() {
    return !(g_cap.device && g_cap.single_threaded && GetCurrentThreadId() != g_cap.owner_thread);
  }

  // The captured swapchain's device was lost (its Present said so): let go
  // of it now, since no further capture may come to notice
  void drop_lost_capture(IDXGISwapChain *swapchain) {
    if (g_captured_swapchain.load(std::memory_order_acquire) != swapchain) {
      return;
    }
    capture_lock_t lock(true);
    if (g_cap.swapchain == swapchain && on_capture_thread() && reserve_retirement()) {
      log("The captured swapchain's device was lost: capture released");
      release_capture(setup_hwnd());
    }
  }

  // While not capturing: retired entries are still collected, and a D3D12
  // capture that must end (lost device, vetoed swapchain, D3D12 stopped)
  // still ends
  void housekeeping(IDXGISwapChain *swapchain) {
    capture_lock_t lock(false);
    if (!lock.held) {
      return;
    }
    if (g_cap.device12 && reserve_retirement() &&
        (FAILED(g_cap.device12->GetDeviceRemovedReason()) || g_d3d12_dead.load(std::memory_order_acquire) || (g_cap.swapchain == swapchain && vetoed(swapchain)))) {
      release_capture(setup_hwnd());
    }
    collect_retired(false);
  }

  // ---- The PresentImpl boundary (D3D11) ------------------------------------
  //
  // An overlay injected before us (Steam's) hooks the public Present entry
  // points and draws into the back buffer from its hook, which sits behind
  // ours: a copy taken before calling the real Present misses it, and a copy
  // after Present reads the wrong buffer on flip-model swapchains (two frames
  // behind, measured). D3D11 has no submission of DXGI's own to slip ahead
  // of, unlike D3D12. What it has: both public entries, Present and Present1,
  // call one internal function, CDXGISwapChain::PresentImpl, once their hook
  // chains have run and before DXGI's own presentation work begins. A detour
  // at its entry runs on the presenting thread with the frame final; the
  // copy is enqueued on the immediate context there, so it precedes the
  // presentation the same way a copy before an unhooked Present would.
  //
  // PresentImpl is not exported and moves with every dxgi.dll build, so the
  // host resolves it from Microsoft's public symbols for the system dxgi.dll
  // (dxgi_symbols.cpp; the symbol's decorated name fixes the argument list
  // this detour forwards) and hands the RVA over in the shared block. The
  // hook takes nothing on trust: the loaded dxgi.dll must be the build the
  // host resolved; Present, Present1 and PresentImpl must each be a function
  // of the exception directory starting at their addresses; Present and
  // Present1, read from dxgi.dll on disk (their entries in memory are the
  // overlays' patches), must each contain exactly one direct call to
  // PresentImpl, whose return addresses the detour then requires; and
  // PresentImpl's entry must be unpatched. Anything else: frames are copied
  // before Present as before, and the log says why.
  //
  // Per frame: the copy is prepared before the real Present (device and
  // textures, a slot with its record written, a reference to the back
  // buffer) and enqueued at the boundary, if it is reached for this
  // swapchain from one of those two call sites during this Present and the
  // capture is unchanged; otherwise the slot goes back and, after
  // kSubmitMisses such Presents in a row, the swapchain copies before
  // Present again. Present1 with dirty or scroll rectangles is not captured
  // at all: its back buffer holds only what changed.

  using present_impl_fn = HRESULT(WINAPI *)(void *self, const void *args, UINT dirty_count, const RECT *dirty_rects, UINT scroll_count, const void *scroll_rects, IUnknown *resource);
  present_impl_fn g_real_present_impl = nullptr;
  void *g_present_impl_return[2] = {};  // the return addresses of Present's and Present1's calls to it
  std::atomic<bool> g_present_impl_ready {false};
  bool g_present_impl_attempted = false;  // capture lock
  void *g_present_target = nullptr;  // DXGI's Present and Present1 themselves (install_hooks)
  void *g_present1_target = nullptr;

  // A D3D11 copy with everything but the copy itself done
  struct prepared11_t {
    bool valid = false;
    int slot = -1;
    ID3D11Texture2D *back = nullptr;  // references, released on submit or give-back
    ID3D11Texture2D *target = nullptr;  // g_cap.textures[slot] when prepared
    ID3D11DeviceContext *context = nullptr;  // g_cap.context when prepared
    IDXGIKeyedMutex *mutex = nullptr;  // held since the slot was claimed (keyed-mutex sync); null under owner words
    bool resolve = false;  // multisampled back buffer
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::uint32_t version = 0;
    std::uint32_t generation = 0;
    std::uint64_t frame_id = 0;
    ID3D11Texture2D *motion_src = nullptr;  // the frame's DLSS vectors (a ring texture) and the slot's motion texture (references), or null
    ID3D11Texture2D *motion_target = nullptr;
    IDXGIKeyedMutex *motion_mutex = nullptr;  // the motion texture's, held with the slot's (keyed-mutex sync)
    D3D11_BOX motion_box {};
  };

  struct armed11_t {
    bool armed = false;  // a copy was prepared this Present (stays set when it is given back: it still counts)
    bool boundary_seen = false;  // PresentImpl was reached for this swapchain from Present or Present1, uncontaminated
    IDXGISwapChain *swapchain = nullptr;  // borrowed: we are inside its Present
    HWND hwnd = nullptr;
    prepared11_t prepared;
  };

  thread_local armed11_t t_armed11;
  thread_local int t_nested_presents = 0;  // public Presents re-entered inside the outermost one (an overlay's test Present, say)

  struct nested_present_t {  // (scoped: a nested Present that throws must not leave the count off)
    nested_present_t() {
      ++t_nested_presents;
    }

    ~nested_present_t() {
      --t_nested_presents;
    }
  };

  void give_back_prepared11(prepared11_t &p) {
    if (!p.valid) {
      return;
    }
    if (p.mutex) {
      p.mutex->ReleaseSync(0);
    } else {
      g_block->owner[p.slot].store(gc::kOwnerNone, std::memory_order_release);
    }
    g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
    safe_release(p.mutex);
    safe_release(p.back);
    safe_release(p.target);
    safe_release(p.context);
    if (p.motion_mutex) {
      p.motion_mutex->ReleaseSync(0);
    }
    safe_release(p.motion_mutex);
    safe_release(p.motion_src);
    safe_release(p.motion_target);
    p.valid = false;
  }

  void give_back_armed11() {
    give_back_prepared11(t_armed11.prepared);
  }

  void disarm11() {
    give_back_prepared11(t_armed11.prepared);
    t_armed11 = armed11_t {};
  }

  // A resize of the swapchain inside its own Present: the prepared copy
  // holds a back-buffer reference the resize would fail on
  void cancel_prepared11_for(IDXGISwapChain *swapchain) {
    if (!t_armed11.prepared.valid || !t_armed11.swapchain) {
      return;
    }
    IUnknown *a = nullptr, *b = nullptr;
    t_armed11.swapchain->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void **>(&a));
    swapchain->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void **>(&b));
    if (a && a == b) {
      give_back_prepared11(t_armed11.prepared);
      log("Swapchain %p resized inside its Present: the frame's copy was dropped", static_cast<void *>(swapchain));
    }
    safe_release(a);
    safe_release(b);
  }

  // A function's extent from dxgi.dll's exception directory; false unless
  // `rva` is where one starts
  bool function_extent(std::uint32_t rva, std::uint32_t &begin, std::uint32_t &end) {
    DWORD64 base = 0;
    const auto address = reinterpret_cast<DWORD64>(g_dxgi_module) + rva;
    PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(address, &base, nullptr);
    if (!entry || base != reinterpret_cast<DWORD64>(g_dxgi_module) || entry->BeginAddress != rva || entry->EndAddress <= entry->BeginAddress) {
      return false;
    }
    begin = entry->BeginAddress;
    end = entry->EndAddress;
    return true;
  }

  // In `image` (dxgi.dll's file, laid out as loaded), the one direct call to
  // `target` within [begin, end): the RVA the call returns to, or 0
  std::uint32_t single_call_to(const std::uint8_t *image, std::uint32_t image_size, std::uint32_t begin, std::uint32_t end, std::uint32_t target, const char *what) {
    std::uint32_t found = 0;
    int count = 0;
    if (end > image_size || end - begin < 5 || image_size - end < 16) {
      log("PresentImpl: %s's extent (+0x%x..+0x%x) is not within the image", what, begin, end);
      return 0;
    }
    for (std::uint32_t at = begin; at < end;) {
      hde64s hs {};
      const unsigned len = hde64_disasm(image + at, &hs);
      if ((hs.flags & F_ERROR) || len == 0 || at + len > end) {
        log("PresentImpl: an instruction of %s at +0x%x could not be decoded within the function", what, at);
        return 0;
      }
      if (hs.opcode == 0xE8 && len == 5 &&
          static_cast<std::int64_t>(at) + 5 + static_cast<std::int32_t>(hs.imm.imm32) == static_cast<std::int64_t>(target)) {
        ++count;
        found = at + 5;
      }
      at += len;
    }
    if (count != 1) {
      log("PresentImpl: %s makes %d direct calls to it (exactly one expected)", what, count);
      return 0;
    }
    return found;
  }

  HRESULT WINAPI hook_present_impl(void *self, const void *args, UINT dirty_count, const RECT *dirty_rects, UINT scroll_count, const void *scroll_rects, IUnknown *resource);

  // Verifies the address the host resolved and detours PresentImpl. Capture
  // lock held; one attempt per process, made once the host has resolved it.
  void ensure_present_impl_hook() {
    if (g_present_impl_attempted || !g_block) {
      return;
    }
    const auto rva = g_block->dxgi_present_impl_rva.load(std::memory_order_acquire);
    if (!rva) {
      return;  // (the host may still be resolving it: looked at again next frame)
    }
    g_present_impl_attempted = true;
    const auto timestamp = g_block->dxgi_timestamp.load(std::memory_order_relaxed);
    const auto image_size = g_block->dxgi_image_size.load(std::memory_order_relaxed);
    const char *const fallback = "D3D11 frames are copied before Present (overlays drawn after our hook are missed)";
    if (!g_dxgi_module || !g_present_target || !g_present1_target) {
      log("PresentImpl: no DXGI module to verify it against; %s", fallback);
      return;
    }
    const auto *const base = reinterpret_cast<const std::uint8_t *>(g_dxgi_module);
    auto same_build = [&](const std::uint8_t *image, const char *which) {
      const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(image);
      const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS64 *>(image + dos->e_lfanew);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE || nt->Signature != IMAGE_NT_SIGNATURE ||
          nt->FileHeader.TimeDateStamp != timestamp || nt->OptionalHeader.SizeOfImage != image_size) {
        log("PresentImpl: the host resolved it for another dxgi.dll build than the one %s (timestamp 0x%08lx, size 0x%lx; host: 0x%08lx, 0x%lx); %s",
            which, static_cast<unsigned long>(nt->FileHeader.TimeDateStamp), static_cast<unsigned long>(nt->OptionalHeader.SizeOfImage),
            static_cast<unsigned long>(timestamp), static_cast<unsigned long>(image_size), fallback);
        return false;
      }
      return true;
    };
    if (!same_build(base, "loaded")) {
      return;
    }

    // The file, laid out as loaded: the public entries' unpatched code (in
    // memory they begin with the overlays' jumps)
    wchar_t path[MAX_PATH];
    if (!GetModuleFileNameW(g_dxgi_module, path, MAX_PATH)) {
      log("PresentImpl: cannot find dxgi.dll's path; %s", fallback);
      return;
    }
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    HANDLE mapping = file != INVALID_HANDLE_VALUE ? CreateFileMappingW(file, nullptr, PAGE_READONLY | SEC_IMAGE, 0, 0, nullptr) : nullptr;
    const auto *image = mapping ? static_cast<const std::uint8_t *>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0)) : nullptr;
    struct unmap_t {
      HANDLE file, mapping;
      const std::uint8_t *image;

      ~unmap_t() {
        if (image) {
          UnmapViewOfFile(image);
        }
        if (mapping) {
          CloseHandle(mapping);
        }
        if (file != INVALID_HANDLE_VALUE) {
          CloseHandle(file);
        }
      }
    } unmap {file, mapping, image};
    if (!image) {
      log("PresentImpl: cannot map dxgi.dll from disk (%lu); %s", GetLastError(), fallback);
      return;
    }
    if (!same_build(image, "on disk")) {
      return;
    }

    const auto present_rva = static_cast<std::uint32_t>(static_cast<const std::uint8_t *>(g_present_target) - base);
    const auto present1_rva = static_cast<std::uint32_t>(static_cast<const std::uint8_t *>(g_present1_target) - base);
    std::uint32_t pb = 0, pe = 0, p1b = 0, p1e = 0, ib = 0, ie = 0;
    if (!function_extent(present_rva, pb, pe) || !function_extent(present1_rva, p1b, p1e) || !function_extent(rva, ib, ie)) {
      log("PresentImpl: dxgi.dll's exception directory does not describe Present (+0x%x), Present1 (+0x%x) and PresentImpl (+0x%x) as functions starting there; %s",
          present_rva, present1_rva, rva, fallback);
      return;
    }
    const auto ret0 = single_call_to(image, image_size, pb, pe, rva, "Present");
    const auto ret1 = single_call_to(image, image_size, p1b, p1e, rva, "Present1");
    if (!ret0 || !ret1) {
      log("PresentImpl: %s", fallback);
      return;
    }
    // Its entry must be DXGI's own code (nothing else detoured it)
    if (ie > image_size || ie - ib < 16 || std::memcmp(base + ib, image + ib, 16) != 0) {
      log("PresentImpl: its code in memory differs from dxgi.dll on disk (detoured by something else?); %s", fallback);
      return;
    }

    void *target = const_cast<std::uint8_t *>(base) + rva;
    if (MH_CreateHook(target, reinterpret_cast<void *>(&hook_present_impl), reinterpret_cast<void **>(&g_real_present_impl)) != MH_OK ||
        MH_EnableHook(target) != MH_OK) {
      log("PresentImpl: could not be detoured; %s", fallback);
      return;
    }
    g_present_impl_return[0] = const_cast<std::uint8_t *>(base) + ret0;
    g_present_impl_return[1] = const_cast<std::uint8_t *>(base) + ret1;
    g_present_impl_ready.store(true, std::memory_order_release);
    log("dxgi!CDXGISwapChain::PresentImpl at +0x%x, called from Present at +0x%x and Present1 at +0x%x (verified against dxgi.dll on disk): D3D11 frames are copied there, after overlays",
        rva, ret0 - 5, ret1 - 5);
  }

  // Whether this swapchain's D3D11 frames are copied at the boundary
  bool late_copy11(IDXGISwapChain *swapchain) {
    // (the prepared copy holds a back-buffer reference across the overlay
    // chain: a resize made there must reach our ResizeBuffers detours)
    if (!g_present_impl_ready.load(std::memory_order_acquire) || g_submit_mode_broken.load(std::memory_order_acquire) ||
        !g_real_resize_buffers || !g_real_resize_buffers1) {
      return false;
    }
    const auto mode = static_cast<submit_mode_e>(swapchain_count(swapchain, kSubmitModeKey) & 0xff);
    if (mode == submit_mode_e::abandoned) {
      return false;
    }
    if (mode != submit_mode_e::at_present_impl) {
      set_swapchain_count(swapchain, kSubmitModeKey, static_cast<std::uint32_t>(submit_mode_e::at_present_impl));
      if (g_submit_mode_broken.load(std::memory_order_acquire)) {
        return false;
      }
      log("Swapchain %p: D3D11 frames are copied at PresentImpl, after overlays", static_cast<void *>(swapchain));
    }
    return true;
  }

  // The copy itself and its publication. Capture lock held; `p` prepared
  // against the current g_cap (checked by the caller); consumed.
  void submit11(prepared11_t p, IDXGISwapChain *swapchain, bool at_boundary) {
    const int slot = p.slot;
    if (p.resolve) {
      p.context->ResolveSubresource(p.target, 0, p.back, 0, p.format);
    } else {
      p.context->CopyResource(p.target, p.back);
    }
    if (p.motion_src && p.motion_target) {
      // (before the slot is let go: the host reads the vectors under it)
      p.context->CopySubresourceRegion(p.motion_target, 0, 0, 0, 0, p.motion_src, 0, &p.motion_box);
    }
    if (p.motion_mutex) {
      p.motion_mutex->ReleaseSync(0);
    }
    if (p.mutex) {
      p.mutex->ReleaseSync(0);
    } else {
      // The copy is still in flight, but the host takes only a published
      // slot, and this one publishes after its fence completes
      g_block->owner[slot].store(gc::kOwnerNone, std::memory_order_release);
    }

    auto &s = g_slots[slot];
    const auto ticket = mark_pending(slot, p.version, p.generation, p.frame_id, (g_cap.fence && g_cap.context4) ? ++g_cap.fence_value : 0);

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
        p.context->Flush();
      }
    }
    if (!signalled) {
      if (s.fence_value) {
        drop_pending(slot, ticket);
      } else {
        SetEvent(s.done_event);
      }
    }
    safe_release(p.mutex);
    safe_release(p.back);
    safe_release(p.target);
    safe_release(p.context);
    safe_release(p.motion_mutex);
    safe_release(p.motion_src);
    safe_release(p.motion_target);
    note_capturing(swapchain, at_boundary ? "D3D11 (at PresentImpl, overlays included)" : "D3D11");
  }

  // Everything but the copy: the device, textures, a slot and its record.
  // Capture lock held. The copy follows at once, or at the boundary.
  void capture_frame11(IDXGISwapChain *swapchain, ID3D11Device *device, HWND hwnd, std::uint64_t present_qpc, std::uint64_t release_qpc) {
    ensure_device(swapchain, device, hwnd);
    ensure_present_impl_hook();

    ID3D11Texture2D *back = nullptr;
    if (FAILED(swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&back)))) {
      return;
    }
    D3D11_TEXTURE2D_DESC back_desc {};
    back->GetDesc(&back_desc);
    // The DLSS vectors this frame carries (their ring's shape also says what
    // motion textures the generation needs)
    motion_pick_t pick;
    motion_want_t want;
    motion_pick(gc::api_e::d3d11, device, nullptr, pick, want);
    struct release_pick_t {
      motion_pick_t &p;

      ~release_pick_t() {
        p.release();
      }
    } release_pick {pick};
    if (!ensure_textures(back_desc, hwnd, sharing_e::any, true, want)) {
      back->Release();
      return;
    }
    const std::uint32_t color_space = swapchain_color_space(swapchain);

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

    // Holding the slot (its pixel mutex, or its owner word): rewrite its
    // record, then the pixels. The host's acquire is GPU-ordered after the
    // release in submit11, so a host holding the mutex and reading an even
    // record reads this frame's.
    prepared11_t p;
    p.valid = true;
    p.slot = slot;
    p.back = back;  // (the reference taken above)
    p.target = g_cap.textures[slot];
    p.target->AddRef();
    p.context = g_cap.context;
    p.context->AddRef();
    if (!g_cap.owner_sync) {
      p.mutex = g_cap.mutexes[slot];
      p.mutex->AddRef();
    }
    p.resolve = back_desc.SampleDesc.Count > 1;
    p.format = g_cap.format;
    bool motion = motion_fits(pick) && g_cap.motion11[slot];
    if (motion && !g_cap.owner_sync) {
      // The motion texture's own mutex, free whenever the slot's is (both go
      // back together); anything else costs this frame its vectors
      const HRESULT hr = g_cap.motion_mutexes[slot] ? g_cap.motion_mutexes[slot]->AcquireSync(0, 0) : E_FAIL;
      if (hr == static_cast<HRESULT>(WAIT_ABANDONED)) {
        g_cap.motion_mutexes[slot]->ReleaseSync(0);
      }
      if (hr == S_OK) {
        p.motion_mutex = g_cap.motion_mutexes[slot];
        p.motion_mutex->AddRef();
      } else {
        motion = false;
      }
    }
    if (motion) {
      p.motion_src = pick.src11;
      p.motion_src->AddRef();
      p.motion_target = g_cap.motion11[slot];
      p.motion_target->AddRef();
      p.motion_box = {0, 0, 0, pick.width, pick.height, 1};
    }
    p.generation = g_current_generation.load(std::memory_order_acquire);
    p.frame_id = ++g_cap.next_frame_id;
    p.version = write_slot_record(slot, p.generation, color_space, p.frame_id, present_qpc, release_qpc, motion ? &pick : nullptr);

    if (!late_copy11(swapchain)) {
      submit11(p, swapchain, false);
      return;
    }
    disarm11();  // (never two)
    t_armed11.armed = true;
    t_armed11.swapchain = swapchain;
    t_armed11.hwnd = hwnd;
    t_armed11.prepared = p;
  }

  // From the PresentImpl detour: reached for the armed swapchain from
  // Present or Present1 during its Present. The copy goes now, unless
  // another swapchain's Present intervened or the capture changed since.
  void run_armed_capture11() {
    auto &p = t_armed11.prepared;
    if (t_seen_contaminated) {
      give_back_prepared11(p);
      return;
    }
    t_armed11.boundary_seen = true;
    if (!p.valid) {
      return;
    }
    // Never wait here: a capture elsewhere holding the lock costs this
    // frame, not a stall inside Present
    struct lock_t {
      bool held = TryAcquireSRWLockExclusive(&g_capture_lock);

      ~lock_t() {
        if (held) {
          ReleaseSRWLockExclusive(&g_capture_lock);
        }
      }
    } lock;
    if (!lock.held) {
      give_back_prepared11(p);
      return;
    }
    const bool current = g_cap.api == gc::api_e::d3d11 && g_cap.swapchain == t_armed11.swapchain && g_cap.context == p.context &&
                         g_cap.textures[p.slot] == p.target && g_current_generation.load(std::memory_order_acquire) == p.generation &&
                         on_capture_thread();
    if (!current) {
      give_back_prepared11(p);
      return;
    }
    const prepared11_t submitting = p;
    p.valid = false;  // (consumed)
    submit11(submitting, t_armed11.swapchain, true);
  }

  // After the real Present (D3D11)
  void settle_armed11(IDXGISwapChain *swapchain, bool presented) {
    if (!t_armed11.armed) {
      return;
    }
    const bool seen = t_armed11.boundary_seen;
    disarm11();
    if (presented) {
      count_boundary(swapchain, submit_mode_e::at_present_impl, seen, "PresentImpl was not reached");
    }
  }

  HRESULT WINAPI hook_present_impl(void *self, const void *args, UINT dirty_count, const RECT *dirty_rects, UINT scroll_count, const void *scroll_rects, IUnknown *resource) {
    // Ours: the outermost Present's own call (a Present re-entered through
    // the public entry inside it, an overlay's test Present of the same
    // swapchain, must not take the copy), of the whole back buffer
    if (t_armed11.armed && t_presenting && t_nested_presents == 0 && self == static_cast<void *>(t_armed11.swapchain) &&
        dirty_count == 0 && scroll_count == 0 && resource == nullptr) {
      void *const from = __builtin_return_address(0);
      if (from == g_present_impl_return[0] || from == g_present_impl_return[1]) {
        run_armed_capture11();
      }
    }
    return g_real_present_impl(self, args, dirty_count, dirty_rects, scroll_count, scroll_rects, resource);
  }

  void capture_frame(IDXGISwapChain *swapchain, std::uint64_t present_qpc, std::uint64_t release_qpc, bool partial) {
    g_block->frames_presented.fetch_add(1, std::memory_order_relaxed);
    motion_on_present(swapchain);  // (every Present, captured or not: see the motion section)
    if (!g_block->capture_enabled.load(std::memory_order_acquire)) {
      housekeeping(swapchain);
      return;
    }
    HWND hwnd = nullptr;
    if (!foreground_window(swapchain, hwnd)) {
      housekeeping(swapchain);
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
    if (!on_capture_thread()) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    if (!reserve_retirement()) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      return;  // out of memory: nothing may be retired this frame
    }
    if (partial) {
      // Present1 with dirty or scroll rectangles: the back buffer holds
      // only what changed
      static bool logged = false;
      if (!logged) {
        logged = true;
        log("Present1 with dirty or scroll rectangles: such frames are not captured");
      }
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
      IDXGISwapChain3 *swapchain3 = nullptr;
      if (submit_mode(swapchain) == submit_mode_e::at_dxgi_submit && g_real_execute_command_lists &&
          SUCCEEDED(swapchain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&swapchain3)))) {
        // Copied from inside the real Present, when DXGI submits (see the
        // D3D12 section): the buffer is the one being presented now
        swapchain3->Release();
        disarm_capture();
        prepared12_t prepared;
        capture_frame12(swapchain, device12, hwnd, present_qpc, release_qpc, &prepared);
        device12->Release();
        if (prepared.valid) {
          t_armed.armed = true;
          t_armed.swapchain = swapchain;
          t_armed.hwnd = hwnd;
          t_armed.prepared = prepared;
        }
        return;
      }
      capture_frame12(swapchain, device12, hwnd, present_qpc, release_qpc);
      device12->Release();
      return;
    }
    capture_frame11(swapchain, device, hwnd, present_qpc, release_qpc);
    device->Release();
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
  HRESULT present_with_capture(IDXGISwapChain *swapchain, UINT sync_interval, UINT flags, bool partial, F &&real) {
    if (flags & DXGI_PRESENT_TEST) {
      const HRESULT hr = real(sync_interval);
      if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        drop_lost_capture(swapchain);
      }
      return hr;
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

    // Armed copies are settled in the observation scope below; this covers an
    // exception between arming (in capture_frame) and entering it
    struct armed_guard_t {
      ~armed_guard_t() {
        disarm_capture();
        disarm11();
      }
    } armed_guard;

    // The back buffer this Present shows (queue learning confirms per buffer)
    t_present_index = UINT_MAX;
    if (IDXGISwapChain3 *swapchain3 = nullptr; SUCCEEDED(swapchain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&swapchain3)))) {
      t_present_index = swapchain3->GetCurrentBackBufferIndex();
      swapchain3->Release();
    }

    capture_frame(swapchain, now, release, partial);

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
        safe_release(t_seen_dxgi_queue);
        t_seen_several = false;
        t_seen_dxgi_several = false;
        t_seen_contaminated = false;
        disarm_capture();  // (normally settled already; this covers an exception)
        disarm11();
      }
    };

    HRESULT hr;
    {
      observation_t observing(swapchain);
      hr = real(paced ? 0 : sync_interval);
      t_presenting = nullptr;
      settle_armed_capture(swapchain, SUCCEEDED(hr));
      settle_armed11(swapchain, SUCCEEDED(hr));
      if (t_seen_dxgi_queue && SUCCEEDED(hr) && !t_seen_contaminated) {
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

  void note_nested(IDXGISwapChain *swapchain, UINT flags) {
    if (!t_presenting) {
      return;
    }
    if (t_presenting == swapchain) {
      // The same swapchain presented again inside its Present (an overlay
      // forwarding through the public entry): a real presentation rotates
      // the back buffer under an armed copy, so the copy is dropped; a test
      // Present (nothing presented) leaves it
      if (!(flags & DXGI_PRESENT_TEST)) {
        give_back_prepared(t_armed.prepared);
        give_back_armed11();
      }
      return;
    }
    // (DXGI may call itself through another interface of the same object)
    IUnknown *outer = nullptr, *inner = nullptr;
    t_presenting->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void **>(&outer));
    swapchain->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void **>(&inner));
    if (!outer || outer != inner) {
      t_seen_contaminated = true;
      give_back_prepared(t_armed.prepared);  // its DXGI submission must not carry our copy (the attempt still counts)
      give_back_armed11();
    }
    safe_release(outer);
    safe_release(inner);
  }

  HRESULT STDMETHODCALLTYPE hook_present(IDXGISwapChain *swapchain, UINT sync_interval, UINT flags) {
    if (t_in_present) {
      note_nested(swapchain, flags);
      nested_present_t nested;
      return g_real_present(swapchain, sync_interval, flags);
    }
    in_present_t in_present;
    return present_with_capture(swapchain, sync_interval, flags, false, [&](UINT interval) {
      return g_real_present(swapchain, interval, flags);
    });
  }

  HRESULT STDMETHODCALLTYPE hook_present1(IDXGISwapChain1 *swapchain, UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS *params) {
    if (t_in_present) {
      note_nested(swapchain, flags);
      nested_present_t nested;
      return g_real_present1(swapchain, sync_interval, flags, params);
    }
    in_present_t in_present;
    const bool partial = params && (params->DirtyRectsCount > 0 || params->pScrollRect != nullptr);
    return present_with_capture(swapchain, sync_interval, flags, partial, [&](UINT interval) {
      return g_real_present1(swapchain, interval, flags, params);
    });
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
      // (DLSS motion vectors: without the Reset detour no D3D12 evaluation is used)
      ID3D12CommandAllocator *allocator = nullptr;
      ID3D12GraphicsCommandList *list = nullptr;
      if (SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void **>(&allocator))) &&
          SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void **>(&list)))) {
        void **list_vtable = *reinterpret_cast<void ***>(list);
        if (MH_CreateHook(list_vtable[kVtListReset], reinterpret_cast<void *>(&hook_list_reset), reinterpret_cast<void **>(&g_real_list_reset)) != MH_OK) {
          g_real_list_reset = nullptr;
          log("ID3D12GraphicsCommandList::Reset could not be detoured; D3D12 DLSS motion vectors are not used");
        }
        list->Close();
      }
      safe_release(list);
      safe_release(allocator);
    } else {
      log("No D3D12 device for the queue detour: 0x%08lx", hr);
    }
    safe_release(queue);
    safe_release(device);
  }

  // The entry points are found from a throwaway swapchain and inline-detoured
  // at their addresses, which every swapchain of the process shares.
  // Whether the swapchain's implementation lives in Windows' dxgi.dll (then
  // `module` is that module: the one DXGI's own submissions are checked against)
  bool windows_dxgi(IDXGISwapChain1 *swapchain, HMODULE &module) {
    module = nullptr;
    void *present = (*reinterpret_cast<void ***>(swapchain))[kVtPresent];
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<LPCWSTR>(present), &module)) {
      return false;
    }
    wchar_t path[MAX_PATH], system[MAX_PATH];
    const UINT n = GetSystemDirectoryW(system, MAX_PATH);
    if (!GetModuleFileNameW(module, path, MAX_PATH) || n == 0 || n + 10 >= MAX_PATH) {
      return false;
    }
    std::wcscat(system, L"\\dxgi.dll");
    log("DXGI: %ls", path);
    if (_wcsicmp(path, system) != 0) {
      module = nullptr;
      return false;
    }
    return true;
  }

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
    if (SUCCEEDED(hr) && !windows_dxgi(swapchain, g_dxgi_module)) {
      // A replacement DXGI (DXVK, say) has other internals than the ones
      // this hook relies on
      if (g_vk_layer_present) {
        log("The process uses a DXGI other than Windows' own (DXVK?): only its Vulkan swapchains are captured");
      } else {
        set_state(gc::hook_state_e::unsupported, "The process uses a DXGI other than Windows' own");
      }
      hr = E_NOTIMPL;
    } else if (DXGI_COLOR_SPACE_TYPE cs; SUCCEEDED(hr) && !dxgi_color_space(swapchain, cs)) {
      hr = E_NOINTERFACE;  // (fatal, reported: without the getter no frame can be interpreted)
    } else if (SUCCEEDED(hr)) {
      void **vtable = *reinterpret_cast<void ***>(swapchain);
      g_present_target = vtable[kVtPresent];
      g_present1_target = vtable[kVtPresent1];
      ok = MH_Initialize() == MH_OK &&
           MH_CreateHook(vtable[kVtPresent], reinterpret_cast<void *>(&hook_present), reinterpret_cast<void **>(&g_real_present)) == MH_OK &&
           MH_CreateHook(vtable[kVtPresent1], reinterpret_cast<void *>(&hook_present1), reinterpret_cast<void **>(&g_real_present1)) == MH_OK;
      IDXGISwapChain3 *swapchain3 = nullptr;
      if (ok && SUCCEEDED(swapchain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&swapchain3)))) {
        void **vtable3 = *reinterpret_cast<void ***>(swapchain3);
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
      if (ok) {
        AcquireSRWLockExclusive(&g_capture_lock);
        ensure_present_impl_hook();  // (if the host has resolved it by now; else at the first D3D11 frame after it has)
        ReleaseSRWLockExclusive(&g_capture_lock);
      }
      if (!ok) {
        MH_DisableHook(MH_ALL_HOOKS);
        set_state(gc::hook_state_e::failed, "Installing the DXGI detours failed");
      }
    } else if (hr != E_NOTIMPL && hr != E_NOINTERFACE) {
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

  // ---- Vulkan ----------------------------------------------------------------
  //
  // Vulkan frames come through Vibepollo's Vulkan layer (tools/vk_layer), which
  // the loader puts in every Vulkan process: it made the swapchain images
  // copyable and recorded them, and calls us inside vkQueuePresentKHR. There,
  // for the captured swapchain, a copy of the presented image into a slot is
  // submitted on the presenting queue: it waits on the semaphores the Present
  // would have waited on (the game's rendering) and signals one of ours that
  // the Present waits on instead, so it runs after the frame is rendered and
  // before it is shown. Registered below Steam's overlay layer, the frame
  // already holds the overlay by then.
  //
  // The slots are D3D11 textures of a device of our own on the game's GPU
  // (Windows' own d3d11.dll, also under DXVK; the host opens them as any
  // other), shared by owner words and imported into Vulkan as external
  // memory; a Vulkan fence per slot, waited on by the fence waiter thread,
  // says when a copy is done and feeds the completion thread like a D3D fence
  // would. Everything runs on the chain below our layer (the layer's
  // next_gdpa), never through layers above it.
  //
  // Nothing after a successful submission may fail or allocate: from then on
  // the Present's own waits are consumed, and it must wait on ours.

  struct vk_fn_t {
#define VVK_FN(name) PFN_vk##name name = nullptr;
    VVK_FN(CreateImage)
    VVK_FN(DestroyImage)
    VVK_FN(GetImageMemoryRequirements2)
    VVK_FN(GetMemoryWin32HandlePropertiesKHR)
    VVK_FN(AllocateMemory)
    VVK_FN(FreeMemory)
    VVK_FN(BindImageMemory2)
    VVK_FN(CreateCommandPool)
    VVK_FN(DestroyCommandPool)
    VVK_FN(AllocateCommandBuffers)
    VVK_FN(ResetCommandBuffer)
    VVK_FN(BeginCommandBuffer)
    VVK_FN(EndCommandBuffer)
    VVK_FN(CmdPipelineBarrier)
    VVK_FN(CmdCopyImage)
    VVK_FN(QueueSubmit)
    VVK_FN(CreateFence)
    VVK_FN(DestroyFence)
    VVK_FN(ResetFences)
    VVK_FN(WaitForFences)
    VVK_FN(CreateSemaphore)
    VVK_FN(DestroySemaphore)
#undef VVK_FN
  };

  // Objects of a device we could not destroy yet (copies in flight when they
  // went, or semaphores a Present may still wait on): destroyed with the
  // device, once nothing of ours is in flight
  struct vk_leftovers_t {
    VkDevice device = VK_NULL_HANDLE;
    vk_fn_t fn;
    std::vector<VkImage> images;
    std::vector<VkDeviceMemory> memories;
    std::vector<VkFence> fences;
    std::vector<VkCommandPool> pools;
    std::vector<VkSemaphore> semaphores;
    std::vector<ID3D11Texture2D *> textures;  // pinned for global-handle imports (released after the images)
  };

  struct vk_state_t {
    vvk::device_t dev {};  // the device captured on (a copy of the layer's record)
    vk_fn_t fn;
    PFN_vkGetPhysicalDeviceMemoryProperties memory_properties = nullptr;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::uint32_t family = UINT32_MAX;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd[gc::kSlots] = {};
    VkFence fence[gc::kSlots] = {};
    VkImage image[gc::kSlots] = {};  // the slots' textures, imported
    VkDeviceMemory memory[gc::kSlots] = {};
    bool acquired_once[gc::kSlots] = {};  // the image went through its first (discarding) acquisition
    ID3D11Texture2D *imported_from[gc::kSlots] = {};  // (identity: which texture set the images alias)
    // A global-handle import does not keep the texture alive: we do, until
    // the image is destroyed (NT imports hold a reference of their own)
    ID3D11Texture2D *pinned[gc::kSlots] = {};
    std::vector<VkSemaphore> present_waits;  // one per swapchain image, signalled by our copy, waited by the Present
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    VkExternalMemoryHandleTypeFlagBits handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    bool kmt_importable = false;
  };

  vk_state_t g_vk;  // capture lock
  std::vector<vk_leftovers_t> g_vk_leftovers;  // capture lock
  bool g_vk_attached = false;

  // Copies submitted, in submission order: at most one per slot, so a fixed
  // ring (nothing allocates once a copy is submitted)
  struct vk_pending_t {
    VkDevice device;
    PFN_vkWaitForFences wait;
    VkFence fence;
    std::uint64_t value;
    int slot;
  };

  constexpr int kVkRing = gc::kSlots + 1;
  SRWLOCK g_vk_queue_lock = SRWLOCK_INIT;
  vk_pending_t g_vk_ring[kVkRing];
  int g_vk_ring_head = 0, g_vk_ring_count = 0;
  HANDLE g_vk_wake = nullptr;
  std::atomic<bool> g_vk_slot_busy[gc::kSlots] = {};  // a copy of this slot is in flight
  std::atomic<int> g_vk_in_flight {0};
  // Fence values: handed out by vk_pre_present, never reused (a stale
  // completion event of an earlier device cannot vouch for a newer copy)
  std::atomic<std::uint64_t> g_vk_next_value {0};
  std::atomic<VkDevice> g_vk_current_device {VK_NULL_HANDLE};  // the device captured on now (a loss of another says nothing about this one's copies)

  DWORD WINAPI vk_waiter_main(void *) {
    for (;;) {
      WaitForSingleObject(g_vk_wake, 100);
      for (;;) {
        vk_pending_t p {};
        AcquireSRWLockExclusive(&g_vk_queue_lock);
        const bool any = g_vk_ring_count > 0;
        if (any) {
          p = g_vk_ring[g_vk_ring_head];
          g_vk_ring_head = (g_vk_ring_head + 1) % kVkRing;
          --g_vk_ring_count;
        }
        ReleaseSRWLockExclusive(&g_vk_queue_lock);
        if (!any) {
          break;
        }
        bool lost = false;
        for (;;) {
          const VkResult r = p.wait(p.device, 1, &p.fence, VK_TRUE, 100'000'000ull);  // 100 ms, then look again
          if (r == VK_SUCCESS) {
            break;
          }
          if (r == VK_ERROR_DEVICE_LOST) {
            lost = true;
            break;
          }
          if (r != VK_TIMEOUT) {
            Sleep(10);  // (out of memory: the copy is not finished, try again)
          }
        }
        if (lost) {
          // (under the queue lock: a device switch, which re-bases the
          // completed value, cannot slip between the check and the store)
          AcquireSRWLockExclusive(&g_vk_queue_lock);
          if (p.device == g_vk_current_device.load(std::memory_order_relaxed)) {
            g_vk_completed.store(UINT64_MAX, std::memory_order_release);  // (nothing more completes)
          }
          ReleaseSRWLockExclusive(&g_vk_queue_lock);
        } else {
          auto done = g_vk_completed.load(std::memory_order_relaxed);
          while (done != UINT64_MAX && done < p.value && !g_vk_completed.compare_exchange_weak(done, p.value, std::memory_order_acq_rel)) {
          }
        }
        g_vk_slot_busy[p.slot].store(false, std::memory_order_release);
        g_vk_in_flight.fetch_sub(1, std::memory_order_acq_rel);
        SetEvent(g_slots[p.slot].done_event);
      }
    }
  }

  bool vk_slot_ready(int slot) {
    return !g_vk_slot_busy[slot].load(std::memory_order_acquire);
  }

  // Waits for every copy in flight: for `ms` (0: until done, logging while
  // it takes long); false if some did not finish
  bool vk_wait_idle(std::uint64_t ms) {
    const auto start = GetTickCount64();
    auto logged = start;
    while (g_vk_in_flight.load(std::memory_order_acquire) > 0) {
      const auto now = GetTickCount64();
      if (ms && now - start >= ms) {
        return false;
      }
      if (!ms && now - logged >= 2000) {
        logged = now;
        log("Vulkan: still waiting for %d copies before the device goes", g_vk_in_flight.load());
      }
      Sleep(1);
    }
    return true;
  }

  vk_leftovers_t *vk_leftovers_for(VkDevice device, const vk_fn_t &fn) {
    for (auto &l : g_vk_leftovers) {
      if (l.device == device) {
        return &l;
      }
    }
    try {
      auto &l = g_vk_leftovers.emplace_back();
      l.device = device;
      l.fn = fn;
      return &l;
    } catch (...) {
      return nullptr;  // (then they leak)
    }
  }

  void vk_destroy_leftovers(vk_leftovers_t &l) {
    for (VkImage i : l.images) {
      l.fn.DestroyImage(l.device, i, nullptr);
    }
    for (VkDeviceMemory m : l.memories) {
      l.fn.FreeMemory(l.device, m, nullptr);
    }
    for (VkFence f : l.fences) {
      l.fn.DestroyFence(l.device, f, nullptr);
    }
    for (VkCommandPool p : l.pools) {
      l.fn.DestroyCommandPool(l.device, p, nullptr);
    }
    for (VkSemaphore s : l.semaphores) {
      l.fn.DestroySemaphore(l.device, s, nullptr);
    }
    for (ID3D11Texture2D *t : l.textures) {
      t->Release();
    }
    l = vk_leftovers_t {};
  }

  // Lets go of the captured swapchain's objects: destroyed now if nothing
  // of ours is in flight (`device_gone`: waits for that), else kept for the
  // device's end; the Present semaphores always wait for it. Capture lock held.
  void vk_release(bool device_gone) {
    auto &v = g_vk;
    if (!v.dev.device) {
      return;
    }
    const bool idle = vk_wait_idle(device_gone ? 0 : 2000);
    vk_leftovers_t *left = vk_leftovers_for(v.dev.device, v.fn);
    try {
      if (left) {
        left->semaphores.insert(left->semaphores.end(), v.present_waits.begin(), v.present_waits.end());
      }
      for (int i = 0; i < gc::kSlots; ++i) {
        if (idle) {
          if (v.image[i]) {
            v.fn.DestroyImage(v.dev.device, v.image[i], nullptr);
          }
          if (v.memory[i]) {
            v.fn.FreeMemory(v.dev.device, v.memory[i], nullptr);
          }
          if (v.fence[i]) {
            v.fn.DestroyFence(v.dev.device, v.fence[i], nullptr);
          }
          safe_release(v.pinned[i]);
        } else if (left) {
          if (v.pinned[i]) {
            left->textures.push_back(v.pinned[i]);
            v.pinned[i] = nullptr;
          }
          if (v.image[i]) {
            left->images.push_back(v.image[i]);
          }
          if (v.memory[i]) {
            left->memories.push_back(v.memory[i]);
          }
          if (v.fence[i]) {
            left->fences.push_back(v.fence[i]);
          }
        }
      }
      if (v.pool) {
        if (idle) {
          v.fn.DestroyCommandPool(v.dev.device, v.pool, nullptr);  // (frees its command buffers)
        } else if (left) {
          left->pools.push_back(v.pool);
        }
      }
    } catch (...) {
      // (what did not fit in the leftovers leaks, rather than be destroyed in use)
    }
    if (!idle) {
      log("Vulkan: copies still in flight after 2 s; their objects wait for the device's end");
    }
    AcquireSRWLockExclusive(&g_vk_queue_lock);
    g_vk_current_device.store(VK_NULL_HANDLE, std::memory_order_relaxed);
    ReleaseSRWLockExclusive(&g_vk_queue_lock);
    if (device_gone) {
      for (auto it = g_vk_leftovers.begin(); it != g_vk_leftovers.end(); ++it) {
        if (it->device == v.dev.device) {
          vk_destroy_leftovers(*it);
          g_vk_leftovers.erase(it);
          break;
        }
      }
    }
    v = vk_state_t {};
  }

  bool vk_formats(VkFormat format, DXGI_FORMAT &dxgi, VkFormat &unorm) {
    switch (format) {
      case VK_FORMAT_B8G8R8A8_UNORM:
      case VK_FORMAT_B8G8R8A8_SRGB:
        dxgi = DXGI_FORMAT_B8G8R8A8_UNORM;
        unorm = VK_FORMAT_B8G8R8A8_UNORM;
        return true;
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_R8G8B8A8_SRGB:
        dxgi = DXGI_FORMAT_R8G8B8A8_UNORM;
        unorm = VK_FORMAT_R8G8B8A8_UNORM;
        return true;
      case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        dxgi = DXGI_FORMAT_R10G10B10A2_UNORM;
        unorm = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        return true;
      case VK_FORMAT_R16G16B16A16_SFLOAT:
        dxgi = DXGI_FORMAT_R16G16B16A16_FLOAT;
        unorm = VK_FORMAT_R16G16B16A16_SFLOAT;
        return true;
      default:
        return false;
    }
  }

  // The colour space is part of the swapchain's creation in Vulkan: exact
  std::uint32_t vk_color_space(VkColorSpaceKHR cs, DXGI_FORMAT format) {
    const bool fp16 = format == DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (cs == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && !fp16) {
      return static_cast<std::uint32_t>(gc::color_space_e::srgb);
    }
    if (cs == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT && fp16) {
      return static_cast<std::uint32_t>(gc::color_space_e::scrgb);
    }
    if (cs == VK_COLOR_SPACE_HDR10_ST2084_EXT && format == DXGI_FORMAT_R10G10B10A2_UNORM) {
      return static_cast<std::uint32_t>(gc::color_space_e::hdr10);
    }
    static std::atomic<bool> warned {false};
    if (!warned.exchange(true)) {
      log("Vulkan: colour space %d with format %d is not one the host converts; its frames are not used", static_cast<int>(cs), static_cast<int>(format));
    }
    return static_cast<std::uint32_t>(gc::color_space_e::unknown);
  }

  // Windows' own D3D11 and DXGI, whatever the process imports (DXVK replaces
  // them with its own, which cannot share textures with the host)
  template<class F>
  F system_function(const wchar_t *dll, const char *name) {
    wchar_t path[MAX_PATH];
    const UINT n = GetSystemDirectoryW(path, MAX_PATH);
    if (n == 0 || n + std::wcslen(dll) + 2 >= MAX_PATH) {
      return nullptr;
    }
    std::wcscat(path, L"\\");
    std::wcscat(path, dll);
    HMODULE module = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    return module ? reinterpret_cast<F>(reinterpret_cast<void *>(GetProcAddress(module, name))) : nullptr;
  }

  enum class vk_setup_e {
    ok,
    transient,  ///< may work on a later Present
    permanent,  ///< this device cannot be captured
  };

  // Our D3D11 device on the Vulkan device's adapter, the objects the copies
  // need and which handle kind Vulkan can import. All or nothing: the
  // capture is installed only once everything exists. Capture lock held.
  vk_setup_e vk_setup(const vvk::present_t &p, HWND hwnd, char (&why)[160]) {
    auto &v = g_vk;
    const auto &d = *p.device;
    if (!d.external_memory_win32 || d.api_version < VK_API_VERSION_1_1) {
      std::snprintf(why, sizeof(why), "the device has no VK_KHR_external_memory_win32 or is older than Vulkan 1.1");
      return vk_setup_e::permanent;
    }
    vk_fn_t fn;
    bool all = true;
#define VVK_FN(name)                                                                        \
  fn.name = reinterpret_cast<PFN_vk##name>(d.next_gdpa(d.device, "vk" #name));             \
  all = all && fn.name != nullptr;
    VVK_FN(CreateImage)
    VVK_FN(DestroyImage)
    VVK_FN(GetImageMemoryRequirements2)
    VVK_FN(GetMemoryWin32HandlePropertiesKHR)
    VVK_FN(AllocateMemory)
    VVK_FN(FreeMemory)
    VVK_FN(BindImageMemory2)
    VVK_FN(CreateCommandPool)
    VVK_FN(DestroyCommandPool)
    VVK_FN(AllocateCommandBuffers)
    VVK_FN(ResetCommandBuffer)
    VVK_FN(BeginCommandBuffer)
    VVK_FN(EndCommandBuffer)
    VVK_FN(CmdPipelineBarrier)
    VVK_FN(CmdCopyImage)
    VVK_FN(QueueSubmit)
    VVK_FN(CreateFence)
    VVK_FN(DestroyFence)
    VVK_FN(ResetFences)
    VVK_FN(WaitForFences)
    VVK_FN(CreateSemaphore)
    VVK_FN(DestroySemaphore)
#undef VVK_FN
    const auto props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(d.next_gipa(d.instance, "vkGetPhysicalDeviceProperties2"));
    const auto format_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties2>(d.next_gipa(d.instance, "vkGetPhysicalDeviceImageFormatProperties2"));
    const auto memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(d.next_gipa(d.instance, "vkGetPhysicalDeviceMemoryProperties"));
    if (!all || !props2 || !format_props2 || !memory_properties) {
      std::snprintf(why, sizeof(why), "a Vulkan function the copy needs is missing");
      return vk_setup_e::permanent;
    }

    // The adapter: D3D11 textures must live on the GPU that copies into them
    VkPhysicalDeviceIDProperties id {};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 props {};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &id;
    props2(d.physical_device, &props);
    if (!id.deviceLUIDValid) {
      std::snprintf(why, sizeof(why), "the device reports no adapter LUID");
      return vk_setup_e::permanent;
    }
    LUID luid;
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));

    // Which handle kinds the device imports D3D11 textures from
    DXGI_FORMAT dxgi = DXGI_FORMAT_UNKNOWN;
    VkFormat unorm = VK_FORMAT_UNDEFINED;
    if (!vk_formats(p.swapchain->format, dxgi, unorm)) {
      std::snprintf(why, sizeof(why), "swapchain format %d has no D3D11 equivalent the host reads", static_cast<int>(p.swapchain->format));
      return vk_setup_e::transient;  // (another swapchain may be fine)
    }
    auto importable = [&](VkExternalMemoryHandleTypeFlagBits type) {
      VkPhysicalDeviceExternalImageFormatInfo ext {};
      ext.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
      ext.handleType = type;
      VkPhysicalDeviceImageFormatInfo2 info {};
      info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
      info.pNext = &ext;
      info.format = unorm;
      info.type = VK_IMAGE_TYPE_2D;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      VkExternalImageFormatProperties ext_props {};
      ext_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
      VkImageFormatProperties2 out {};
      out.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
      out.pNext = &ext_props;
      return format_props2(d.physical_device, &info, &out) == VK_SUCCESS &&
             (ext_props.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);
    };
    const bool nt = importable(VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT);
    const bool kmt = importable(VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT);
    if (!nt && !kmt) {
      std::snprintf(why, sizeof(why), "the device cannot import D3D11 textures of format %d", static_cast<int>(unorm));
      return vk_setup_e::transient;
    }

    // Our D3D11 device (it only creates the shared textures)
    using create_factory_fn = HRESULT(WINAPI *)(REFIID, void **);
    const auto create_factory = system_function<create_factory_fn>(L"dxgi.dll", "CreateDXGIFactory1");
    const auto create_device = system_function<PFN_D3D11_CREATE_DEVICE>(L"d3d11.dll", "D3D11CreateDevice");
    if (!create_factory || !create_device) {
      std::snprintf(why, sizeof(why), "Windows' d3d11.dll or dxgi.dll could not be loaded");
      return vk_setup_e::permanent;
    }
    IDXGIFactory4 *factory = nullptr;
    IDXGIAdapter *adapter = nullptr;
    ID3D11Device *device = nullptr;
    HRESULT hr = create_factory(__uuidof(IDXGIFactory4), reinterpret_cast<void **>(&factory));
    if (SUCCEEDED(hr)) {
      hr = factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter), reinterpret_cast<void **>(&adapter));
    }
    if (SUCCEEDED(hr)) {
      hr = create_device(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, nullptr);
    }
    safe_release(adapter);
    safe_release(factory);
    if (FAILED(hr)) {
      std::snprintf(why, sizeof(why), "no D3D11 device on the Vulkan device's adapter (0x%08lx)", hr);
      return vk_setup_e::transient;
    }

    // The copy's objects, into locals first
    VkCommandPool pool_handle = VK_NULL_HANDLE;
    VkCommandBuffer cmd[gc::kSlots] = {};
    VkFence fences[gc::kSlots] = {};
    VkCommandPoolCreateInfo pool {};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = p.queue_family;
    VkResult r = fn.CreateCommandPool(d.device, &pool, nullptr, &pool_handle);
    if (r == VK_SUCCESS) {
      VkCommandBufferAllocateInfo alloc {};
      alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      alloc.commandPool = pool_handle;
      alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      alloc.commandBufferCount = gc::kSlots;
      r = fn.AllocateCommandBuffers(d.device, &alloc, cmd);
    }
    for (int i = 0; i < gc::kSlots && r == VK_SUCCESS; ++i) {
      // Command buffers are dispatchable: the loader sets the device's
      // dispatch table in those the application allocates; ours come from
      // below it, so we set it (as the layer interface documents)
      *reinterpret_cast<void **>(cmd[i]) = *reinterpret_cast<void **>(d.device);
      VkFenceCreateInfo fence {};
      fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      r = fn.CreateFence(d.device, &fence, nullptr, &fences[i]);
    }
    if (r != VK_SUCCESS) {
      for (VkFence f : fences) {
        if (f) {
          fn.DestroyFence(d.device, f, nullptr);
        }
      }
      if (pool_handle) {
        fn.DestroyCommandPool(d.device, pool_handle, nullptr);
      }
      device->Release();
      std::snprintf(why, sizeof(why), "creating the copy's command objects failed (%d)", static_cast<int>(r));
      return vk_setup_e::transient;
    }

    // Commit: the capture is ours, on our device
    const auto identity = reinterpret_cast<IDXGISwapChain *>(p.swapchain->swapchain);
    release_capture(hwnd);
    g_cap.api = gc::api_e::vulkan;
    g_cap.swapchain = identity;
    g_captured_swapchain.store(identity, std::memory_order_release);
    g_cap.device = device;  // (the reference moves in)
    device->GetImmediateContext(&g_cap.context);
    g_cap.owner_thread = GetCurrentThreadId();
    g_cap.single_threaded = false;

    v = vk_state_t {};
    v.dev = d;
    v.fn = fn;
    v.memory_properties = memory_properties;
    v.swapchain = p.swapchain->swapchain;
    v.family = p.queue_family;
    v.vk_format = unorm;
    v.handle_type = nt ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT : VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
    v.kmt_importable = kmt;
    v.pool = pool_handle;
    for (int i = 0; i < gc::kSlots; ++i) {
      v.cmd[i] = cmd[i];
      v.fence[i] = fences[i];
    }
    // A device lost earlier left the completed value at "everything": what
    // this one submits starts from where the values are now
    AcquireSRWLockExclusive(&g_vk_queue_lock);
    g_vk_current_device.store(d.device, std::memory_order_relaxed);
    if (g_vk_completed.load(std::memory_order_acquire) == UINT64_MAX) {
      g_vk_completed.store(g_vk_next_value.load(std::memory_order_acquire), std::memory_order_release);
    }
    ReleaseSRWLockExclusive(&g_vk_queue_lock);
    log("Vulkan: capturing swapchain 0x%llx (%ux%u, format %d, colour space %d) on queue family %u; D3D11 textures imported as %s handles",
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(p.swapchain->swapchain)), p.swapchain->extent.width, p.swapchain->extent.height,
        static_cast<int>(p.swapchain->format), static_cast<int>(p.swapchain->color_space), p.queue_family, nt ? "NT" : "global");
    return vk_setup_e::ok;
  }

  // The slot's texture as a Vulkan image (imported once per texture set).
  // The slot's previous copy has completed (it was free to claim).
  bool vk_import(int slot) {
    auto &v = g_vk;
    if (v.image[slot] && v.imported_from[slot] == g_cap.textures[slot]) {
      return true;
    }
    if (v.image[slot]) {
      v.fn.DestroyImage(v.dev.device, v.image[slot], nullptr);
      v.fn.FreeMemory(v.dev.device, v.memory[slot], nullptr);
      v.image[slot] = VK_NULL_HANDLE;
      v.memory[slot] = VK_NULL_HANDLE;
    }
    safe_release(v.pinned[slot]);
    v.acquired_once[slot] = false;
    v.imported_from[slot] = nullptr;
    const HANDLE handle = v.handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT ? g_cap.shared_handles[slot] : reinterpret_cast<HANDLE>(g_cap.legacy_handles[slot]);
    if (!handle) {
      return false;
    }

    VkExternalMemoryImageCreateInfo ext {};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext.handleTypes = v.handle_type;
    VkImageCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.pNext = &ext;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = v.vk_format;
    info.extent = {g_cap.width, g_cap.height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    VkResult r = v.fn.CreateImage(v.dev.device, &info, nullptr, &image);
    if (r != VK_SUCCESS) {
      log("Vulkan: creating the image for slot %d failed (%d)", slot, static_cast<int>(r));
      return false;
    }
    VkImageMemoryRequirementsInfo2 req_info {};
    req_info.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
    req_info.image = image;
    VkMemoryDedicatedRequirements dedicated_req {};
    dedicated_req.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    VkMemoryRequirements2 req {};
    req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    req.pNext = &dedicated_req;
    v.fn.GetImageMemoryRequirements2(v.dev.device, &req_info, &req);

    // The memory type must suit both the image and this very handle
    VkMemoryWin32HandlePropertiesKHR handle_props {};
    handle_props.sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR;
    r = v.fn.GetMemoryWin32HandlePropertiesKHR(v.dev.device, v.handle_type, handle, &handle_props);
    const std::uint32_t types = r == VK_SUCCESS ? req.memoryRequirements.memoryTypeBits & handle_props.memoryTypeBits : 0;
    VkPhysicalDeviceMemoryProperties memory_props {};
    v.memory_properties(v.dev.physical_device, &memory_props);

    VkMemoryDedicatedAllocateInfo dedicated {};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.image = image;
    VkImportMemoryWin32HandleInfoKHR import {};
    import.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    import.pNext = &dedicated;
    import.handleType = v.handle_type;
    import.handle = handle;  // (not transferred: NT handles stay ours, closed at retirement)
    VkMemoryAllocateInfo alloc {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &import;
    alloc.allocationSize = req.memoryRequirements.size;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    // Device-local types first, then the others
    for (int pass = 0; pass < 2 && r != VK_SUCCESS; ++pass) {
      for (std::uint32_t t = 0; t < memory_props.memoryTypeCount && r != VK_SUCCESS; ++t) {
        const bool local = (memory_props.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        if ((types & (1u << t)) && local == (pass == 0)) {
          alloc.memoryTypeIndex = t;
          r = v.fn.AllocateMemory(v.dev.device, &alloc, nullptr, &memory);
        }
      }
    }
    if (r == VK_SUCCESS) {
      VkBindImageMemoryInfo bind {};
      bind.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
      bind.image = image;
      bind.memory = memory;
      r = v.fn.BindImageMemory2(v.dev.device, 1, &bind);
    }
    if (r != VK_SUCCESS) {
      if (memory) {
        v.fn.FreeMemory(v.dev.device, memory, nullptr);
      }
      v.fn.DestroyImage(v.dev.device, image, nullptr);
      log("Vulkan: importing the texture of slot %d as %s failed (%d; memory types 0x%x)", slot,
          v.handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT ? "NT" : "global", static_cast<int>(r), types);
      return false;
    }
    v.image[slot] = image;
    v.memory[slot] = memory;
    v.imported_from[slot] = g_cap.textures[slot];
    if (v.handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT) {
      g_cap.textures[slot]->AddRef();
      v.pinned[slot] = g_cap.textures[slot];
    }
    return true;
  }

  void vk_barrier(VkImageMemoryBarrier &b, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst, std::uint32_t src_family, std::uint32_t dst_family) {
    b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = src_family;
    b.dstQueueFamilyIndex = dst_family;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  }

  // The Present being handled on this thread: the limiter's state, and our
  // copy's semaphore (the Present waits on it: it outlives the call, being
  // destroyed only with the device)
  thread_local bool t_vk_paced = false;
  thread_local double t_vk_period = 0;
  thread_local VkSemaphore t_vk_wait = VK_NULL_HANDLE;
  thread_local PFN_vkQueueSubmit t_vk_submit = nullptr;  // set when our copy consumed the Present's waits

  bool vk_pre_present_impl(const vvk::present_t &p, std::uint32_t *wait_count, const VkSemaphore **waits) {
    if (p.swapchain_slot != 0) {
      return false;  // (a Present of several swapchains: the first is the one captured)
    }
    t_vk_submit = nullptr;
    if (!g_block || !p.swapchain || !p.swapchain->hwnd) {
      return false;
    }
    const auto now = qpc_now();
    const auto identity = reinterpret_cast<IDXGISwapChain *>(p.swapchain->swapchain);

    // The front-edge limiter, as for DXGI (the wait is in vk_post_present).
    // The present mode is the application's: a FIFO swapchain still waits
    // for the host display's vblank too.
    std::uint64_t release = 0;
    t_vk_paced = false;
    t_vk_period = limiter_period_qpc(now);
    if (t_vk_period > 0) {
      if (limiter_take()) {
        t_vk_paced = limiter_select(identity, now);
        if (t_vk_paced) {
          release = g_limiter.release_qpc;
        } else {
          limiter_give_back();
        }
      }
    } else if (limiter_take()) {
      limiter_forget();
      limiter_give_back();
    }

    g_block->frames_presented.fetch_add(1, std::memory_order_relaxed);
    if (!g_block->capture_enabled.load(std::memory_order_acquire)) {
      return false;
    }
    const HWND hwnd = p.swapchain->hwnd;
    const HWND foreground = GetForegroundWindow();
    if (hwnd != foreground && GetAncestor(hwnd, GA_ROOT) != foreground) {
      return false;
    }
    static std::atomic<bool> warned_swapchain {false}, warned_queue {false}, warned_waits {false};
    if (!p.swapchain->capturable) {
      if (!warned_swapchain.exchange(true)) {
        log("Vulkan: swapchain 0x%llx is not one the copy handles (no TRANSFER_SRC, protected, shared-present, layered or split); not captured",
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(p.swapchain->swapchain)));
      }
      return false;
    }
    if (p.queue_family == UINT32_MAX || !(p.queue_flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT))) {
      if (!warned_queue.exchange(true)) {
        log("Vulkan: the presenting queue's family is unknown or cannot copy; not captured");
      }
      return false;
    }
    constexpr std::uint32_t kMaxWaits = 16;
    if (p.info->waitSemaphoreCount > kMaxWaits) {
      if (!warned_waits.exchange(true)) {
        log("Vulkan: a Present waits on %u semaphores (more than %u); not captured", p.info->waitSemaphoreCount, kMaxWaits);
      }
      return false;
    }

    if (!TryAcquireSRWLockExclusive(&g_capture_lock)) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    struct unlock_t {
      ~unlock_t() {
        ReleaseSRWLockExclusive(&g_capture_lock);
      }
    } unlock;
    if (!reserve_retirement()) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    collect_retired(false);
    static std::uint32_t seen_recreate = 0;
    const auto recreate = g_block->recreate_request.load(std::memory_order_acquire);
    if (recreate != seen_recreate) {
      seen_recreate = recreate;
      log("The host asked for fresh capture textures");
      retire_generation(hwnd);
    }

    auto &v = g_vk;
    // Swapchains and devices that failed: not tried again (a permanent
    // device-level failure also tells the host, which then uses desktop capture)
    static VkDevice rejected_device = VK_NULL_HANDLE;
    static VkSwapchainKHR rejected_swapchain = VK_NULL_HANDLE;
    static std::uint64_t retry_after = 0;
    if (p.device->device == rejected_device || p.swapchain->swapchain == rejected_swapchain) {
      if (p.device->device == rejected_device || now < retry_after) {
        return false;
      }
      rejected_swapchain = VK_NULL_HANDLE;  // (a transient failure: tried again every few seconds)
    }
    if (g_cap.api != gc::api_e::vulkan || v.dev.device != p.device->device || v.swapchain != p.swapchain->swapchain || v.family != p.queue_family ||
        !g_cap.device) {
      vk_release(false);  // (another swapchain, device or queue family: its objects start over)
      char why[160] = {};
      const auto setup = vk_setup(p, hwnd, why);
      if (setup != vk_setup_e::ok) {
        char msg[192];
        std::snprintf(msg, sizeof(msg), "Vulkan: %s", why);
        if (setup == vk_setup_e::permanent) {
          // (not reported to the host as final: another device of the
          // process may be captured; until then it shows the desktop)
          rejected_device = p.device->device;
          log("%s; this device is not captured", msg);
        } else {
          rejected_swapchain = p.swapchain->swapchain;
          retry_after = now + 5 * qpc_frequency();
          log("%s; trying again in 5 s", msg);
        }
        return false;
      }
    }

    DXGI_FORMAT dxgi = DXGI_FORMAT_UNKNOWN;
    VkFormat unorm = VK_FORMAT_UNDEFINED;
    if (!vk_formats(p.swapchain->format, dxgi, unorm) || p.image_index >= p.swapchain->image_count) {
      return false;
    }
    D3D11_TEXTURE2D_DESC back {};
    back.Width = p.swapchain->extent.width;
    back.Height = p.swapchain->extent.height;
    back.Format = dxgi;
    back.SampleDesc.Count = 1;
    const bool nt_first = v.handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    if (!ensure_textures(back, hwnd, nt_first ? sharing_e::owner_nt : sharing_e::owner_kmt, !(nt_first && v.kmt_importable))) {
      if (v.handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT && v.kmt_importable) {
        log("Vulkan: NT-handle textures could not be created; switching to global handles");
        v.handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
      }
      return false;
    }
    if (v.present_waits.size() != p.swapchain->image_count) {
      vk_leftovers_t *left = vk_leftovers_for(v.dev.device, v.fn);
      if (!left) {
        return false;
      }
      left->semaphores.insert(left->semaphores.end(), v.present_waits.begin(), v.present_waits.end());  // (a Present may still wait on them)
      v.present_waits.assign(p.swapchain->image_count, VK_NULL_HANDLE);
      for (auto &s : v.present_waits) {
        VkSemaphoreCreateInfo info {};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (v.fn.CreateSemaphore(v.dev.device, &info, nullptr, &s) != VK_SUCCESS) {
          s = VK_NULL_HANDLE;
        }
      }
    }
    const VkSemaphore signal = v.present_waits[p.image_index];
    if (!signal) {
      return false;
    }

    const int slot = acquire_free_slot(&vk_slot_ready);
    if (slot < 0) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    auto give_back = [&] {
      g_block->owner[slot].store(gc::kOwnerNone, std::memory_order_release);
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
    };
    if (!vk_import(slot)) {
      give_back();
      if (v.handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT && v.kmt_importable) {
        // NT import advertised but failing: the next frame's textures are
        // made with global handles instead
        log("Vulkan: NT-handle import failed; switching to global handles");
        v.handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
        retire_generation(hwnd);
      } else {
        rejected_swapchain = p.swapchain->swapchain;
        retry_after = now + 5 * qpc_frequency();
        log("Vulkan: the capture textures cannot be imported; trying again in 5 s");
      }
      return false;
    }

    // Record: the presented image -> our slot's image, and both back
    const VkImage source = p.swapchain->images[p.image_index];
    const VkCommandBuffer cmd = v.cmd[slot];
    VkCommandBufferBeginInfo begin {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (v.fn.ResetCommandBuffer(cmd, 0) != VK_SUCCESS || v.fn.BeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
      give_back();
      return false;
    }
    VkImageMemoryBarrier b[2];
    // Our image is acquired from the external owner (D3D11): its first time
    // discarding (UNDEFINED), after that from the GENERAL it was released in
    vk_barrier(b[0], source, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
    vk_barrier(b[1], v.image[slot], v.acquired_once[slot] ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_QUEUE_FAMILY_EXTERNAL, v.family);
    v.fn.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, b);
    VkImageCopy region {};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {p.swapchain->extent.width, p.swapchain->extent.height, 1};
    v.fn.CmdCopyImage(cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, v.image[slot], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    vk_barrier(b[0], source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT, 0, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
    vk_barrier(b[1], v.image[slot], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0, v.family, VK_QUEUE_FAMILY_EXTERNAL);
    v.fn.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 2, b);
    if (v.fn.EndCommandBuffer(cmd) != VK_SUCCESS) {
      give_back();
      return false;
    }

    const auto generation = g_current_generation.load(std::memory_order_acquire);
    const auto frame_id = ++g_cap.next_frame_id;
    const auto version = write_slot_record(slot, generation, vk_color_space(p.swapchain->color_space, dxgi), frame_id, now, release);

    // After the game's rendering (the Present's own waits), before the
    // Present (which waits on our semaphore instead)
    VkPipelineStageFlags stages[kMaxWaits];
    for (std::uint32_t i = 0; i < p.info->waitSemaphoreCount; ++i) {
      stages[i] = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    VkSubmitInfo submit {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = p.info->waitSemaphoreCount;
    submit.pWaitSemaphores = p.info->pWaitSemaphores;
    submit.pWaitDstStageMask = stages;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &signal;
    v.fn.ResetFences(v.dev.device, 1, &v.fence[slot]);
    const VkResult r = v.fn.QueueSubmit(p.queue, 1, &submit, v.fence[slot]);
    if (r != VK_SUCCESS) {
      // (a failed submit consumes nothing: the Present keeps its own waits)
      static std::atomic<bool> warned_submit {false};
      if (!warned_submit.exchange(true)) {
        log("Vulkan: submitting the copy failed (%d)", static_cast<int>(r));
      }
      give_back();
      return false;
    }

    // ---- from here: nothing fails, nothing allocates ----
    v.acquired_once[slot] = true;
    g_block->owner[slot].store(gc::kOwnerNone, std::memory_order_release);  // (publishes once the fence is done)
    const auto value = g_vk_next_value.fetch_add(1, std::memory_order_acq_rel) + 1;
    g_vk_slot_busy[slot].store(true, std::memory_order_release);
    g_vk_in_flight.fetch_add(1, std::memory_order_acq_rel);
    mark_pending(slot, version, generation, frame_id, value);
    AcquireSRWLockExclusive(&g_vk_queue_lock);
    g_vk_ring[(g_vk_ring_head + g_vk_ring_count) % kVkRing] = {v.dev.device, v.fn.WaitForFences, v.fence[slot], value, slot};
    ++g_vk_ring_count;  // (at most one per slot in flight: never full)
    ReleaseSRWLockExclusive(&g_vk_queue_lock);
    SetEvent(g_vk_wake);

    note_capturing(identity, "Vulkan");
    t_vk_wait = signal;
    t_vk_submit = v.fn.QueueSubmit;
    *wait_count = 1;
    *waits = &t_vk_wait;
    return true;
  }

  bool vk_pre_present(void *, const vvk::present_t &p, std::uint32_t *wait_count, const VkSemaphore **waits) {
    try {
      return vk_pre_present_impl(p, wait_count, waits);
    } catch (...) {
      // (only before submission can anything throw: the Present keeps its own waits)
      return false;
    }
  }

  VkResult vk_post_present(void *, const vvk::present_t &p, VkResult result) {
    if (p.swapchain_slot != 0) {
      return result;
    }
    // A Present that failed for lack of memory was not queued: it waited on
    // nothing, so our semaphore is still signalled and the application's
    // (consumed by our copy) are not. Hand them back signalled, so a retry
    // of that Present does not wait forever.
    if (t_vk_submit && (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY)) {
      VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      VkSubmitInfo restore {};
      restore.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      restore.waitSemaphoreCount = 1;
      restore.pWaitSemaphores = &t_vk_wait;
      restore.pWaitDstStageMask = &stage;
      restore.signalSemaphoreCount = p.info->waitSemaphoreCount;
      restore.pSignalSemaphores = p.info->pWaitSemaphores;
      if (t_vk_submit(p.queue, 1, &restore, VK_NULL_HANDLE) != VK_SUCCESS) {
        // Its waits are gone for good: a retry would wait forever, so the
        // application is told what it cannot retry
        log("Vulkan: the Present failed (%d) and its semaphores could not be handed back: reporting the device lost", static_cast<int>(result));
        result = VK_ERROR_DEVICE_LOST;
      }
    }
    t_vk_submit = nullptr;
    if (!g_block || !t_vk_paced) {
      return result;
    }
    t_vk_paced = false;
    if (result >= 0) {
      limiter_wait(t_vk_period);
    }
    limiter_give_back();
    return result;
  }

  void vk_swapchain_destroyed(void *, const vvk::device_t &device, VkSwapchainKHR swapchain) {
    AcquireSRWLockExclusive(&g_capture_lock);
    if (g_vk.dev.device == device.device && g_vk.swapchain == swapchain) {
      vk_release(false);
      if (g_cap.api == gc::api_e::vulkan) {
        release_capture(setup_hwnd());
      }
      log("Vulkan: the captured swapchain was destroyed");
    }
    ReleaseSRWLockExclusive(&g_capture_lock);
  }

  void vk_device_destroyed(void *, const vvk::device_t &device) {
    AcquireSRWLockExclusive(&g_capture_lock);
    if (g_vk.dev.device == device.device) {
      vk_release(true);  // (waits for our copies, then everything of the device goes)
      if (g_cap.api == gc::api_e::vulkan) {
        release_capture(setup_hwnd());
      }
    } else {
      for (auto it = g_vk_leftovers.begin(); it != g_vk_leftovers.end(); ++it) {
        if (it->device == device.device) {
          vk_wait_idle(0);
          vk_destroy_leftovers(*it);
          g_vk_leftovers.erase(it);
          break;
        }
      }
    }
    ReleaseSRWLockExclusive(&g_capture_lock);
  }

  const vvk::callbacks_t g_vk_callbacks {&vk_pre_present, &vk_post_present, &vk_swapchain_destroyed, &vk_device_destroyed};

  // Registers with the Vulkan layer if it is in this process
  bool vk_attach() {
    HMODULE layer = GetModuleHandleW(vvk::kModuleName);
    if (!layer) {
      return false;
    }
    const auto entry = reinterpret_cast<vvk::entry_point_fn>(reinterpret_cast<void *>(GetProcAddress(layer, vvk::kEntryPoint)));
    const vvk::api_t *api = entry ? entry(vvk::kApiVersion) : nullptr;
    if (!api) {
      log("Vulkan: the layer in this process speaks another interface version; Vulkan swapchains are not captured");
      return false;
    }
    g_vk_wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HANDLE waiter = g_vk_wake ? CreateThread(nullptr, 0, vk_waiter_main, nullptr, 0, nullptr) : nullptr;
    if (!waiter) {
      log("Vulkan: could not start the fence waiter; Vulkan swapchains are not captured");
      return false;
    }
    CloseHandle(waiter);
    if (!api->register_callbacks(&g_vk_callbacks, nullptr)) {
      log("Vulkan: the layer already has a hook registered");
      return false;
    }
    g_vk_attached = true;
    log("Vulkan: attached to the Vulkan layer; Vulkan swapchains are captured from their Present");
    return true;
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

    g_vk_layer_present = GetModuleHandleW(vvk::kModuleName) != nullptr;
    const bool dxgi = install_hooks();
    const bool vk = vk_attach();
    if (dxgi) {
      // DLSS motion vectors (D3D11/D3D12): hooked once the driver's NGX is loaded
      if (HANDLE watcher = CreateThread(nullptr, 0, ngx_watch_main, nullptr, 0, nullptr)) {
        CloseHandle(watcher);
      }
    }
    if (!dxgi && !vk) {
      return 1;
    }
    set_state(gc::hook_state_e::hooked);
    log("Hooked in pid %lu: DXGI Present/Present1 %s, Vulkan %s", GetCurrentProcessId(), dxgi ? "detoured" : "not detoured", vk ? "attached" : "not present");
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
