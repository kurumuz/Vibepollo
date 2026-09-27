/**
 * @file tools/game_hook/game_hook.cpp
 * @brief In-game capture hook (vibepollo_game_hook.dll), injected by Vibepollo
 *        into a focused fullscreen game. See
 *        src/platform/windows/game_capture/protocol.h for the protocol.
 *
 * Step 1 captures D3D11 swapchains. The DXGI entry points (Present, Present1,
 * SetColorSpace1) are inline-detoured with MinHook at their function
 * addresses, found from a throwaway swapchain: that catches every swapchain
 * in the process whatever its swap effect or creation API, and overlays that
 * keep private vtables (their cached "original" is the function we patched).
 * D3D12 swapchains present through the same functions and are reported as
 * unsupported for now, so the host falls back to desktop capture.
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
 *    occupant. The completion thread is the only writer of `latest`.
 *  - Nothing is drained or waited for on the render thread: retired textures
 *    are released on a later Present, once their copies have completed.
 */
#include "src/platform/windows/game_capture/protocol.h"

#include <MinHook.h>

#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_4.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace {

  namespace gc = game_capture;

  using present_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT);
  using present1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain1 *, UINT, UINT, const DXGI_PRESENT_PARAMETERS *);
  using set_color_space1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain3 *, DXGI_COLOR_SPACE_TYPE);

  // IDXGISwapChain vtable slots (IUnknown 0-2, IDXGIObject 3-6, IDXGIDeviceSubObject 7)
  constexpr int kVtPresent = 8;
  constexpr int kVtPresent1 = 22;  // IDXGISwapChain1
  constexpr int kVtSetColorSpace1 = 38;  // IDXGISwapChain3

  // Private data key under which a swapchain remembers its colour space
  // {8f4c2a6e-5b1d-4c7a-9e3f-2d6a8b1c4e70}
  constexpr GUID kColorSpaceKey = {0x8f4c2a6e, 0x5b1d, 0x4c7a, {0x9e, 0x3f, 0x2d, 0x6a, 0x8b, 0x1c, 0x4e, 0x70}};

  present_fn g_real_present = nullptr;
  present1_fn g_real_present1 = nullptr;
  set_color_space1_fn g_real_set_color_space1 = nullptr;

  gc::shared_block_t *g_block = nullptr;
  thread_local bool t_in_present = false;

  // ---- logging -----------------------------------------------------------

  FILE *g_log = nullptr;
  SRWLOCK g_log_lock = SRWLOCK_INIT;

  void log(const char *fmt, ...) {
    AcquireSRWLockExclusive(&g_log_lock);
    if (!g_log) {
      wchar_t dir[MAX_PATH];
      const DWORD n = GetTempPathW(MAX_PATH, dir);
      if (n != 0 && n < MAX_PATH) {
        wchar_t path[MAX_PATH];
        std::swprintf(path, MAX_PATH, L"%svibepollo_game_hook_%lu.log", dir, GetCurrentProcessId());
        g_log = _wfopen(path, L"a");
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

  // The fence the completion thread checks completions against: one
  // reference held for it by the render thread at publication, released by
  // the completion thread when it sees a newer pointer.
  std::atomic<ID3D11Fence *> g_worker_fence {nullptr};

  // Completion-thread state
  int g_published_slot = -1;
  std::uint64_t g_published_word = 0;
  std::uint64_t g_last_published_frame_id = 0;
  ID3D11Fence *g_worker_fence_held = nullptr;

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
    std::atomic_thread_fence(std::memory_order_acquire);
    if (s.word.load(std::memory_order_relaxed) != word) {
      return;  // reused meanwhile: those fields were not this submission's
    }

    // Is this event the copy's real completion? Only a fence can say; an
    // event set early for this slot (a previous occupant's registration
    // firing late) is ignored and the real one publishes.
    if (fence_value) {
      ID3D11Fence *fence = g_worker_fence.load(std::memory_order_acquire);
      if (fence != g_worker_fence_held) {
        safe_release(g_worker_fence_held);
        g_worker_fence_held = fence;  // the reference the render thread left for us
      }
      if (!fence || fence->GetCompletedValue() < fence_value) {
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
      complete_slot(static_cast<int>(r - WAIT_OBJECT_0));
    }
  }

  // ---- the captured swapchain ---------------------------------------------

  // A retired texture set, kept until its copies have completed (or a second
  // has passed), released on a later Present. Its shared handles stay open
  // one retirement longer than the textures so a host still opening them
  // never meets a recycled handle value.
  struct retired_t {
    ID3D11Texture2D *textures[gc::kSlots] = {};
    IDXGIKeyedMutex *mutexes[gc::kSlots] = {};
    HANDLE handles[gc::kSlots] = {};
    ID3D11Fence *fence = nullptr;  // the fence its copies were signalled on (a reference)
    std::uint64_t highest_fence_value = 0;
    std::uint64_t retired_qpc = 0;
    bool textures_released = false;
    bool in_use = false;
  };

  constexpr int kRetiredSets = 4;

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
    bool reported_d3d12 = false;

    ID3D11Texture2D *textures[gc::kSlots] = {};
    IDXGIKeyedMutex *mutexes[gc::kSlots] = {};
    HANDLE shared_handles[gc::kSlots] = {};
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::uint64_t highest_fence_value_used = 0;  // of this texture set

    retired_t retired[kRetiredSets];
  };

  capture_t g_cap;
  SRWLOCK g_capture_lock = SRWLOCK_INIT;  // held by the one thread capturing; others skip

  void publish_setup(HWND hwnd, bool valid) {
    auto &b = *g_block;
    const auto seq = b.setup_seq.load(std::memory_order_relaxed);
    b.setup_seq.store(seq + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    b.setup.api.store(static_cast<std::uint32_t>(gc::api_e::d3d11), std::memory_order_relaxed);
    b.setup.width.store(valid ? g_cap.width : 0, std::memory_order_relaxed);
    b.setup.height.store(valid ? g_cap.height : 0, std::memory_order_relaxed);
    b.setup.format.store(static_cast<std::uint32_t>(valid ? g_cap.format : DXGI_FORMAT_UNKNOWN), std::memory_order_relaxed);
    b.setup.hwnd.store(reinterpret_cast<std::uint64_t>(hwnd), std::memory_order_relaxed);
    for (int i = 0; i < gc::kSlots; ++i) {
      b.setup.textures[i].store(valid ? reinterpret_cast<std::uint64_t>(g_cap.shared_handles[i]) : 0, std::memory_order_relaxed);
    }
    if (valid && g_cap.device) {
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
    }
    safe_release(r.fence);
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
        const bool done = !r.fence || r.fence->GetCompletedValue() >= r.highest_fence_value;
        const bool old = now - r.retired_qpc > qpc_frequency();
        if (done || old || force) {
          release_retired(r);
        }
      }
      if (r.textures_released && (now - r.retired_qpc > 2 * qpc_frequency() || force)) {
        close_handles(r.handles);
        r.in_use = false;
      }
    }
  }

  // Retires the current texture set: one transition, however it was reached.
  // The set stays alive in `retired` until its copies are done; slots are
  // freed here on exactly the submissions they hold (a completion in flight
  // for one of them fails its CAS and does nothing).
  void retire_generation(HWND hwnd) {
    if (!g_cap.textures[0]) {
      return;
    }
    g_current_generation.fetch_add(1, std::memory_order_acq_rel);
    publish_setup(hwnd, false);

    retired_t *slot = nullptr;
    for (auto &r : g_cap.retired) {
      if (!r.in_use) {
        slot = &r;
        break;
      }
    }
    if (!slot) {
      // Retirement storm: release the oldest set now (its copies had a
      // second and more to complete)
      slot = &g_cap.retired[0];
      for (auto &r : g_cap.retired) {
        if (r.retired_qpc < slot->retired_qpc) {
          slot = &r;
        }
      }
      if (!slot->textures_released) {
        release_retired(*slot);
      }
      close_handles(slot->handles);
    }
    *slot = retired_t {};
    slot->in_use = true;
    slot->retired_qpc = qpc_now();
    slot->highest_fence_value = g_cap.highest_fence_value_used;
    if (g_cap.fence) {
      g_cap.fence->AddRef();
      slot->fence = g_cap.fence;
    }
    std::memcpy(slot->textures, g_cap.textures, sizeof(g_cap.textures));
    std::memcpy(slot->mutexes, g_cap.mutexes, sizeof(g_cap.mutexes));
    std::memcpy(slot->handles, g_cap.shared_handles, sizeof(g_cap.shared_handles));
    std::memset(g_cap.textures, 0, sizeof(g_cap.textures));
    std::memset(g_cap.mutexes, 0, sizeof(g_cap.mutexes));
    std::memset(g_cap.shared_handles, 0, sizeof(g_cap.shared_handles));
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

  void release_capture(HWND hwnd) {
    retire_generation(hwnd);
    collect_retired(false);
    safe_release(g_cap.fence);
    safe_release(g_cap.context4);
    safe_release(g_cap.context);
    safe_release(g_cap.device);
    g_cap.swapchain = nullptr;
    g_cap.owner_thread = 0;
    g_cap.single_threaded = false;
  }

  bool ensure_device(IDXGISwapChain *swapchain, ID3D11Device *device, HWND hwnd) {
    if (g_cap.device == device && g_cap.swapchain == swapchain) {
      return true;
    }
    release_capture(hwnd);
    g_cap.swapchain = swapchain;
    device->AddRef();
    g_cap.device = device;
    device->GetImmediateContext(&g_cap.context);
    g_cap.owner_thread = GetCurrentThreadId();
    g_cap.single_threaded = (device->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED) != 0;

    // A fence tells us when the GPU finished each copy. The completion
    // thread checks its completed value through a reference of its own, so
    // a device created single-threaded gets no fence (frames publish at
    // Present time), as do runtimes without fences.
    ID3D11Fence *worker_fence = nullptr;
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
      if (g_cap.fence) {
        g_cap.fence->AddRef();  // the completion thread's reference
        worker_fence = g_cap.fence;
      }
    }
    g_cap.fence_value = 0;
    // Hand the completion thread its fence (it releases the previous one).
    // An unclaimed handoff (no completion ever ran) leaks one reference,
    // bounded by device changes, which are rare.
    g_worker_fence.store(worker_fence, std::memory_order_release);
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
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    for (int i = 0; i < gc::kSlots; ++i) {
      HRESULT hr = g_cap.device->CreateTexture2D(&desc, nullptr, &g_cap.textures[i]);
      if (SUCCEEDED(hr)) {
        hr = g_cap.textures[i]->QueryInterface(__uuidof(IDXGIKeyedMutex), reinterpret_cast<void **>(&g_cap.mutexes[i]));
      }
      IDXGIResource1 *resource = nullptr;
      if (SUCCEEDED(hr)) {
        hr = g_cap.textures[i]->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void **>(&resource));
      }
      if (SUCCEEDED(hr)) {
        hr = resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &g_cap.shared_handles[i]);
        resource->Release();
      }
      if (FAILED(hr)) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "Creating capture texture %d (%ux%u fmt %d) failed: 0x%08lx", i, desc.Width, desc.Height, desc.Format, hr);
        set_state(gc::hook_state_e::failed, msg);
        for (int j = 0; j < gc::kSlots; ++j) {
          safe_release(g_cap.mutexes[j]);
          safe_release(g_cap.textures[j]);
        }
        close_handles(g_cap.shared_handles);
        return false;
      }
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

  int find_free_slot() {
    for (int i = 0; i < gc::kSlots; ++i) {
      if (word_state(g_slots[i].word.load(std::memory_order_acquire)) == st_free) {
        return i;
      }
    }
    return -1;
  }

  void capture_frame(IDXGISwapChain *swapchain, std::uint64_t present_qpc) {
    g_block->frames_presented.fetch_add(1, std::memory_order_relaxed);
    if (!g_block->capture_enabled.load(std::memory_order_acquire)) {
      return;
    }
    HWND hwnd = nullptr;
    if (!foreground_window(swapchain, hwnd)) {
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

    ID3D11Device *device = nullptr;
    if (FAILED(swapchain->GetDevice(__uuidof(ID3D11Device), reinterpret_cast<void **>(&device)))) {
      if (!g_cap.reported_d3d12) {
        g_cap.reported_d3d12 = true;
        set_state(gc::hook_state_e::unsupported, "The swapchain is not D3D11 (likely D3D12); not captured yet");
      }
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

    const int slot = find_free_slot();
    if (slot < 0 || g_cap.mutexes[slot]->AcquireSync(0, 0) != S_OK) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      back->Release();
      return;
    }

    // Holding the slot's pixel mutex: rewrite its record, then the pixels.
    // The host's acquire is GPU-ordered after the release below, so a host
    // holding the mutex and reading an even record reads this frame's.
    const auto generation = g_current_generation.load(std::memory_order_acquire);
    const auto frame_id = ++g_cap.next_frame_id;
    auto &record = g_block->slots[slot];
    const auto seq = record.seq.load(std::memory_order_relaxed);
    record.seq.store(seq + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    record.generation.store(generation, std::memory_order_relaxed);
    record.color_space.store(color_space, std::memory_order_relaxed);
    record.frame_id.store(frame_id, std::memory_order_relaxed);
    record.present_qpc.store(present_qpc, std::memory_order_relaxed);
    record.gpu_done_qpc.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    record.seq.store(seq + 2, std::memory_order_release);

    if (back_desc.SampleDesc.Count > 1) {
      g_cap.context->ResolveSubresource(g_cap.textures[slot], 0, back, 0, g_cap.format);
    } else {
      g_cap.context->CopyResource(g_cap.textures[slot], back);
    }
    g_cap.mutexes[slot]->ReleaseSync(0);
    back->Release();

    // The submission: fields first, then the pending word (release)
    auto &s = g_slots[slot];
    s.version = (seq + 2) >> 1;
    s.generation = generation;
    s.frame_id = frame_id;
    s.fence_value = (g_cap.fence && g_cap.context4) ? ++g_cap.fence_value : 0;
    const auto ticket = g_next_ticket++;
    s.word.store(make_word(st_pending, ticket), std::memory_order_release);

    // Completion always goes through the completion thread (single
    // publisher). Without a fence, or if registering its event fails, the
    // event is set here and the frame publishes at Present time.
    bool signalled = false;
    if (s.fence_value) {
      g_cap.highest_fence_value_used = s.fence_value;
      g_cap.context4->Signal(g_cap.fence, s.fence_value);
      signalled = SUCCEEDED(g_cap.fence->SetEventOnCompletion(s.fence_value, s.done_event));
    }
    if (!signalled) {
      if (s.fence_value) {
        // The completion thread would wait for the fence; without a
        // registration nothing wakes it, so this frame is dropped instead
        auto pending = make_word(st_pending, ticket);
        s.word.compare_exchange_strong(pending, make_word(st_free, ticket), std::memory_order_acq_rel);
        g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      } else {
        SetEvent(s.done_event);
      }
    }

    if (g_block->hook_state.load(std::memory_order_relaxed) != static_cast<std::uint32_t>(gc::hook_state_e::capturing)) {
      set_state(gc::hook_state_e::capturing);
      log("Capturing D3D11 frames from swapchain %p", static_cast<void *>(swapchain));
    }
  }

  // ---- detours -------------------------------------------------------------

  // DXGI may implement one Present entry point through the other; only the
  // outermost call on a thread captures.
  HRESULT STDMETHODCALLTYPE hook_present(IDXGISwapChain *swapchain, UINT sync_interval, UINT flags) {
    if (t_in_present) {
      return g_real_present(swapchain, sync_interval, flags);
    }
    t_in_present = true;
    if (!(flags & DXGI_PRESENT_TEST)) {
      capture_frame(swapchain, qpc_now());
    }
    const HRESULT hr = g_real_present(swapchain, sync_interval, flags);
    t_in_present = false;
    return hr;
  }

  HRESULT STDMETHODCALLTYPE hook_present1(IDXGISwapChain1 *swapchain, UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS *params) {
    if (t_in_present) {
      return g_real_present1(swapchain, sync_interval, flags, params);
    }
    t_in_present = true;
    if (!(flags & DXGI_PRESENT_TEST)) {
      capture_frame(swapchain, qpc_now());
    }
    const HRESULT hr = g_real_present1(swapchain, sync_interval, flags, params);
    t_in_present = false;
    return hr;
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
      swapchain->SetPrivateData(kColorSpaceKey, sizeof(value), &value);
      log("Swapchain %p color space %d", static_cast<void *>(swapchain), static_cast<int>(color_space));
    }
    return hr;
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
        swapchain3->Release();
      }
      if (ok) {
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
