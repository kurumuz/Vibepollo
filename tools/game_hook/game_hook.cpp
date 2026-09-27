/**
 * @file tools/game_hook/game_hook.cpp
 * @brief In-game capture hook (vibepollo_game_hook.dll), injected by Vibepollo
 *        into a focused fullscreen game. See
 *        src/platform/windows/game_capture/protocol.h for the protocol.
 *
 * Step 1 captures D3D11 swapchains. The DXGI entry points (Present, Present1,
 * ResizeBuffers, SetColorSpace1) are inline-detoured with MinHook at their
 * function addresses, found from a throwaway swapchain: that catches every
 * swapchain in the process whatever its swap effect or creation API, and
 * overlays that keep private vtables (their cached "original" is the function
 * we patched). D3D12 swapchains present through the same functions and are
 * reported as unsupported for now, so the host falls back to desktop capture.
 *
 * Threading, so that nothing here can stall or crash the game:
 *  - All D3D calls happen on the game's own thread inside its Present (or
 *    ResizeBuffers) call. The completion thread only waits on events, reads
 *    the clock and writes the shared block.
 *  - One swapchain is captured at a time; a Present on another thread that
 *    finds the capture busy skips (try-lock), never waits.
 *  - A free slot's keyed mutex is taken with a zero timeout; if the host still
 *    holds it, the frame is skipped.
 *  - The pending-frame bookkeeping is a fixed per-slot array, no allocation,
 *    no lock shared with the completion thread.
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
  using resize_buffers_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT, UINT, DXGI_FORMAT, UINT);
  using set_color_space1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain3 *, DXGI_COLOR_SPACE_TYPE);

  // IDXGISwapChain vtable slots (IUnknown 0-2, IDXGIObject 3-6, IDXGIDeviceSubObject 7)
  constexpr int kVtPresent = 8;
  constexpr int kVtResizeBuffers = 13;
  constexpr int kVtPresent1 = 22;  // IDXGISwapChain1
  constexpr int kVtSetColorSpace1 = 38;  // IDXGISwapChain3

  present_fn g_real_present = nullptr;
  present1_fn g_real_present1 = nullptr;
  resize_buffers_fn g_real_resize_buffers = nullptr;
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

  template<class T>
  void safe_release(T *&p) {
    if (p) {
      p->Release();
      p = nullptr;
    }
  }

  // ---- slots ---------------------------------------------------------------

  enum slot_state_e : int {
    slot_free = 0,
    slot_pending = 1,
    slot_published = 2,
  };

  // Per-slot frame metadata: written by the render thread before the slot's
  // completion event is set, read by the completion thread after it fires.
  struct slot_t {
    std::atomic<int> state {slot_free};
    std::atomic<std::uint32_t> version {0};
    std::uint32_t generation = 0;
    std::uint32_t color_space = 0;
    std::uint64_t frame_id = 0;
    std::uint64_t present_qpc = 0;
    HANDLE done_event = nullptr;  // for the life of the DLL
  };

  slot_t g_slots[gc::kSlots];
  HANDLE g_frame_event = nullptr;
  HANDLE g_completion_thread = nullptr;
  std::atomic<std::uint32_t> g_current_generation {0};

  void publish(int slot, std::uint64_t gpu_done_qpc) {
    auto &s = g_slots[slot];
    auto &b = *g_block;
    const auto seq = b.frame_seq.load(std::memory_order_relaxed);
    b.frame_seq.store(seq + 1, std::memory_order_release);  // odd: writing
    std::atomic_thread_fence(std::memory_order_release);
    b.slot.store(static_cast<std::uint32_t>(slot), std::memory_order_relaxed);
    b.slot_version.store(s.version.load(std::memory_order_relaxed), std::memory_order_relaxed);
    b.generation.store(s.generation, std::memory_order_relaxed);
    b.color_space.store(s.color_space, std::memory_order_relaxed);
    b.frame_id.store(s.frame_id, std::memory_order_relaxed);
    b.present_qpc.store(s.present_qpc, std::memory_order_relaxed);
    b.gpu_done_qpc.store(gpu_done_qpc, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    b.frame_seq.store(seq + 2, std::memory_order_release);  // even: stable
    b.publish_qpc.store(qpc_now(), std::memory_order_release);
    b.frames_published.fetch_add(1, std::memory_order_relaxed);
    SetEvent(g_frame_event);
  }

  // Moves a completed slot to published (and the previously published slot
  // to free). Only this function writes slot_published/slot_free for a
  // pending slot, so the render thread's free-slot search is consistent.
  void complete_slot(int slot, std::uint64_t gpu_done_qpc) {
    auto &s = g_slots[slot];
    int expected = slot_pending;
    if (!s.state.compare_exchange_strong(expected, slot_published, std::memory_order_acq_rel)) {
      return;  // reset (generation change) took it back
    }
    for (int i = 0; i < gc::kSlots; ++i) {
      if (i != slot) {
        int published = slot_published;
        g_slots[i].state.compare_exchange_strong(published, slot_free, std::memory_order_acq_rel);
      }
    }
    if (s.generation != g_current_generation.load(std::memory_order_acquire)) {
      return;  // textures of a retired generation: not offered to the host
    }
    publish(slot, gpu_done_qpc);
  }

  // Waits for the GPU to finish each slot's copy. Touches no D3D object.
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
      complete_slot(static_cast<int>(r - WAIT_OBJECT_0), qpc_now());
    }
  }

  // ---- the captured swapchain ---------------------------------------------

  struct capture_t {
    IDXGISwapChain *swapchain = nullptr;  // identity only, never dereferenced without a Present in progress
    ID3D11Device *device = nullptr;
    ID3D11DeviceContext *context = nullptr;
    ID3D11DeviceContext4 *context4 = nullptr;  // null without fence support
    ID3D11Fence *fence = nullptr;
    std::uint64_t fence_value = 0;
    std::uint64_t next_frame_id = 0;
    std::uint64_t last_present_qpc = 0;
    std::uint32_t color_space = static_cast<std::uint32_t>(gc::color_space_e::unknown);
    bool color_space_set = false;  // SetColorSpace1 seen on this swapchain
    bool reported_d3d12 = false;

    // Current generation's textures, plus the previous generation's shared
    // handles kept open so the host never meets a recycled handle value
    ID3D11Texture2D *textures[gc::kSlots] = {};
    IDXGIKeyedMutex *mutexes[gc::kSlots] = {};
    HANDLE shared_handles[gc::kSlots] = {};
    HANDLE previous_handles[gc::kSlots] = {};
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
  };

  capture_t g_cap;
  SRWLOCK g_capture_lock = SRWLOCK_INIT;  // held by the one thread capturing; others skip

  void reset_slots() {
    g_current_generation.fetch_add(1, std::memory_order_acq_rel);
    for (auto &s : g_slots) {
      s.state.store(slot_free, std::memory_order_release);
    }
  }

  void close_handles(HANDLE *handles) {
    for (int i = 0; i < gc::kSlots; ++i) {
      if (handles[i]) {
        CloseHandle(handles[i]);
        handles[i] = nullptr;
      }
    }
  }

  void release_textures() {
    reset_slots();
    for (int i = 0; i < gc::kSlots; ++i) {
      safe_release(g_cap.mutexes[i]);
      safe_release(g_cap.textures[i]);
    }
    close_handles(g_cap.previous_handles);
    std::memcpy(g_cap.previous_handles, g_cap.shared_handles, sizeof(g_cap.shared_handles));
    std::memset(g_cap.shared_handles, 0, sizeof(g_cap.shared_handles));
    g_cap.width = g_cap.height = 0;
    g_cap.format = DXGI_FORMAT_UNKNOWN;
  }

  void release_capture() {
    release_textures();
    safe_release(g_cap.fence);
    safe_release(g_cap.context4);
    safe_release(g_cap.context);
    safe_release(g_cap.device);
    g_cap.swapchain = nullptr;
    g_cap.color_space_set = false;
    g_cap.color_space = static_cast<std::uint32_t>(gc::color_space_e::unknown);
  }

  bool ensure_device(IDXGISwapChain *swapchain, ID3D11Device *device) {
    if (g_cap.device == device && g_cap.swapchain == swapchain) {
      return true;
    }
    release_capture();
    g_cap.swapchain = swapchain;
    device->AddRef();
    g_cap.device = device;
    device->GetImmediateContext(&g_cap.context);

    // A fence tells us when the GPU finished each copy; the render thread
    // signals it and registers the slot's completion event, the completion
    // thread only waits on that event. A device created single-threaded is
    // still left alone (no fence, frames publish at Present time), as are
    // runtimes without fences.
    const bool single_threaded = (device->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED) != 0;
    if (!single_threaded) {
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
    log("Device %p (%s): frames timestamped at %s", static_cast<void *>(device), single_threaded ? "single-threaded" : "multithreaded",
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

  void publish_setup(HWND hwnd) {
    auto &b = *g_block;
    const auto seq = b.setup_seq.load(std::memory_order_relaxed);
    b.setup_seq.store(seq + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    b.setup.api.store(static_cast<std::uint32_t>(gc::api_e::d3d11), std::memory_order_relaxed);
    b.setup.width.store(g_cap.width, std::memory_order_relaxed);
    b.setup.height.store(g_cap.height, std::memory_order_relaxed);
    b.setup.format.store(static_cast<std::uint32_t>(g_cap.format), std::memory_order_relaxed);
    b.setup.hwnd.store(reinterpret_cast<std::uint64_t>(hwnd), std::memory_order_relaxed);
    for (int i = 0; i < gc::kSlots; ++i) {
      b.setup.textures[i].store(reinterpret_cast<std::uint64_t>(g_cap.shared_handles[i]), std::memory_order_relaxed);
    }
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
    b.setup.generation.store(g_current_generation.load(std::memory_order_acquire), std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    b.setup_seq.store(seq + 2, std::memory_order_release);
  }

  bool ensure_textures(const D3D11_TEXTURE2D_DESC &back, HWND hwnd) {
    const DXGI_FORMAT format = shareable_format(back.Format);
    if (g_cap.textures[0] && g_cap.width == back.Width && g_cap.height == back.Height && g_cap.format == format) {
      return true;
    }
    release_textures();

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
        release_textures();
        return false;
      }
    }

    g_cap.width = back.Width;
    g_cap.height = back.Height;
    g_cap.format = format;
    if (!g_cap.color_space_set) {
      g_cap.color_space = default_color_space(format);
    }
    publish_setup(hwnd);
    log("Capture textures %ux%u format %d, generation %u, color space %u", back.Width, back.Height, format,
        g_current_generation.load(), g_cap.color_space);
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
      if (g_slots[i].state.load(std::memory_order_acquire) == slot_free) {
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

    ID3D11Device *device = nullptr;
    if (FAILED(swapchain->GetDevice(__uuidof(ID3D11Device), reinterpret_cast<void **>(&device)))) {
      if (!g_cap.reported_d3d12) {
        g_cap.reported_d3d12 = true;
        set_state(gc::hook_state_e::unsupported, "The swapchain is not D3D11 (likely D3D12); not captured yet");
      }
      return;
    }
    ensure_device(swapchain, device);
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

    const int slot = find_free_slot();
    if (slot < 0 || g_cap.mutexes[slot]->AcquireSync(0, 0) != S_OK) {
      g_block->frames_skipped.fetch_add(1, std::memory_order_relaxed);
      back->Release();
      return;
    }
    if (back_desc.SampleDesc.Count > 1) {
      g_cap.context->ResolveSubresource(g_cap.textures[slot], 0, back, 0, g_cap.format);
    } else {
      g_cap.context->CopyResource(g_cap.textures[slot], back);
    }
    g_cap.mutexes[slot]->ReleaseSync(0);
    back->Release();

    auto &s = g_slots[slot];
    s.version.fetch_add(1, std::memory_order_acq_rel);
    s.generation = g_current_generation.load(std::memory_order_acquire);
    s.color_space = g_cap.color_space;
    s.frame_id = ++g_cap.next_frame_id;
    s.present_qpc = present_qpc;
    s.state.store(slot_pending, std::memory_order_release);

    if (g_cap.fence && g_cap.context4) {
      g_cap.context4->Signal(g_cap.fence, ++g_cap.fence_value);
      if (FAILED(g_cap.fence->SetEventOnCompletion(g_cap.fence_value, s.done_event))) {
        complete_slot(slot, 0);
      }
    } else {
      complete_slot(slot, 0);
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

  // We hold no back-buffer references outside Present, so ResizeBuffers is
  // free to proceed; our textures are dropped here so the next Present
  // recreates them at the new size (and the memory is not held meanwhile).
  HRESULT STDMETHODCALLTYPE hook_resize_buffers(IDXGISwapChain *swapchain, UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags) {
    if (swapchain == g_cap.swapchain && TryAcquireSRWLockExclusive(&g_capture_lock)) {
      release_textures();
      ReleaseSRWLockExclusive(&g_capture_lock);
    }
    return g_real_resize_buffers(swapchain, count, width, height, format, flags);
  }

  HRESULT STDMETHODCALLTYPE hook_set_color_space1(IDXGISwapChain3 *swapchain, DXGI_COLOR_SPACE_TYPE color_space) {
    const HRESULT hr = g_real_set_color_space1(swapchain, color_space);
    if (SUCCEEDED(hr) && TryAcquireSRWLockExclusive(&g_capture_lock)) {
      IDXGISwapChain *base = nullptr;
      if (SUCCEEDED(swapchain->QueryInterface(__uuidof(IDXGISwapChain), reinterpret_cast<void **>(&base)))) {
        if (base == g_cap.swapchain) {
          auto cs = gc::color_space_e::unknown;
          if (color_space == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) {
            cs = gc::color_space_e::srgb;
          } else if (color_space == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709) {
            cs = gc::color_space_e::scrgb;
          } else if (color_space == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
            cs = gc::color_space_e::hdr10;
          }
          g_cap.color_space = static_cast<std::uint32_t>(cs);
          g_cap.color_space_set = true;
          log("Swapchain %p color space %d", static_cast<void *>(base), static_cast<int>(color_space));
        }
        base->Release();
      }
      ReleaseSRWLockExclusive(&g_capture_lock);
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
           MH_CreateHook(vtable[kVtPresent1], reinterpret_cast<void *>(&hook_present1), reinterpret_cast<void **>(&g_real_present1)) == MH_OK &&
           MH_CreateHook(vtable[kVtResizeBuffers], reinterpret_cast<void *>(&hook_resize_buffers), reinterpret_cast<void **>(&g_real_resize_buffers)) == MH_OK;
      IDXGISwapChain3 *swapchain3 = nullptr;
      if (ok && SUCCEEDED(swapchain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&swapchain3)))) {
        void **vtable3 = *reinterpret_cast<void ***>(swapchain3);
        MH_CreateHook(vtable3[kVtSetColorSpace1], reinterpret_cast<void *>(&hook_set_color_space1), reinterpret_cast<void **>(&g_real_set_color_space1));
        swapchain3->Release();
      }
      if (ok) {
        ok = MH_EnableHook(MH_ALL_HOOKS) == MH_OK;
      }
      if (!ok) {
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
    log("Detoured DXGI Present/Present1/ResizeBuffers in pid %lu", GetCurrentProcessId());
    return 0;
  }

}  // namespace

// The DLL stays loaded for the life of the process: the detours point into
// it, and there is no safe moment to take them back mid-frame.
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
