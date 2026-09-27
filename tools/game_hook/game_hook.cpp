/**
 * @file tools/game_hook/game_hook.cpp
 * @brief In-game capture hook (vibepollo_game_hook.dll), injected by Vibepollo
 *        into a focused fullscreen game. See
 *        src/platform/windows/game_capture/protocol.h for the protocol.
 *
 * Step 1 captures D3D11 swapchains. It hooks IDXGISwapChain::Present and
 * IDXGISwapChain1::Present1 by patching the DXGI swapchain vtable (found from
 * a throwaway swapchain), plus IDXGISwapChain3::SetColorSpace1 to learn how
 * the pixels are encoded. D3D12 swapchains present through the same vtable and
 * are detected and reported as unsupported for now, so the host falls back to
 * desktop capture.
 *
 * Everything the Present hook does runs on the game's own render thread,
 * inside its Present call: that is the only place the game's (possibly
 * single-threaded) immediate context may be used. The hook never waits on the
 * host: a slot whose keyed mutex is not immediately available is skipped.
 */
#include "src/platform/windows/game_capture/protocol.h"

#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_4.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <mutex>

namespace {

  using present_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT);
  using present1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain1 *, UINT, UINT, const DXGI_PRESENT_PARAMETERS *);
  using set_color_space1_fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain3 *, DXGI_COLOR_SPACE_TYPE);

  // IDXGISwapChain vtable slots (IUnknown 0-2, IDXGIObject 3-6, IDXGIDeviceSubObject 7)
  constexpr int kVtPresent = 8;
  constexpr int kVtPresent1 = 22;  // IDXGISwapChain1
  constexpr int kVtSetColorSpace1 = 38;  // IDXGISwapChain3

  present_fn g_real_present = nullptr;
  present1_fn g_real_present1 = nullptr;
  set_color_space1_fn g_real_set_color_space1 = nullptr;

  game_capture::shared_block_t *g_block = nullptr;
  std::atomic<std::uint32_t> g_color_space {static_cast<std::uint32_t>(game_capture::color_space_e::srgb)};
  thread_local bool t_in_present = false;

  FILE *g_log = nullptr;
  std::mutex g_log_lock;

  void log(const char *fmt, ...) {
    std::lock_guard lg {g_log_lock};
    if (!g_log) {
      wchar_t dir[MAX_PATH];
      const DWORD n = GetTempPathW(MAX_PATH, dir);
      if (n == 0 || n >= MAX_PATH) {
        return;
      }
      wchar_t path[MAX_PATH];
      std::swprintf(path, MAX_PATH, L"%svibepollo_game_hook_%lu.log", dir, GetCurrentProcessId());
      g_log = _wfopen(path, L"a");
      if (!g_log) {
        return;
      }
    }
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

  void set_state(game_capture::hook_state_e state, const char *error = nullptr) {
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

  // Resources for the swapchain being captured. Touched only on the game's
  // present thread, except the fence and the pending list (fence thread).
  struct capture_t {
    ID3D11Device *device = nullptr;
    ID3D11DeviceContext *context = nullptr;
    ID3D11DeviceContext4 *context4 = nullptr;  // null without fence support
    ID3D11Fence *fence = nullptr;
    HANDLE fence_event = nullptr;
    HANDLE frame_event = nullptr;
    ID3D11Texture2D *textures[game_capture::kSlots] = {};
    IDXGIKeyedMutex *mutexes[game_capture::kSlots] = {};
    HANDLE shared_handles[game_capture::kSlots] = {};
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::uint32_t generation = 0;
    std::uint64_t fence_value = 0;
    std::uint64_t next_frame_id = 0;
    std::atomic<int> published_slot {-1};  // written by the fence thread
    bool reported_d3d12 = false;
  } g_cap;

  struct pending_t {
    std::uint64_t fence_value;
    std::uint32_t slot;
    std::uint32_t generation;
    std::uint64_t frame_id;
    std::uint64_t present_qpc;
  };

  std::mutex g_pending_lock;
  std::deque<pending_t> g_pending;
  HANDLE g_fence_thread = nullptr;

  void publish(const pending_t &frame, std::uint64_t gpu_done_qpc) {
    auto &b = *g_block;
    const auto seq = b.seq.load(std::memory_order_relaxed);
    b.seq.store(seq + 1, std::memory_order_release);  // odd: writing
    std::atomic_thread_fence(std::memory_order_release);
    b.latest_slot.store(frame.slot, std::memory_order_relaxed);
    b.latest_generation.store(frame.generation, std::memory_order_relaxed);
    b.frame_id.store(frame.frame_id, std::memory_order_relaxed);
    b.present_qpc.store(frame.present_qpc, std::memory_order_relaxed);
    b.gpu_done_qpc.store(gpu_done_qpc, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    b.seq.store(seq + 2, std::memory_order_release);  // even: stable
    b.frames_published.fetch_add(1, std::memory_order_relaxed);
    SetEvent(g_cap.frame_event);
  }

  // Publishes frames once the GPU has finished copying them, stamped with that
  // moment: the game's frame is complete when its copy is.
  DWORD WINAPI fence_thread_main(void *) {
    for (;;) {
      if (WaitForSingleObject(g_cap.fence_event, 100) == WAIT_FAILED) {
        return 1;
      }
      ID3D11Fence *fence = g_cap.fence;
      if (!fence) {
        continue;
      }
      const std::uint64_t completed = fence->GetCompletedValue();
      const std::uint64_t now = qpc_now();

      pending_t newest {};
      bool have = false;
      {
        std::lock_guard lg {g_pending_lock};
        while (!g_pending.empty() && g_pending.front().fence_value <= completed) {
          newest = g_pending.front();
          have = true;
          g_pending.pop_front();
        }
      }
      if (have) {
        g_cap.published_slot.store(static_cast<int>(newest.slot));
        publish(newest, now);
      }
    }
  }

  void release_textures() {
    for (int i = 0; i < game_capture::kSlots; ++i) {
      safe_release(g_cap.mutexes[i]);
      safe_release(g_cap.textures[i]);
      if (g_cap.shared_handles[i]) {
        CloseHandle(g_cap.shared_handles[i]);
        g_cap.shared_handles[i] = nullptr;
      }
    }
    std::lock_guard lg {g_pending_lock};
    g_pending.clear();
    g_cap.published_slot.store(-1);
  }

  void release_device() {
    release_textures();
    safe_release(g_cap.fence);
    safe_release(g_cap.context4);
    safe_release(g_cap.context);
    safe_release(g_cap.device);
  }

  bool ensure_device(ID3D11Device *device) {
    if (g_cap.device == device) {
      return true;
    }
    release_device();
    device->AddRef();
    g_cap.device = device;
    device->GetImmediateContext(&g_cap.context);

    // A fence tells us when the GPU has finished each copy; without one
    // (pre-1703 runtimes, some drivers) frames publish at Present time.
    ID3D11Device5 *device5 = nullptr;
    if (SUCCEEDED(device->QueryInterface(__uuidof(ID3D11Device5), reinterpret_cast<void **>(&device5)))) {
      if (SUCCEEDED(g_cap.context->QueryInterface(__uuidof(ID3D11DeviceContext4), reinterpret_cast<void **>(&g_cap.context4))) &&
          SUCCEEDED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, __uuidof(ID3D11Fence), reinterpret_cast<void **>(&g_cap.fence)))) {
        g_cap.fence_value = 0;
      } else {
        safe_release(g_cap.context4);
      }
      device5->Release();
    }
    if (!g_cap.fence) {
      log("No D3D11 fence support; frames are timestamped at Present");
    }

    IDXGIDevice *dxgi_device = nullptr;
    if (SUCCEEDED(device->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void **>(&dxgi_device)))) {
      IDXGIAdapter *adapter = nullptr;
      if (SUCCEEDED(dxgi_device->GetAdapter(&adapter))) {
        DXGI_ADAPTER_DESC desc {};
        adapter->GetDesc(&desc);
        g_block->adapter_luid_low.store(desc.AdapterLuid.LowPart, std::memory_order_relaxed);
        g_block->adapter_luid_high.store(desc.AdapterLuid.HighPart, std::memory_order_relaxed);
        adapter->Release();
      }
      dxgi_device->Release();
    }
    return true;
  }

  DXGI_FORMAT shareable_format(DXGI_FORMAT format) {
    // Swapchain buffers are never typeless in practice, but map the families
    // anyway so a CopyResource-compatible concrete format is always created
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

  bool ensure_textures(const D3D11_TEXTURE2D_DESC &back) {
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

    for (int i = 0; i < game_capture::kSlots; ++i) {
      HRESULT hr = g_cap.device->CreateTexture2D(&desc, nullptr, &g_cap.textures[i]);
      if (FAILED(hr)) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "CreateTexture2D(%ux%u fmt %d) failed: 0x%08lx", desc.Width, desc.Height, desc.Format, hr);
        set_state(game_capture::hook_state_e::failed, msg);
        release_textures();
        return false;
      }
      g_cap.textures[i]->QueryInterface(__uuidof(IDXGIKeyedMutex), reinterpret_cast<void **>(&g_cap.mutexes[i]));
      IDXGIResource1 *resource = nullptr;
      hr = g_cap.textures[i]->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void **>(&resource));
      if (SUCCEEDED(hr)) {
        hr = resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &g_cap.shared_handles[i]);
        resource->Release();
      }
      if (FAILED(hr) || !g_cap.mutexes[i]) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "Sharing the capture texture failed: 0x%08lx", hr);
        set_state(game_capture::hook_state_e::failed, msg);
        release_textures();
        return false;
      }
    }

    g_cap.width = back.Width;
    g_cap.height = back.Height;
    g_cap.format = format;
    ++g_cap.generation;

    auto &b = *g_block;
    for (int i = 0; i < game_capture::kSlots; ++i) {
      b.textures[i].store(reinterpret_cast<std::uint64_t>(g_cap.shared_handles[i]), std::memory_order_relaxed);
    }
    b.width.store(back.Width, std::memory_order_relaxed);
    b.height.store(back.Height, std::memory_order_relaxed);
    b.format.store(static_cast<std::uint32_t>(format), std::memory_order_relaxed);
    b.api.store(static_cast<std::uint32_t>(game_capture::api_e::d3d11), std::memory_order_relaxed);
    b.frame_event.store(reinterpret_cast<std::uint64_t>(g_cap.frame_event), std::memory_order_relaxed);
    b.generation.store(g_cap.generation, std::memory_order_release);
    log("Capture textures %ux%u format %d (generation %u, %s)", back.Width, back.Height, format, g_cap.generation,
        g_cap.fence ? "fenced" : "unfenced");
    return true;
  }

  bool is_foreground(IDXGISwapChain *swapchain) {
    DXGI_SWAP_CHAIN_DESC desc {};
    if (FAILED(swapchain->GetDesc(&desc)) || !desc.OutputWindow) {
      return false;
    }
    g_block->hwnd.store(reinterpret_cast<std::uint64_t>(desc.OutputWindow), std::memory_order_relaxed);
    const HWND foreground = GetForegroundWindow();
    return desc.OutputWindow == foreground || GetAncestor(desc.OutputWindow, GA_ROOT) == foreground;
  }

  void capture_frame(IDXGISwapChain *swapchain, std::uint64_t present_qpc) {
    g_block->frames_presented.fetch_add(1, std::memory_order_relaxed);
    if (!g_block->capture_enabled.load(std::memory_order_acquire) || !is_foreground(swapchain)) {
      return;
    }

    ID3D11Device *device = nullptr;
    if (FAILED(swapchain->GetDevice(__uuidof(ID3D11Device), reinterpret_cast<void **>(&device)))) {
      if (!g_cap.reported_d3d12) {
        g_cap.reported_d3d12 = true;
        set_state(game_capture::hook_state_e::unsupported, "The swapchain is not D3D11 (likely D3D12); not captured yet");
      }
      return;
    }
    ensure_device(device);
    device->Release();

    ID3D11Texture2D *back = nullptr;
    if (FAILED(swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&back)))) {
      return;
    }
    D3D11_TEXTURE2D_DESC back_desc {};
    back->GetDesc(&back_desc);
    if (!ensure_textures(back_desc)) {
      back->Release();
      return;
    }
    g_block->color_space.store(g_color_space.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // Only the slot the host is not reading from, and only without waiting
    const int slot = g_cap.published_slot.load() == 0 ? 1 : 0;
    if (g_cap.mutexes[slot]->AcquireSync(0, 0) != S_OK) {
      g_block->frames_skipped_busy.fetch_add(1, std::memory_order_relaxed);
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

    const pending_t frame {0, static_cast<std::uint32_t>(slot), g_cap.generation, ++g_cap.next_frame_id, present_qpc};
    if (g_cap.fence && g_cap.context4) {
      pending_t fenced = frame;
      fenced.fence_value = ++g_cap.fence_value;
      g_cap.context4->Signal(g_cap.fence, fenced.fence_value);
      {
        std::lock_guard lg {g_pending_lock};
        // A frame still in flight in this slot is overwritten by this one
        for (auto it = g_pending.begin(); it != g_pending.end();) {
          it = it->slot == fenced.slot ? g_pending.erase(it) : it + 1;
        }
        g_pending.push_back(fenced);
      }
      g_cap.fence->SetEventOnCompletion(fenced.fence_value, g_cap.fence_event);
    } else {
      g_cap.published_slot.store(slot);
      publish(frame, 0);
    }

    if (g_block->hook_state.load(std::memory_order_relaxed) != static_cast<std::uint32_t>(game_capture::hook_state_e::capturing)) {
      set_state(game_capture::hook_state_e::capturing);
      log("Capturing D3D11 frames");
    }
  }

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
      auto cs = game_capture::color_space_e::srgb;
      if (color_space == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709) {
        cs = game_capture::color_space_e::scrgb;
      } else if (color_space == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
        cs = game_capture::color_space_e::hdr10;
      }
      g_color_space.store(static_cast<std::uint32_t>(cs), std::memory_order_relaxed);
      log("Swapchain color space %d", static_cast<int>(color_space));
    }
    return hr;
  }

  template<class Fn>
  bool patch_slot(void **vtable, int index, void *hook, Fn &original) {
    DWORD old_protect = 0;
    if (!VirtualProtect(&vtable[index], sizeof(void *), PAGE_READWRITE, &old_protect)) {
      return false;
    }
    original = reinterpret_cast<Fn>(vtable[index]);
    vtable[index] = hook;
    VirtualProtect(&vtable[index], sizeof(void *), old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), &vtable[index], sizeof(void *));
    return true;
  }

  // The DXGI swapchain implementation's vtable is shared by every swapchain
  // in the process, so patching it through a throwaway flip-model swapchain
  // hooks the game's too. Overlays that inline-hook Present are unaffected:
  // our entry calls through the original pointer, into their hook.
  bool install_hooks() {
    WNDCLASSW wc {};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"VibepolloGameHookDummy";
    RegisterClassW(&wc);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"", WS_OVERLAPPEDWINDOW, 0, 0, 16, 16, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) {
      set_state(game_capture::hook_state_e::failed, "Could not create the dummy window");
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
      ok = patch_slot(vtable, kVtPresent, reinterpret_cast<void *>(&hook_present), g_real_present) &&
           patch_slot(vtable, kVtPresent1, reinterpret_cast<void *>(&hook_present1), g_real_present1);
      IDXGISwapChain3 *swapchain3 = nullptr;
      if (ok && SUCCEEDED(swapchain->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&swapchain3)))) {
        void **vtable3 = *reinterpret_cast<void ***>(swapchain3);
        patch_slot(vtable3, kVtSetColorSpace1, reinterpret_cast<void *>(&hook_set_color_space1), g_real_set_color_space1);
        swapchain3->Release();
      }
    } else {
      char msg[96];
      std::snprintf(msg, sizeof(msg), "Could not create the dummy swapchain: 0x%08lx", hr);
      set_state(game_capture::hook_state_e::failed, msg);
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
    game_capture::shared_block_name(name, 96, GetCurrentProcessId());
    HANDLE mapping = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name);
    if (!mapping) {
      log("No shared block %ls (error %lu); not capturing", name, GetLastError());
      return 1;
    }
    g_block = static_cast<game_capture::shared_block_t *>(MapViewOfFile(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(game_capture::shared_block_t)));
    if (!g_block || g_block->magic != game_capture::kMagic || g_block->version != game_capture::kVersion) {
      log("Shared block missing or version mismatch; not capturing");
      g_block = nullptr;
      return 1;
    }

    g_cap.frame_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    g_cap.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!g_cap.frame_event || !g_cap.fence_event) {
      set_state(game_capture::hook_state_e::failed, "Could not create events");
      return 1;
    }
    g_block->frame_event.store(reinterpret_cast<std::uint64_t>(g_cap.frame_event), std::memory_order_relaxed);
    g_fence_thread = CreateThread(nullptr, 0, fence_thread_main, nullptr, 0, nullptr);

    if (!install_hooks()) {
      return 1;
    }
    set_state(game_capture::hook_state_e::hooked);
    log("Hooked DXGI Present/Present1 in pid %lu", GetCurrentProcessId());
    return 0;
  }

}  // namespace

// The DLL stays loaded for the life of the process: the patched vtable points
// into it, and there is no safe moment to take that back mid-frame.
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
