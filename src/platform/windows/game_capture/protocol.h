/**
 * @file src/platform/windows/game_capture/protocol.h
 * @brief Shared layout between Vibepollo and the in-game capture hook
 *        (tools/game_hook). Header-only; no dependencies beyond the standard
 *        library so the hook DLL stays small.
 *
 * Flow:
 *  1. The host creates the shared block `Global\VibepolloGameCapture_<pid>`
 *     (so the SYSTEM-owned name is valid across sessions and the game process
 *     never needs SeCreateGlobalPrivilege), then injects the hook DLL.
 *  2. The hook patches the DXGI swapchain vtable. On Present it copies the
 *     back buffer into one of kSlots shared textures, each guarded by a DXGI
 *     keyed mutex (GPU-ordered across devices, so the host can never read a
 *     half-written texture), and publishes the frame once the GPU has finished
 *     the copy (D3D11 fence), with the game's timing.
 *  3. The host duplicates the hook's NT handles (texture per slot, frame
 *     event) out of the game process and reads the latest slot.
 *
 * Slot rule that makes this race-free without ever blocking the game: the
 * hook only writes the slot that is NOT currently published, and only if it
 * gets that slot's keyed mutex with a zero timeout; the host only reads the
 * published slot. A frame the hook cannot place is skipped (counted), never
 * waited for.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <cwchar>

namespace game_capture {

  constexpr std::uint32_t kMagic = 0x50434756;  // "VGCP"
  constexpr std::uint32_t kVersion = 1;
  constexpr int kSlots = 2;

  inline void shared_block_name(wchar_t *buffer, std::size_t count, std::uint32_t pid) {
    std::swprintf(buffer, count, L"Global\\VibepolloGameCapture_%u", pid);
  }

  enum class hook_state_e : std::uint32_t {
    none = 0,  ///< not loaded yet
    hooked = 1,  ///< vtable patched, no frame captured yet
    capturing = 2,  ///< frames are being published
    unsupported = 3,  ///< this process presents through an API the hook does not capture (see last_error)
    failed = 4,  ///< setup failed (see last_error)
  };

  enum class api_e : std::uint32_t {
    unknown = 0,
    d3d11 = 1,
  };

  // How the game's pixels are encoded (from IDXGISwapChain3::SetColorSpace1)
  enum class color_space_e : std::uint32_t {
    srgb = 0,  ///< DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 (the default)
    scrgb = 1,  ///< DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709, linear, 1.0 = 80 nits
    hdr10 = 2,  ///< DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, PQ
  };

  struct shared_block_t {
    std::uint32_t magic;
    std::uint32_t version;

    // Host -> hook
    std::atomic<std::uint32_t> capture_enabled;  ///< the hook copies only while non-zero

    // Hook -> host: setup, rewritten when the textures are recreated. The host
    // re-duplicates the texture handles whenever `generation` changes.
    std::atomic<std::uint32_t> hook_state;  ///< hook_state_e
    std::atomic<std::uint32_t> api;  ///< api_e
    std::atomic<std::uint32_t> generation;
    std::atomic<std::uint64_t> frame_event;  ///< auto-reset event (handle value in the game process)
    std::atomic<std::uint64_t> textures[kSlots];  ///< NT handles (values in the game process)
    std::atomic<std::uint32_t> width;
    std::atomic<std::uint32_t> height;
    std::atomic<std::uint32_t> format;  ///< DXGI_FORMAT
    std::atomic<std::uint32_t> color_space;  ///< color_space_e
    std::atomic<std::uint64_t> hwnd;  ///< the swapchain's output window
    std::atomic<std::uint32_t> adapter_luid_low;
    std::atomic<std::int32_t> adapter_luid_high;

    // Hook -> host: the latest completed frame, seqlock-published (odd while
    // being written). Read `seq`, the fields, then `seq` again.
    std::atomic<std::uint32_t> seq;
    std::atomic<std::uint32_t> latest_slot;
    std::atomic<std::uint32_t> latest_generation;
    std::atomic<std::uint64_t> frame_id;
    std::atomic<std::uint64_t> present_qpc;  ///< QueryPerformanceCounter at the game's Present()
    std::atomic<std::uint64_t> gpu_done_qpc;  ///< when the GPU finished the copy (0 = no fence support)

    // Hook -> host: statistics
    std::atomic<std::uint64_t> frames_presented;
    std::atomic<std::uint64_t> frames_published;
    std::atomic<std::uint64_t> frames_skipped_busy;  ///< both slots held: frame not captured

    char last_error[160];
  };

  static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "shared block atomics must be lock-free");

}  // namespace game_capture
