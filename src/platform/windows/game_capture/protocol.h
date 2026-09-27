/**
 * @file src/platform/windows/game_capture/protocol.h
 * @brief Shared layout between Vibepollo and the in-game capture hook
 *        (tools/game_hook). Header-only; no dependencies beyond the standard
 *        library so the hook DLL stays small.
 *
 * Flow:
 *  1. The host creates the shared block `Global\VibepolloGameCapture_<pid>`
 *     (writable by SYSTEM, administrators and the game's own user; medium
 *     integrity label so the game can write it) and injects the hook DLL.
 *  2. The hook detours the DXGI Present entry points. On Present it copies the
 *     back buffer into a free slot (a shared texture guarded by a DXGI keyed
 *     mutex, GPU-ordered across devices) and, once the GPU has finished the
 *     copy, publishes that slot with the game's timing.
 *  3. The host duplicates the hook's NT handles out of the game process and
 *     reads the published slot.
 *
 * Slot protocol (three slots: free, pending, published):
 *  - The hook's render thread writes only a FREE slot, taking its keyed
 *    mutex with a zero timeout; if none is free or the mutex is held, the
 *    frame is skipped. Present is never blocked and never waits.
 *  - Each slot has its own record here, seqlocked by the render thread: the
 *    record's version is what identifies the pixels currently in that slot.
 *  - A written slot is PENDING until the hook's completion thread sees its
 *    copy finish; that thread is the ONLY writer of `latest`. It publishes
 *    (slot, version) first and only then frees the previously published
 *    slot, and it never publishes a frame older than the last one.
 *  - The host takes the keyed mutex of the slot `latest` names, then reads
 *    that slot's record and uses the frame only if the record's version is
 *    the published one. A slot rewritten meanwhile has a new version, so
 *    pixels and metadata can never be mixed.
 *
 * Everything the host reads from this block is untrusted: the game's user
 * can write it. The host bounds and validates every field.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <cwchar>

namespace game_capture {

  constexpr std::uint32_t kMagic = 0x50434756;  // "VGCP"
  constexpr std::uint32_t kVersion = 3;
  constexpr int kSlots = 3;
  constexpr std::size_t kErrorLength = 160;

  inline void shared_block_name(wchar_t *buffer, std::size_t count, std::uint32_t pid) {
    std::swprintf(buffer, count, L"Global\\VibepolloGameCapture_%u", pid);
  }

  enum class hook_state_e : std::uint32_t {
    none = 0,  ///< not loaded yet
    hooked = 1,  ///< detours installed, no frame captured yet
    capturing = 2,  ///< frames are being published
    unsupported = 3,  ///< this process presents through an API the hook does not capture (see last_error)
    failed = 4,  ///< setup failed (see last_error)
  };

  enum class api_e : std::uint32_t {
    unknown = 0,
    d3d11 = 1,
  };

  // How the game's pixels are encoded
  enum class color_space_e : std::uint32_t {
    unknown = 0,  ///< not established (a 10-bit swapchain whose SetColorSpace1 the hook never saw): do not use the frame
    srgb = 1,  ///< DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709
    scrgb = 2,  ///< DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709, linear, 1.0 = 80 nits
    hdr10 = 3,  ///< DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, PQ
  };

  // Texture setup for one generation (rewritten on every resize / device
  // change), published as a unit under setup_seq. Handles of the previous
  // generation stay open in the hook until the one after it, so a host still
  // opening them never meets a recycled handle value.
  struct setup_t {
    std::atomic<std::uint32_t> generation;
    std::atomic<std::uint32_t> api;  ///< api_e
    std::atomic<std::uint32_t> width;
    std::atomic<std::uint32_t> height;
    std::atomic<std::uint32_t> format;  ///< DXGI_FORMAT (informational; the host uses the opened texture's own description)
    std::atomic<std::uint32_t> adapter_luid_low;
    std::atomic<std::int32_t> adapter_luid_high;
    std::atomic<std::uint64_t> hwnd;  ///< the swapchain's output window
    std::atomic<std::uint64_t> textures[kSlots];  ///< NT handle values in the game process (0 = none)
  };

  // What one slot's texture holds. `seq` is odd while the render thread
  // rewrites the record; version = seq >> 1.
  struct slot_record_t {
    std::atomic<std::uint32_t> seq;
    std::atomic<std::uint32_t> generation;
    std::atomic<std::uint32_t> color_space;  ///< color_space_e
    std::atomic<std::uint64_t> frame_id;
    std::atomic<std::uint64_t> present_qpc;  ///< QueryPerformanceCounter at the game's Present()
    std::atomic<std::uint64_t> gpu_done_qpc;  ///< written by the completion thread just before publishing (0 = no fence)
  };

  inline std::uint64_t make_latest(std::uint32_t slot, std::uint32_t version) {
    return (static_cast<std::uint64_t>(slot) << 32) | version;
  }

  struct shared_block_t {
    std::uint32_t magic;
    std::uint32_t version;

    // Host -> hook
    std::atomic<std::uint32_t> capture_enabled;  ///< the hook copies only while non-zero

    // Hook -> host
    std::atomic<std::uint32_t> hook_state;  ///< hook_state_e
    std::atomic<std::uint64_t> frame_event;  ///< auto-reset event (handle value in the game process), set on each publish
    std::atomic<std::uint64_t> publish_qpc;  ///< QueryPerformanceCounter of the last publish (for staleness)

    std::atomic<std::uint32_t> setup_seq;  ///< odd while setup is being rewritten
    setup_t setup;

    slot_record_t slots[kSlots];
    std::atomic<std::uint64_t> latest;  ///< make_latest(slot, version) of the published frame; 0 = none. Single writer.

    // Statistics
    std::atomic<std::uint64_t> frames_presented;
    std::atomic<std::uint64_t> frames_published;
    std::atomic<std::uint64_t> frames_skipped;  ///< no free slot / mutex held / another thread capturing

    char last_error[kErrorLength];  ///< the host copies at most kErrorLength bytes and terminates locally
  };

  static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "shared block atomics must be lock-free");

}  // namespace game_capture
