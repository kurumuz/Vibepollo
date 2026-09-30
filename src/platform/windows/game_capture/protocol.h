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
 * Without keyed mutexes (sync_e::owner) the per-slot owner word takes the
 * mutex's place: the hook claims a free slot's word while it rewrites the
 * record and submits the copy (the copy may still be running when it lets
 * go: the slot publishes only after the hook's fence saw it complete, and a
 * host that claims it before then finds a record version that is not the
 * published one). The host claims the published slot's word before reading
 * it and lets go only once its own GPU reads of it completed. One host at a
 * time: host_lock names it.
 *
 * Frame limiter: while limiter_period_ps is non-zero and host_heartbeat_qpc
 * is recent, the hook paces the game from inside Present (front edge: it
 * waits after the real Present returns, so the game starts its next frame,
 * and samples input, right at the release time). Each frame's record carries
 * the release that started it.
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
 * Motion vectors (while motion_enabled): the hook detours the driver's DLSS
 * entry points (_nvngx.dll EvaluateFeature, D3D11 and D3D12). After each
 * successful DLSS evaluation it appends, to the game's own command list, a
 * copy of the evaluation's motion-vector region into a ring of its own; the
 * next captured frame's copy also moves the vectors DLSS saw for it into the
 * slot's motion texture, under the slot's ownership like its pixels. A
 * generation that has motion textures says so in setup_t (they are created,
 * retired and shared with its textures); a slot record whose motion_id is 0
 * carries none. The vectors are DLSS's: in pixels of the region they cover
 * (after multiplying by motion_scale), pointing from a pixel to where it was
 * in the previous frame. Consecutive evaluations have consecutive ids, so a
 * frame's vectors describe the step from the previous captured frame only
 * when both ids and frame ids are consecutive.
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
  constexpr std::uint32_t kVersion = 13;
  constexpr int kSlots = 3;
  constexpr std::size_t kErrorLength = 320;

  // Limiter periods the hook accepts (1000 fps .. 10 fps)
  constexpr std::uint64_t kMinLimiterPeriodPs = 1'000'000'000ull;
  constexpr std::uint64_t kMaxLimiterPeriodPs = 100'000'000'000ull;
  // The limiter stops when the host has not refreshed its heartbeat this long
  constexpr std::uint64_t kHeartbeatTimeoutMs = 2000;

  inline void shared_block_name(wchar_t *buffer, std::size_t count, std::uint32_t pid) {
    // Versioned: a mapping left by an earlier implementation (other layout,
    // wider ACL) is never reattached
    std::swprintf(buffer, count, L"Global\\VibepolloGameCapture%u_%u", kVersion, pid);
  }

  enum class hook_state_e : std::uint32_t {
    none = 0,  ///< not loaded yet
    hooked = 1,  ///< detours installed, no frame captured yet
    capturing = 2,  ///< frames are being published
    unsupported = 3,  ///< this process presents through an API the hook does not capture (see last_error)
    failed = 4,  ///< setup failed (see last_error)
    fatal = 5,  ///< Windows no longer behaves as the hook relies on (see last_error): Vibepollo needs updating; the host stops loudly
  };

  enum class api_e : std::uint32_t {
    unknown = 0,
    d3d11 = 1,
    d3d12 = 2,
    vulkan = 3,  ///< a Vulkan swapchain (frames copied through the Vulkan layer, into D3D11 textures the hook shares)
  };

  // How the game's pixels are encoded
  enum class color_space_e : std::uint32_t {
    unknown = 0,  ///< one the host does not convert (or the hook could not read it): do not use the frame
    srgb = 1,  ///< DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709
    scrgb = 2,  ///< DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709, linear, 1.0 = 80 nits
    hdr10 = 3,  ///< DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, PQ
  };

  // How setup_t::textures are shared
  enum class handle_kind_e : std::uint32_t {
    nt = 0,  ///< NT handles in the game process: the host duplicates them
    legacy = 1,  ///< global (KMT) share handles, opened directly (for devices that refuse NT-handle sharing)
  };

  // How a slot's pixels are guarded between the hook's copy and the host's read
  enum class sync_e : std::uint32_t {
    keyed_mutex = 0,  ///< a DXGI keyed mutex per texture (GPU-ordered)
    owner = 1,  ///< shared_block_t::owner, for devices that cannot create keyed-mutex textures. The hook publishes only
                ///< after its fence saw the copy complete; the host releases a slot only after its own reads completed.
  };

  // shared_block_t::owner values
  constexpr std::uint32_t kOwnerNone = 0;
  constexpr std::uint32_t kOwnerHook = 1;  ///< rewriting the slot's record and submitting its copy
  constexpr std::uint32_t kOwnerHost = 2;  ///< reading the slot (until its GPU reads completed)

  // Texture setup for one generation (rewritten on every resize / device
  // change), published as a unit under setup_seq. The hook keeps a retired
  // generation's handles (NT) or textures (legacy) for a two-second grace
  // period, and the host re-checks the generation after opening, so a host
  // still opening an old setup does not commit a recycled handle value.
  struct setup_t {
    std::atomic<std::uint32_t> generation;
    std::atomic<std::uint32_t> api;  ///< api_e
    std::atomic<std::uint32_t> width;
    std::atomic<std::uint32_t> height;
    std::atomic<std::uint32_t> format;  ///< DXGI_FORMAT (informational; the host uses the opened texture's own description)
    std::atomic<std::uint32_t> adapter_luid_low;
    std::atomic<std::int32_t> adapter_luid_high;
    std::atomic<std::uint64_t> hwnd;  ///< the swapchain's output window
    std::atomic<std::uint32_t> handle_kind;  ///< handle_kind_e
    std::atomic<std::uint32_t> sync;  ///< sync_e
    std::atomic<std::uint64_t> textures[kSlots];  ///< handle values, see handle_kind (0 = none)
    // Motion textures of this generation (same handle kind; guarded with the
    // slot's pixels: no mutex of their own). format 0 = the generation has none.
    std::atomic<std::uint32_t> motion_format;  ///< DXGI_FORMAT_R16G16_FLOAT or DXGI_FORMAT_R32G32_FLOAT
    std::atomic<std::uint32_t> motion_width;
    std::atomic<std::uint32_t> motion_height;
    std::atomic<std::uint64_t> motion_textures[kSlots];
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
    std::atomic<std::uint64_t> release_qpc;  ///< the limiter release that started this frame (0 = not paced)
    // The motion vectors in the slot's motion texture (see the top comment)
    std::atomic<std::uint64_t> motion_id;  ///< the DLSS evaluation they come from; 0 = none for this frame
    std::atomic<std::uint32_t> motion_width;  ///< the region holding them, from the texture's top-left
    std::atomic<std::uint32_t> motion_height;
    std::atomic<std::uint32_t> motion_out_width;  ///< the DLSS output the region spans
    std::atomic<std::uint32_t> motion_out_height;
    std::atomic<std::uint32_t> motion_scale_x;  ///< float bits: DLSS's MV.Scale (vector = stored value * scale)
    std::atomic<std::uint32_t> motion_scale_y;
  };

  inline std::uint64_t make_latest(std::uint32_t slot, std::uint32_t version) {
    return (static_cast<std::uint64_t>(slot) << 32) | version;
  }

  struct shared_block_t {
    std::uint32_t magic;
    std::uint32_t version;

    // Host -> hook
    std::atomic<std::uint32_t> capture_enabled;  ///< the hook copies only while non-zero
    std::atomic<std::uint32_t> recreate_request;  ///< bumped by the host after an abandoned slot mutex: the hook recreates its textures
    std::atomic<std::uint64_t> host_lock;  ///< (host pid << 32) | host instance of the one host reading this block; 0 = none
    std::atomic<std::uint64_t> host_heartbeat_qpc;  ///< refreshed by the host while it streams; the limiter stops when it goes stale
    std::atomic<std::uint64_t> limiter_period_ps;  ///< frame period the hook paces the game at, picoseconds; 0 = no limiter
    // dxgi!CDXGISwapChain::PresentImpl, resolved by the host from Microsoft's
    // public symbols for the system dxgi.dll it describes (PE timestamp and
    // image size); written before injection, 0 = not resolved. The hook
    // copies D3D11 frames at that function's entry, after overlays hooked on
    // the public Present have drawn, and verifies the address itself first.
    std::atomic<std::uint32_t> dxgi_timestamp;
    std::atomic<std::uint32_t> dxgi_image_size;
    std::atomic<std::uint32_t> dxgi_present_impl_rva;
    std::atomic<std::uint32_t> motion_enabled;  ///< the hook captures DLSS motion vectors only while non-zero

    // Hook -> host
    std::atomic<std::uint32_t> hook_state;  ///< hook_state_e
    std::atomic<std::uint64_t> frame_event;  ///< auto-reset event (handle value in the game process), set on each publish
    std::atomic<std::uint64_t> publish_qpc;  ///< QueryPerformanceCounter of the last publish (for staleness)

    std::atomic<std::uint32_t> setup_seq;  ///< odd while setup is being rewritten
    setup_t setup;

    slot_record_t slots[kSlots];
    std::atomic<std::uint32_t> owner[kSlots];  ///< kOwner*, claimed by compare-exchange from kOwnerNone (sync_e::owner only)
    std::atomic<std::uint64_t> latest;  ///< make_latest(slot, version) of the published frame; 0 = none. Single writer.

    // Statistics
    std::atomic<std::uint64_t> frames_presented;
    std::atomic<std::uint64_t> frames_published;
    std::atomic<std::uint64_t> frames_skipped;  ///< no free slot / mutex held / another thread capturing
    std::atomic<std::uint64_t> limiter_waits;  ///< Presents the limiter paced
    std::atomic<std::uint64_t> limiter_late;  ///< ... that arrived after their release time (no wait)
    std::atomic<std::uint64_t> limiter_resets;  ///< ... so late the release grid restarted
    std::atomic<std::uint64_t> limiter_wait_us;  ///< total time spent waiting

    char last_error[kErrorLength];  ///< the host copies at most kErrorLength bytes and terminates locally
  };

  static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "shared block atomics must be lock-free");

}  // namespace game_capture
