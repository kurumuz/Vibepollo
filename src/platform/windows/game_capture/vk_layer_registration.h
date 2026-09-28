/**
 * @file src/platform/windows/game_capture/vk_layer_registration.h
 * @brief Keeps Vibepollo's Vulkan layer (tools/vk_layer) registered as the
 *        last Vulkan implicit layer.
 *
 * The Vulkan loader orders implicit layers as their registry values are
 * listed under HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers, first nearest the
 * application (checked with VK_LOADER_DEBUG=layer on the target machine). Last
 * means nearest the driver: below an overlay's layer (Steam's), whose drawing
 * is then in the frame our layer copies. Another layer registered after ours
 * (an update re-registering Steam's, say) would move above us, so each
 * start of game capture checks the order and moves our value back to the
 * end if needed.
 */
#pragma once

namespace game_capture {

  /**
   * @brief With game capture on: registers the layer's manifest (installed
   *        beside the hook in tools/) as the last, enabled implicit layer,
   *        if it is not already. Off: removes the registration, so the
   *        layer is not loaded into every Vulkan application for nothing.
   *        Needs HKLM write access (the host runs as SYSTEM); on failure it
   *        tries again on the next call. Logs what it did.
   */
  void sync_vk_layer_registration(bool enabled);

  /// sync_vk_layer_registration(true), at most once successfully per process
  void ensure_vk_layer_registered();

}  // namespace game_capture
