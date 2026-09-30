/**
 * @file src/platform/windows/display_vram.h
 * @brief DXGI/D3D11 VRAM image structures and utilities for Windows platform display capture.
 */
#pragma once

// standard includes
#include <memory>

// local includes
#include "display.h"

// platform includes
#include <d3d11.h>
#include <dxgi.h>

namespace platf::dxgi {
  namespace game_capture {
    struct motion_field_t;
  }

  /**
   * @brief Direct3D-backed image container used for WGC/DXGI capture paths.
   *
   * Extends platf::img_t with Direct3D 11 resources required for capture and
   * inter-process texture sharing.
   */
  struct img_d3d_t: public platf::img_t {
    texture2d_t capture_texture;  ///< Staging/CPU readable or GPU shared texture.
    render_target_t capture_rt;  ///< Render target bound when copying / compositing.
    keyed_mutex_t capture_mutex;  ///< Keyed mutex for cross-process synchronization.
    HANDLE encoder_texture_handle = {};  ///< Duplicated shared handle opened by encoder side.
    bool dummy = false;  ///< True if placeholder prior to first successful frame.
    bool blank = true;  ///< True if contains no desktop or cursor content.
    uint32_t id = 0;  ///< Monotonically increasing identifier.
    DXGI_FORMAT format;  ///< Underlying DXGI texture format.
    std::uint64_t game_frame_id = 0;  ///< In-game capture: the game frame it holds (0: not a game frame)
    std::shared_ptr<const game_capture::motion_field_t> motion;  ///< ... and that frame's motion hints (null: none)

    ~img_d3d_t() override {
      if (encoder_texture_handle) {
        CloseHandle(encoder_texture_handle);
      }
    }
  };

}  // namespace platf::dxgi
