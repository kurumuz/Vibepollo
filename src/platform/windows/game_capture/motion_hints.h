/**
 * @file src/platform/windows/game_capture/motion_hints.h
 * @brief A DLSS game's motion vectors (see protocol.h) as one vector per
 *        16x16 block of the captured frame, checked against the frames
 *        themselves, for the encoder's motion search.
 */
#pragma once

#include "game_source.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <d3d11.h>
#include <winrt/base.h>

namespace platf::dxgi::game_capture {

  /**
   * @brief Per 16x16 block of one captured game frame: where the block was in
   *        the previous game frame, in quarter pixels.
   */
  struct motion_field_t {
    static constexpr std::int32_t kNone = std::numeric_limits<std::int32_t>::min();  ///< in x: no vector for the block

    std::uint64_t frame_id = 0;  ///< the game frame (frame_t::frame_id) it belongs to; its predecessor is frame_id - 1
    std::uint32_t width = 0;  ///< the frame's size in pixels
    std::uint32_t height = 0;
    std::uint32_t cols = 0;  ///< blocks
    std::uint32_t rows = 0;
    std::vector<std::int32_t> vectors;  ///< (x, y) per block, row by row
  };

  /**
   * @brief Turns each game frame's DLSS vectors into a block field on the
   *        capture device (a compute pass and a small readback).
   */
  class motion_pass_t {
  public:
    /**
     * @brief For a locked game frame: its block field, or null when it has no
     *        usable vectors (none, or the previous frame it would be checked
     *        against was not the game's previous frame). Every game frame
     *        should come through here: each one's luma checks the next.
     */
    std::shared_ptr<const motion_field_t> run(ID3D11Device *device, ID3D11DeviceContext *context, const frame_t &frame);

    /**
     * @brief A frame that was not the game's came in between: the next one
     *        cannot be checked.
     */
    void reset();

    /**
     * @brief What the checks found since the last call (for the periodic
     *        log line), and resets the counts.
     */
    std::string stats();

  private:
    bool init(ID3D11Device *device, std::uint32_t width, std::uint32_t height);

    bool _init_failed = false;
    bool _ready = false;
    std::uint32_t _width = 0;
    std::uint32_t _height = 0;
    winrt::com_ptr<ID3D11ComputeShader> _luma_cs;
    winrt::com_ptr<ID3D11ComputeShader> _blocks_cs;
    winrt::com_ptr<ID3D11Buffer> _params;
    winrt::com_ptr<ID3D11Texture2D> _luma[2];
    winrt::com_ptr<ID3D11ShaderResourceView> _luma_srv[2];
    winrt::com_ptr<ID3D11UnorderedAccessView> _luma_uav[2];
    winrt::com_ptr<ID3D11Buffer> _field;
    winrt::com_ptr<ID3D11UnorderedAccessView> _field_uav;
    winrt::com_ptr<ID3D11Buffer> _staging;
    winrt::com_ptr<ID3D11Buffer> _stats;  // the shader's counters (raw buffer)
    winrt::com_ptr<ID3D11UnorderedAccessView> _stats_uav;
    winrt::com_ptr<ID3D11Buffer> _stats_staging;
    std::uint64_t _counts[8] = {};  // accumulated counters (see the shader)
    std::uint64_t _fields = 0;  // fields handed out
    int _current = 0;  // _luma[_current] takes the next frame's luma

    // The previous frame whose luma is in _luma[1 - _current]
    bool _previous_valid = false;
    std::uint64_t _previous_frame_id = 0;
    std::uint64_t _previous_motion_id = 0;

    // Views of the frame's textures (identity only, for reuse)
    ID3D11Texture2D *_color_texture = nullptr;
    winrt::com_ptr<ID3D11ShaderResourceView> _color_srv;
    ID3D11Texture2D *_motion_texture = nullptr;
    winrt::com_ptr<ID3D11ShaderResourceView> _motion_srv;
    bool _logged_aspect = false;
  };

}  // namespace platf::dxgi::game_capture
