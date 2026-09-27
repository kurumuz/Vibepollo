/**
 * @file src/platform/windows/game_capture/game_source.h
 * @brief Host side of in-game capture: decides when a focused fullscreen game
 *        should be captured from inside, injects the hook, and hands its
 *        frames to the capture backend (see protocol.h).
 */
#pragma once

#include "protocol.h"
#include "src/platform/common.h"
#include "src/platform/windows/foreground_app.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include <d3d11_1.h>
#include <winrt/base.h>

namespace platf::dxgi::game_capture {

  struct frame_t {
    ID3D11Texture2D *texture = nullptr;  ///< keyed mutex held until source_t::unlock()
    std::uint32_t width = 0;  ///< from the opened texture, not the shared block
    std::uint32_t height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    ::game_capture::color_space_e color_space = ::game_capture::color_space_e::unknown;
    std::uint64_t frame_id = 0;
    std::uint64_t present_qpc = 0;  ///< validated: within the last second
    std::uint64_t gpu_done_qpc = 0;  ///< 0 when the hook had no fence
  };

  class source_t {
  public:
    explicit source_t(ID3D11Device *device);
    ~source_t();

    source_t(const source_t &) = delete;
    source_t &operator=(const source_t &) = delete;

    /**
     * @brief Whether this snapshot should come from the game. Tracks the
     *        foreground process, attaches the hook (on a worker, never
     *        blocking capture) the first time a candidate is focused and
     *        fullscreen, and enables or disables copying in hooked processes.
     *        True only while the focused fullscreen window belongs to a
     *        hooked process that published a frame recently.
     */
    bool active(const foreground_app::state_t &foreground);

    /**
     * @brief Cheap re-check right before a frame is taken from the game:
     *        the foreground window still belongs to the captured process
     *        and still covers the captured display.
     */
    bool still_foreground(const RECT &capture_rect) const;

    /**
     * @brief Wait for the game to publish a frame newer than the last one used.
     */
    capture_e wait(std::chrono::milliseconds timeout);

    /**
     * @brief Lock the newest published frame (keyed mutex held on success).
     */
    capture_e lock(frame_t &frame);
    void unlock();

  private:
    struct attach_job_t;
    struct target_t;

    void reap_exited();
    bool open_generation(target_t &target);
    target_t *current();

    winrt::com_ptr<ID3D11Device1> _device;
    LUID _adapter_luid {};
    std::map<DWORD, std::unique_ptr<target_t>> _targets;
    DWORD _current_pid = 0;
    int _locked_slot = -1;
    std::chrono::steady_clock::time_point _last_reap {};
  };

  /**
   * @brief Converts a game frame into the capture texture's format and color
   *        encoding (the one the desktop compositor would have produced).
   */
  class converter_t {
  public:
    /**
     * @return false when this source/target combination is not handled
     *         (caller falls back to desktop capture).
     */
    bool convert(ID3D11Device *device, ID3D11DeviceContext *context, const frame_t &frame, ID3D11Texture2D *target_texture, ID3D11RenderTargetView *target_rtv, DXGI_FORMAT target_format, float sdr_white_scale);

  private:
    bool init(ID3D11Device *device);

    bool _init_attempted = false;
    bool _ready = false;  // every resource below created
    winrt::com_ptr<ID3D11VertexShader> _vs;
    winrt::com_ptr<ID3D11PixelShader> _ps;
    winrt::com_ptr<ID3D11Buffer> _params;
    winrt::com_ptr<ID3D11SamplerState> _sampler;
    ID3D11Texture2D *_srv_texture = nullptr;  // identity only, for SRV reuse
    winrt::com_ptr<ID3D11ShaderResourceView> _srv;
  };

  /**
   * @brief The desktop compositor's current SDR white level for an output, as
   *        a scRGB multiplier (1.0 = 80 nits). 1.0 when unavailable.
   */
  float sdr_white_scale_for_output(const wchar_t *gdi_device_name);

}  // namespace platf::dxgi::game_capture
