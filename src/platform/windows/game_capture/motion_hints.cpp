/**
 * @file src/platform/windows/game_capture/motion_hints.cpp
 * @brief DLSS motion vectors to per-block encoder hints (see motion_hints.h and
 *        game_capture_motion_cs.hlsl).
 */
#include "motion_hints.h"

#include "src/logging.h"
#include "src/platform/windows/display.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#define GAME_CAPTURE_SHADERS_DIR SUNSHINE_ASSETS_DIR "/shaders/directx"

namespace platf::dxgi {
  blob_t compile_shader(LPCSTR file, LPCSTR entrypoint, LPCSTR shader_model);
}  // namespace platf::dxgi

namespace platf::dxgi::game_capture {

  namespace {
    struct params_t {
      std::uint32_t frame_size[2];
      std::uint32_t blocks[2];
      float texel_per_pixel[2];
      float pixel_per_unit[2];
      std::uint32_t linear_light;
      std::uint32_t verify;
      std::uint32_t pad[2];
    };

    static_assert(sizeof(params_t) % 16 == 0, "constant buffer size");

    // Whether a view of this format reads linear light (the luma is then
    // compressed, so a difference means about the same everywhere)
    bool linear_format(DXGI_FORMAT f) {
      return f == DXGI_FORMAT_R16G16B16A16_FLOAT || f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    }
  }  // namespace

  bool motion_pass_t::init(ID3D11Device *device, std::uint32_t width, std::uint32_t height) {
    _ready = false;
    _previous_valid = false;
    if (!_luma_cs) {
      auto luma = compile_shader(GAME_CAPTURE_SHADERS_DIR "/game_capture_motion_cs.hlsl", "luma_cs", "cs_5_0");
      auto blocks = compile_shader(GAME_CAPTURE_SHADERS_DIR "/game_capture_motion_cs.hlsl", "blocks_cs", "cs_5_0");
      if (!luma || !blocks ||
          FAILED(device->CreateComputeShader(luma->GetBufferPointer(), luma->GetBufferSize(), nullptr, _luma_cs.put())) ||
          FAILED(device->CreateComputeShader(blocks->GetBufferPointer(), blocks->GetBufferSize(), nullptr, _blocks_cs.put()))) {
        _luma_cs = nullptr;
        _blocks_cs = nullptr;
        return false;
      }
      D3D11_BUFFER_DESC bd {};
      bd.ByteWidth = sizeof(params_t);
      bd.Usage = D3D11_USAGE_DEFAULT;
      bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      if (FAILED(device->CreateBuffer(&bd, nullptr, _params.put()))) {
        return false;
      }
    }

    for (int i = 0; i < 2; ++i) {
      _luma[i] = nullptr;
      _luma_srv[i] = nullptr;
      _luma_uav[i] = nullptr;
      D3D11_TEXTURE2D_DESC td {};
      td.Width = width;
      td.Height = height;
      td.MipLevels = 1;
      td.ArraySize = 1;
      td.Format = DXGI_FORMAT_R16_FLOAT;
      td.SampleDesc.Count = 1;
      td.Usage = D3D11_USAGE_DEFAULT;
      td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(device->CreateTexture2D(&td, nullptr, _luma[i].put())) ||
          FAILED(device->CreateShaderResourceView(_luma[i].get(), nullptr, _luma_srv[i].put())) ||
          FAILED(device->CreateUnorderedAccessView(_luma[i].get(), nullptr, _luma_uav[i].put()))) {
        return false;
      }
    }

    const std::uint32_t count = ((width + 15) / 16) * ((height + 15) / 16);
    D3D11_BUFFER_DESC fd {};
    fd.ByteWidth = count * 8;
    fd.Usage = D3D11_USAGE_DEFAULT;
    fd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    fd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    fd.StructureByteStride = 8;
    _field = nullptr;
    _field_uav = nullptr;
    _staging = nullptr;
    if (FAILED(device->CreateBuffer(&fd, nullptr, _field.put())) ||
        FAILED(device->CreateUnorderedAccessView(_field.get(), nullptr, _field_uav.put()))) {
      return false;
    }
    D3D11_BUFFER_DESC sd {};
    sd.ByteWidth = count * 8;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(device->CreateBuffer(&sd, nullptr, _staging.put()))) {
      return false;
    }
    D3D11_BUFFER_DESC cd {};
    cd.ByteWidth = sizeof(std::uint32_t) * 8;
    cd.Usage = D3D11_USAGE_DEFAULT;
    cd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    cd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud {};
    ud.Format = DXGI_FORMAT_R32_TYPELESS;
    ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = 8;
    ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    D3D11_BUFFER_DESC csd = cd;
    csd.Usage = D3D11_USAGE_STAGING;
    csd.BindFlags = 0;
    csd.MiscFlags = 0;
    csd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    _stats = nullptr;
    _stats_uav = nullptr;
    _stats_staging = nullptr;
    if (FAILED(device->CreateBuffer(&cd, nullptr, _stats.put())) || FAILED(device->CreateUnorderedAccessView(_stats.get(), &ud, _stats_uav.put())) ||
        FAILED(device->CreateBuffer(&csd, nullptr, _stats_staging.put()))) {
      return false;
    }
    _width = width;
    _height = height;
    _ready = true;
    BOOST_LOG(info) << "Game capture: DLSS motion vectors become encoder hints (" << width << 'x' << height << ')';
    return true;
  }

  void motion_pass_t::reset() {
    _previous_valid = false;
  }

  std::string motion_pass_t::stats() {
    if (!_counts[0]) {
      return {};
    }
    const auto pct = [](std::uint64_t n, std::uint64_t of) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%.1f%%", of ? 100.0 * static_cast<double>(n) / static_cast<double>(of) : 0.0);
      return std::string(buf);
    };
    const auto checked = _counts[3] + _counts[4];
    std::string s = " motion: fields=" + std::to_string(_fields) + " blocks: none " + pct(_counts[1], _counts[0]) + ", DLSS zero " + pct(_counts[2], _counts[0]) +
                    ", vector won " + pct(_counts[3], _counts[0]) + ", zero won " + pct(_counts[4], _counts[0]) + "; of " + std::to_string(checked) +
                    " checked: flip fits better " + pct(_counts[5], checked) + ", half " + pct(_counts[6], checked) + ", double " + pct(_counts[7], checked);
    std::memset(_counts, 0, sizeof(_counts));
    _fields = 0;
    return s;
  }

  std::shared_ptr<const motion_field_t> motion_pass_t::run(ID3D11Device *device, ID3D11DeviceContext *context, const frame_t &frame) {
    if (!frame.motion || !frame.texture || _init_failed) {
      _previous_valid = false;
      return nullptr;
    }
    if (!_ready || _width != frame.width || _height != frame.height) {
      if (!init(device, frame.width, frame.height)) {
        _init_failed = true;
        BOOST_LOG(error) << "Game capture: the motion-vector pass is unavailable; encoding without motion hints";
        return nullptr;
      }
    }

    // The vectors must span the frame (DLSS's output is the frame, maybe at
    // another resolution but of the same shape)
    const double aspect_frame = static_cast<double>(frame.width) / frame.height;
    const double aspect_out = static_cast<double>(frame.motion_out_width) / frame.motion_out_height;
    if (std::abs(aspect_frame / aspect_out - 1.0) > 0.01) {
      if (!_logged_aspect) {
        _logged_aspect = true;
        BOOST_LOG(info) << "Game capture: DLSS output " << frame.motion_out_width << 'x' << frame.motion_out_height
                        << " does not span the " << frame.width << 'x' << frame.height << " frame; no motion hints";
      }
      _previous_valid = false;
      return nullptr;
    }

    if (_color_texture != frame.texture) {
      _color_srv = nullptr;
      _color_texture = nullptr;
      if (FAILED(device->CreateShaderResourceView(frame.texture, nullptr, _color_srv.put()))) {
        _previous_valid = false;
        return nullptr;
      }
      _color_texture = frame.texture;
    }
    if (_motion_texture != frame.motion) {
      _motion_srv = nullptr;
      _motion_texture = nullptr;
      if (FAILED(device->CreateShaderResourceView(frame.motion, nullptr, _motion_srv.put()))) {
        _previous_valid = false;
        return nullptr;
      }
      _motion_texture = frame.motion;
    }

    // Checked against the previous frame only if that was the game's
    // previous frame, with the previous evaluation's vectors
    const bool verify = _previous_valid && frame.frame_id == _previous_frame_id + 1 && frame.motion_id == _previous_motion_id + 1;

    params_t p {};
    p.frame_size[0] = frame.width;
    p.frame_size[1] = frame.height;
    p.blocks[0] = (frame.width + 15) / 16;
    p.blocks[1] = (frame.height + 15) / 16;
    p.texel_per_pixel[0] = static_cast<float>(frame.motion_width) / frame.width;
    p.texel_per_pixel[1] = static_cast<float>(frame.motion_height) / frame.height;
    p.pixel_per_unit[0] = static_cast<float>(frame.width) / frame.motion_width * frame.motion_scale_x;
    p.pixel_per_unit[1] = static_cast<float>(frame.height) / frame.motion_height * frame.motion_scale_y;
    D3D11_TEXTURE2D_DESC color_desc {};
    frame.texture->GetDesc(&color_desc);
    p.linear_light = linear_format(color_desc.Format) ? 1 : 0;
    p.verify = verify ? 1 : 0;
    context->UpdateSubresource(_params.get(), 0, nullptr, &p, 0, 0);

    const int cur = _current;
    const int prev = 1 - cur;
    ID3D11Buffer *cb = _params.get();
    context->CSSetConstantBuffers(0, 1, &cb);

    // This frame's luma
    ID3D11ShaderResourceView *luma_in[1] = {_color_srv.get()};
    context->CSSetShaderResources(0, 1, luma_in);
    ID3D11UnorderedAccessView *luma_out[1] = {_luma_uav[cur].get()};
    context->CSSetUnorderedAccessViews(0, 1, luma_out, nullptr);
    context->CSSetShader(_luma_cs.get(), nullptr, 0);
    context->Dispatch((frame.width + 7) / 8, (frame.height + 7) / 8, 1);
    ID3D11UnorderedAccessView *null_uav[2] = {};
    context->CSSetUnorderedAccessViews(0, 1, null_uav, nullptr);

    // The blocks
    ID3D11ShaderResourceView *block_in[4] = {nullptr, _motion_srv.get(), _luma_srv[prev].get(), _luma_srv[cur].get()};
    context->CSSetShaderResources(0, 4, block_in);
    const UINT zero[4] = {0, 0, 0, 0};
    context->ClearUnorderedAccessViewUint(_stats_uav.get(), zero);
    ID3D11UnorderedAccessView *block_out[3] = {nullptr, _field_uav.get(), _stats_uav.get()};
    context->CSSetUnorderedAccessViews(0, 3, block_out, nullptr);
    context->CSSetShader(_blocks_cs.get(), nullptr, 0);
    context->Dispatch((p.blocks[0] + 7) / 8, (p.blocks[1] + 7) / 8, 1);

    ID3D11ShaderResourceView *null_srv[4] = {};
    context->CSSetShaderResources(0, 4, null_srv);
    ID3D11UnorderedAccessView *null_uav3[3] = {};
    context->CSSetUnorderedAccessViews(0, 3, null_uav3, nullptr);
    context->CSSetShader(nullptr, nullptr, 0);

    _current = prev;
    _previous_valid = true;
    _previous_frame_id = frame.frame_id;
    _previous_motion_id = frame.motion_id;
    if (!verify) {
      return nullptr;  // (the luma is kept for the next frame)
    }

    // The field, read back now: the encoder needs it on the CPU with the
    // frame, and the frame's GPU work is what the encoder waits for anyway
    context->CopyResource(_staging.get(), _field.get());
    context->CopyResource(_stats_staging.get(), _stats.get());
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (SUCCEEDED(context->Map(_stats_staging.get(), 0, D3D11_MAP_READ, 0, &mapped))) {
      const auto *c = static_cast<const std::uint32_t *>(mapped.pData);
      for (int i = 0; i < 8; ++i) {
        _counts[i] += c[i];
      }
      context->Unmap(_stats_staging.get(), 0);
    }
    if (FAILED(context->Map(_staging.get(), 0, D3D11_MAP_READ, 0, &mapped))) {
      return nullptr;
    }
    ++_fields;
    auto field = std::make_shared<motion_field_t>();
    field->frame_id = frame.frame_id;
    field->width = frame.width;
    field->height = frame.height;
    field->cols = p.blocks[0];
    field->rows = p.blocks[1];
    field->vectors.resize(static_cast<std::size_t>(field->cols) * field->rows * 2);
    std::memcpy(field->vectors.data(), mapped.pData, field->vectors.size() * sizeof(std::int32_t));
    context->Unmap(_staging.get(), 0);
    return field;
  }

}  // namespace platf::dxgi::game_capture
