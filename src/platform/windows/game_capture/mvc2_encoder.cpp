/**
 * @file src/platform/windows/game_capture/mvc2_encoder.cpp
 * @brief The MVC2 motion field encoder on the GPU (see mvc2_encoder.h and
 *        game_capture_mvc2_cs.hlsl).
 */
#include "mvc2_encoder.h"

#include "src/logging.h"
#include "src/platform/windows/display.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>

#define GAME_CAPTURE_SHADERS_DIR SUNSHINE_ASSETS_DIR "/shaders/directx"

namespace platf::dxgi {
  blob_t compile_shader(LPCSTR file, LPCSTR entrypoint, LPCSTR shader_model);
}  // namespace platf::dxgi

namespace platf::dxgi::game_capture {

  namespace {
    constexpr std::uint32_t kBlock = 16;  // pixels per vector
    constexpr std::uint32_t kErrorBound = 1;  // quarter pixels per component
    constexpr std::int32_t kInvalid = -32768;
    constexpr std::int32_t kVectorLimit = 8191;  // (so residuals fit 16 bits)
    constexpr int kRounds = 48, kPhaseA = 24;  // greedy rounds; the translation refresh after phase A
    constexpr std::uint32_t kMaxHypotheses = 4096;

    constexpr const char *kEntryPoints[] = {
      "tile_fit_cs",
      "prior_cs",
      "bits_cs",
      "refresh_cs",
      "dedupe_cs",
      "clear_cs",
      "list_cs",
      "finalize_cs",
      "sbits_cs",
      "sdedupe_cs",
      "slist_cs",
      "sgreedy_cs",
      "keep_cs",
      "solve_cs",
      "args_cs",
      "sums_cs",
      "count_cs",
      "select_cs",
      "tile_encode_cs",
      "scan_cs",
      "pack_cs",
    };
    constexpr int kKernelCount = sizeof(kEntryPoints) / sizeof(kEntryPoints[0]);

    // Indirect arguments live in the state buffer at bytes 32-103 and are copied
    // to the argument buffer before use (count 0, select 12, per listed hypothesis 24, bits 36, sums 60)
    constexpr D3D11_BOX kArgsBox {32, 0, 0, 104, 1, 1};

    std::uint32_t header_bytes(std::uint32_t tiles, std::uint32_t palette_words) {
      return 28 + palette_words * 4 + ((tiles * 2 + 3) & ~3u);
    }
  }  // namespace

  struct mvc2_encoder_t::kernels_t {
    winrt::com_ptr<ID3D11ComputeShader> cs[kKernelCount];

    ID3D11ComputeShader *operator[](const char *name) const {
      for (int i = 0; i < kKernelCount; ++i) {
        if (std::strcmp(kEntryPoints[i], name) == 0) {
          return cs[i].get();
        }
      }
      return nullptr;
    }
  };

  mvc2_encoder_t::mvc2_encoder_t() = default;

  mvc2_encoder_t::~mvc2_encoder_t() {
    if (_compiler.joinable()) {
      _compiler.join();
    }
  }

  bool mvc2_encoder_t::start_compile(ID3D11Device *device) {
    if (device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_1) {
      // (the encoder binds 11 UAVs; 11.0 has 8)
      BOOST_LOG(warning) << "Motion sideband: the encode device is below feature level 11.1, sideband off"sv;
      _unsupported = true;
      return false;
    }
    _kernels = std::make_unique<kernels_t>();
    _compile_state = 1;
    winrt::com_ptr<ID3D11Device> dev;
    dev.copy_from(device);
    _compiler = std::thread([this, dev]() {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kKernelCount; ++i) {
        auto blob = compile_shader(GAME_CAPTURE_SHADERS_DIR "/game_capture_mvc2_cs.hlsl", kEntryPoints[i], "cs_5_0");
        if (!blob || FAILED(dev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, _kernels->cs[i].put()))) {
          BOOST_LOG(error) << "Motion sideband: compiling "sv << kEntryPoints[i] << " failed, sideband off"sv;
          _compile_state = 3;
          return;
        }
      }
      BOOST_LOG(info) << "Motion sideband: encoder ready ("sv
                      << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() << " ms to compile)"sv;
      _compile_state = 2;
    });
    return true;
  }

  bool mvc2_encoder_t::ensure_buffers(ID3D11Device *device, std::uint32_t cols, std::uint32_t rows) {
    if (cols == _cols && rows == _rows && _state) {
      return true;
    }
    _cols = 0;
    _rows = 0;
    _state = nullptr;
    const std::uint32_t tiles_x = (cols + 7) / 8, tiles_y = (rows + 7) / 8, tiles = tiles_x * tiles_y;
    const std::uint32_t nhyp = tiles * 4 + 32, nslots = nhyp * 3;
    if (nhyp > kMaxHypotheses || nslots >= 65536) {
      if (!_logged_size) {
        BOOST_LOG(warning) << "Motion sideband: a "sv << cols << 'x' << rows << " field is beyond the encoder's search limits, sideband off"sv;
        _logged_size = true;
      }
      return false;
    }
    _tiles_x = tiles_x;
    _tiles_y = tiles_y;
    _tiles = tiles;
    _nvec = cols * rows;
    _words = (_nvec + 31) / 32;
    _nhyp = nhyp;
    _nslots = nslots;
    _tsize = 1;
    while (_tsize < 2 * _nslots) {
      _tsize <<= 1;
    }
    _nparts = 16;
    _cap = 28 + 32 * 8 * 4 + ((tiles * 2 + 3) & ~3u) + tiles * 68 * 4 + 64;
    const std::uint32_t nlists = 7 * _nhyp + 2 * _tsize + 128;
    const std::uint32_t init[20] = {cols, rows, tiles_x, tiles_y, kErrorBound, tiles, kBlock, _nvec, _words, _nhyp, 3, 12, 16, 0, 0, _tsize, 1, 0, 1, 0};
    std::copy(init, init + 20, _cb);

    auto make = [&](winrt::com_ptr<ID3D11Buffer> &b, UINT size, UINT stride, UINT bind, UINT misc, const void *data = nullptr) {
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = size;
      d.StructureByteStride = stride;
      d.BindFlags = bind;
      d.MiscFlags = misc;
      d.Usage = D3D11_USAGE_DEFAULT;
      D3D11_SUBRESOURCE_DATA s {data, 0, 0};
      b = nullptr;
      return SUCCEEDED(device->CreateBuffer(&d, data ? &s : nullptr, b.put()));
    };
    auto staging = [&](winrt::com_ptr<ID3D11Buffer> &b, UINT size) {
      D3D11_BUFFER_DESC d {};
      d.ByteWidth = size;
      d.Usage = D3D11_USAGE_STAGING;
      d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      b = nullptr;
      return SUCCEEDED(device->CreateBuffer(&d, nullptr, b.put()));
    };
    auto uav = [&](winrt::com_ptr<ID3D11UnorderedAccessView> &v, ID3D11Buffer *b, UINT elements, bool raw) {
      D3D11_UNORDERED_ACCESS_VIEW_DESC d {};
      d.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      d.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
      d.Buffer.NumElements = elements;
      d.Buffer.Flags = raw ? D3D11_BUFFER_UAV_FLAG_RAW : 0;
      v = nullptr;
      return SUCCEEDED(device->CreateUnorderedAccessView(b, &d, v.put()));
    };
    const UINT U = D3D11_BIND_UNORDERED_ACCESS, S = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, R = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    const std::vector<std::uint32_t> zeros(64, 0);
    bool ok = make(_cbuf, 80, 0, D3D11_BIND_CONSTANT_BUFFER, 0, _cb) &&
              make(_field, _nvec * 8, 8, D3D11_BIND_SHADER_RESOURCE, S) &&
              make(_tile, tiles * 64, 64, U, S) &&
              make(_hyp, _nslots * 80, 80, U, S) &&
              make(_bits, (_nslots + 1) * _words * 4, 4, U, S) &&
              make(_count, _nslots * 4, 4, U, S) &&
              make(_pal, 32 * 80, 80, U, S) &&
              make(_prior, 32 * 80, 80, U, S) &&
              make(_state, 256, 0, U, R, zeros.data()) &&
              make(_part, _nhyp * _nparts * 56 * 4, 4, U, S) &&
              make(_offs, (tiles + 2) * 4, 4, U, S) &&
              make(_stream, _cap, 0, U, R) &&
              make(_args, 128, 0, U, D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) &&
              make(_lists, nlists * 4, 4, U, S) &&
              staging(_st_stream, _cap) &&
              staging(_st_state, 256);
    if (ok) {
      D3D11_SHADER_RESOURCE_VIEW_DESC d {};
      d.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      d.Format = DXGI_FORMAT_UNKNOWN;
      d.Buffer.NumElements = _nvec;
      _field_srv = nullptr;
      ok = SUCCEEDED(device->CreateShaderResourceView(_field.get(), &d, _field_srv.put())) &&
           uav(_uavs[0], _tile.get(), tiles, false) &&
           uav(_uavs[1], _hyp.get(), _nslots, false) &&
           uav(_uavs[2], _bits.get(), (_nslots + 1) * _words, false) &&
           uav(_uavs[3], _count.get(), _nslots, false) &&
           uav(_uavs[4], _pal.get(), 32, false) &&
           uav(_uavs[5], _state.get(), 64, true) &&
           uav(_uavs[6], _offs.get(), tiles + 2, false) &&
           uav(_uavs[7], _stream.get(), _cap / 4, true) &&
           uav(_uavs[8], _lists.get(), nlists, false) &&
           uav(_uavs[9], _part.get(), _nhyp * _nparts * 56, false) &&
           uav(_uavs[10], _prior.get(), 32, false);
    }
    if (!ok) {
      BOOST_LOG(error) << "Motion sideband: creating the encoder's buffers failed"sv;
      _state = nullptr;
      return false;
    }
    _cols = cols;
    _rows = rows;
    _rounds_a = kPhaseA;
    _rounds_b = kRounds - kPhaseA;
    return true;
  }

  bool mvc2_encoder_t::submit(ID3D11Device *device, ID3D11DeviceContext *context, std::uint32_t cols, std::uint32_t rows, const std::vector<std::int32_t> &vectors, std::int32_t none) {
    _pending = false;
    if (_unsupported || cols == 0 || rows == 0 || vectors.size() < std::size_t(cols) * rows * 2) {
      return false;
    }
    if (_compile_state == 0 && !start_compile(device)) {
      return false;
    }
    if (_compile_state != 2) {
      if (_compile_state == 3) {
        _unsupported = true;
        if (_compiler.joinable()) {
          _compiler.join();
        }
      }
      return false;
    }
    if (_compiler.joinable()) {
      _compiler.join();
    }
    if (!ensure_buffers(device, cols, rows)) {
      ++_failed;
      return false;
    }
    // The field as the shader reads it: int32 pairs, -32768 in x for no vector
    _upload.resize(std::size_t(_nvec) * 2);
    for (std::size_t k = 0; k < _upload.size(); k += 2) {
      const std::int32_t x = vectors[k], y = vectors[k + 1];
      if (x == none) {
        _upload[k] = kInvalid;
        _upload[k + 1] = 0;
      } else {
        _upload[k] = std::clamp(x, -kVectorLimit, kVectorLimit);
        _upload[k + 1] = std::clamp(y, -kVectorLimit, kVectorLimit);
      }
    }
    context->UpdateSubresource(_field.get(), 0, nullptr, _upload.data(), 0, 0);
    encode(context);
    _pending = true;
    return true;
  }

  void mvc2_encoder_t::encode(ID3D11DeviceContext *ctx) {
    const kernels_t &K = *_kernels;
    auto set_cb = [&](std::uint32_t phase, std::uint32_t mode) {
      _cb[13] = phase;
      _cb[14] = mode;
      ctx->UpdateSubresource(_cbuf.get(), 0, nullptr, _cb, 0, 0);
    };
    auto args = [&] {
      ctx->CopySubresourceRegion(_args.get(), 0, 0, 0, 0, _state.get(), 0, &kArgsBox);
    };
    auto groups = [](std::uint32_t n) {
      return (n + 63) / 64;
    };
    auto run = [&](const char *name, UINT x, UINT y = 1) {
      ctx->CSSetShader(K[name], nullptr, 0);
      ctx->Dispatch(x, y, 1);
    };
    auto run_ind = [&](const char *name, UINT offset) {
      ctx->CSSetShader(K[name], nullptr, 0);
      ctx->DispatchIndirect(_args.get(), offset);
    };

    ID3D11Buffer *cb = _cbuf.get();
    ID3D11ShaderResourceView *srv = _field_srv.get();
    ID3D11UnorderedAccessView *uavs[11];
    for (int i = 0; i < 11; ++i) {
      uavs[i] = _uavs[i].get();
    }
    ctx->CSSetConstantBuffers(0, 1, &cb);
    ctx->CSSetShaderResources(0, 1, &srv);
    ctx->CSSetUnorderedAccessViews(0, 11, uavs, nullptr);

    // Every frame from scratch: no previous palette as hypotheses
    const std::uint32_t zero = 0;
    const D3D11_BOX prior_box {112, 0, 0, 116, 1, 1};
    ctx->UpdateSubresource(_state.get(), 0, &prior_box, &zero, 0, 0);

    const UINT clear_n = groups(std::max(std::max(_words, _tsize), 128u));
    _cb[17] = 0;  // on_sample
    _cb[18] = 1;  // nparts
    _cb[19] = 0;  // fast
    set_cb(0, 0);
    // Hypotheses: 4 per tile, duplicates dropped, the live ones listed
    run("clear_cs", clear_n);
    run("tile_fit_cs", _tiles_x, _tiles_y);
    run("prior_cs", 1);
    run("dedupe_cs", groups(_nhyp));
    run("list_cs", groups(_nhyp));
    args();
    // The shortlist: every seed refit on a 1/16 sample, sample bitsets, merged by signature,
    // a greedy on the sample picks the seeds kept; those listed again
    _cb[17] = 1;
    ctx->UpdateSubresource(_cbuf.get(), 0, nullptr, _cb, 0, 0);
    run_ind("sums_cs", 24);
    _cb[17] = 0;
    set_cb(0, 1);
    run_ind("sbits_cs", 24);
    run("clear_cs", clear_n);
    run("sdedupe_cs", groups(_nslots));
    run("slist_cs", groups(_nslots));
    run("sgreedy_cs", 1);
    run("keep_cs", groups(_nhyp));
    set_cb(0, 3);
    run("list_cs", groups(_nhyp));
    args();
    // Full-field refits (in parts, then solved); slots deduplicated, listed, their bitsets
    _cb[18] = _nparts;
    ctx->UpdateSubresource(_cbuf.get(), 0, nullptr, _cb, 0, 0);
    run("args_cs", 1);
    args();
    run_ind("sums_cs", 60);
    run_ind("solve_cs", 24);
    set_cb(0, 1);
    run("clear_cs", clear_n);
    run("dedupe_cs", groups(_nslots));
    run("list_cs", groups(_nslots));
    run("finalize_cs", 1);
    args();
    run_ind("bits_cs", 36);
    // The greedy palette selection, in rounds sized from the previous frame; halfway,
    // translations from the vectors not fitted yet
    for (int r = 0; r < kRounds; ++r) {
      if ((r < kPhaseA && r >= _rounds_a) || (r >= kPhaseA && r - kPhaseA >= _rounds_b)) {
        continue;
      }
      if (r == kPhaseA) {
        set_cb(2, 2);
        run("clear_cs", clear_n);
        run("refresh_cs", _tiles_x, _tiles_y);
        run("dedupe_cs", groups(2 * _tiles));
        run("list_cs", groups(_nslots));
        run("finalize_cs", 1);
        args();
        run_ind("bits_cs", 36);
      }
      args();
      run_ind("count_cs", 0);
      run_ind("select_cs", 12);
    }
    // Per tile the cheapest plan; the header, palette and tile headers; the payloads
    run("tile_encode_cs", _tiles_x, _tiles_y);
    run("scan_cs", 1);
    run("pack_cs", _tiles_x, _tiles_y);
    ctx->CopyResource(_st_stream.get(), _stream.get());
    ctx->CopyResource(_st_state.get(), _state.get());

    // Leave the pipeline's compute bindings clear for whoever runs next
    ID3D11UnorderedAccessView *null_uavs[11] = {};
    ID3D11ShaderResourceView *null_srv = nullptr;
    ID3D11Buffer *null_cb = nullptr;
    ctx->CSSetUnorderedAccessViews(0, 11, null_uavs, nullptr);
    ctx->CSSetShaderResources(0, 1, &null_srv);
    ctx->CSSetConstantBuffers(0, 1, &null_cb);
    ctx->CSSetShader(nullptr, nullptr, 0);
  }

  std::vector<std::uint8_t> mvc2_encoder_t::collect(ID3D11DeviceContext *context) {
    if (!_pending) {
      return {};
    }
    _pending = false;
    const auto t0 = std::chrono::steady_clock::now();
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(context->Map(_st_state.get(), 0, D3D11_MAP_READ, 0, &m))) {
      ++_failed;
      return {};
    }
    std::uint32_t st[8];
    std::memcpy(st, m.pData, sizeof(st));
    context->Unmap(_st_state.get(), 0);
    const std::uint32_t models = st[0], rounds = st[2], rounds_a = st[4];
    if (FAILED(context->Map(_st_stream.get(), 0, D3D11_MAP_READ, 0, &m))) {
      ++_failed;
      return {};
    }
    const auto *p = static_cast<const std::uint8_t *>(m.pData);
    std::uint32_t hw[5];
    std::memcpy(hw, p, sizeof(hw));
    const std::size_t size = header_bytes(_tiles, hw[4] >> 8 & 0xffff) + std::size_t(hw[3] & 0xffffff) * 4;
    std::vector<std::uint8_t> out;
    if (hw[0] == 0x3243564du && size <= _cap) {
      out.assign(p, p + size);
    } else {
      ++_failed;
    }
    context->Unmap(_st_stream.get(), 0);
    const double wait_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // Next frame's round budgets: this one's rounds per phase, plus two
    if (rounds_a != 0xffffffffu && rounds_a <= rounds) {
      _rounds_a = std::min(kPhaseA, int(rounds_a) + 2);
      _rounds_b = std::min(kRounds - kPhaseA, int(rounds - rounds_a) + 2);
    } else {
      _rounds_a = kPhaseA;
      _rounds_b = kRounds - kPhaseA;
    }
    if (!out.empty()) {
      ++_frames;
      _bytes += out.size();
      _models += models;
      _wait_ms += wait_ms;
      _wait_max_ms = std::max(_wait_max_ms, wait_ms);
    }
    return out;
  }

  std::string mvc2_encoder_t::stats() {
    std::ostringstream s;
    if (_frames) {
      s << "motion sideband: "sv << _frames << " fields, "sv << _bytes / _frames << " B avg, "sv << double(_models) / _frames << " models avg, readback wait "sv
        << _wait_ms / _frames << " ms avg / "sv << _wait_max_ms << " max"sv;
    }
    if (_failed) {
      s << (_frames ? ", "sv : "motion sideband: "sv) << _failed << " failed"sv;
    }
    _frames = _bytes = _models = _failed = 0;
    _wait_ms = _wait_max_ms = 0;
    return s.str();
  }

}  // namespace platf::dxgi::game_capture
