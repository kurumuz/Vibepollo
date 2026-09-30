/**
 * @file src/platform/windows/game_capture/mvc2_encoder.h
 * @brief MVC2: a game frame's motion field (one quarter-pixel vector per 16x16
 *        block of the encoded picture), compressed on the GPU for the motion
 *        sideband (motion_sideband_wire.h).
 *
 * The stream (little-endian; every vector decodable on its own):
 *
 *   stream  = header (28 bytes) | palette | tile headers (u16 each, padded to 4 bytes) | tile payloads (32-bit words)
 *   header  = u32 'MVC2' | u16 cols | u16 rows | u16 tiles_x | u16 tiles_y | u32: payload words (0-23), E (24-31)
 *             | u32: models K (0-7), palette words (8-23), block size B (24-31)
 *             | u32: bit m set = model m is affine | u32: bit m set = model m is a homography
 *   palette = per model: translation 1 word (x int16 | y int16 << 16, quarter pixels);
 *             affine 6 words int32 ax, bxc, bxr, ay, byc, byr: m.x = (ax + bxc * c + bxr * r + 2048) >> 12
 *             (c, r the vector's column and row in the field);
 *             homography 8 words int32 p0-p7 (the previous position of the block centre
 *             (u, w) = (c B + B/2, r B + B/2) is H (u, w, 1), H = [1 + p0/2^30, p1/2^30, p2/2^16;
 *             p3/2^30, 1 + p4/2^30, p5/2^16; p6/2^40, p7/2^40, 1]): in int64, D = 2^40 + p6 u + p7 w,
 *             nx = (p0 u + p1 w) 2^10 + p2 2^24 - u (p6 u + p7 w), m.x = floor((8 nx + D) / (2 D)),
 *             ny alike with p3, p4, p5 and w
 *   tiles   = 8x8 vectors each, in raster order. Tile header (u16): bits 0-6 payload words,
 *             7-9 mode, 10 mask present, 11-15 model index (mode 3)
 *   payload = [mask: 2 words, bit k = vector (k % 8, k / 8) valid] then by mode:
 *     1 local constant: word cx | cy << 16 (int16 each); [widths word, residuals]
 *     2 local affine:   word cx | cy; word gxu, gxw, gyu, gyw (int8 each); [widths word, residuals]
 *                       (prediction cx + ((gxu * (2i - 7) + gxw * (2j - 7)) >> 4))
 *     3 palette model:  [widths word, residuals]
 *     4 two models:     word a (0-4) | b (5-9) | wx (10-14) | wy (15-19); 2 words selector
 *                       (bit k set: model b); residuals
 *     5 four models:    word m0..m3 (5 bits each) | wx (20-24) | wy (25-29); 4 words labels
 *                       (2 bits per vector k at bit 2k); residuals
 *   widths word: wx (0-4) | wy (5-9); modes 1-3 without it (the payload ends before it) have
 *   zero residuals. Residuals: for each valid vector in order zx (wx bits), zy (wy bits), LSB
 *   first; vector = prediction + unzigzag(z) * (2E + 1). A tile without a mask has every
 *   vector inside the field valid.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <d3d11.h>
#include <winrt/base.h>

namespace platf::dxgi::game_capture {

  class mvc2_encoder_t {
  public:
    ~mvc2_encoder_t();

    /**
     * @brief Queues the encode of a field on the context and returns without
     *        waiting; collect() takes the stream. `vectors` holds (x, y) per
     *        block, row by row, in quarter pixels, x == `none` for no vector.
     *        Returns false (nothing queued) while the shaders still compile on
     *        their thread, when the device cannot run them, or on failure.
     */
    bool submit(ID3D11Device *device, ID3D11DeviceContext *context, std::uint32_t cols, std::uint32_t rows, const std::vector<std::int32_t> &vectors, std::int32_t none);

    /**
     * @brief The stream of the field submitted last (waits for the GPU to
     *        finish it), or empty when nothing is pending.
     */
    std::vector<std::uint8_t> collect(ID3D11DeviceContext *context);

    /**
     * @brief Frames, sizes and waits since the last call (for the periodic
     *        log line), and resets them.
     */
    std::string stats();

  private:
    struct kernels_t;

    bool start_compile(ID3D11Device *device);
    bool ensure_buffers(ID3D11Device *device, std::uint32_t cols, std::uint32_t rows);
    void encode(ID3D11DeviceContext *context);

    // Shaders: compiled once, on their own thread
    std::thread _compiler;
    std::atomic<int> _compile_state {0};  // 0 not started, 1 compiling, 2 ready, 3 failed
    std::unique_ptr<kernels_t> _kernels;
    bool _unsupported = false;
    bool _logged_size = false;

    // Buffers for one field size
    std::uint32_t _cols = 0, _rows = 0, _tiles_x = 0, _tiles_y = 0, _tiles = 0, _nvec = 0, _words = 0, _nhyp = 0, _nslots = 0;
    std::uint32_t _tsize = 0, _nparts = 0, _cap = 0;
    std::uint32_t _cb[20] = {};
    winrt::com_ptr<ID3D11Buffer> _cbuf, _field, _tile, _hyp, _bits, _count, _pal, _prior, _state, _part, _offs, _stream, _args, _lists;
    winrt::com_ptr<ID3D11Buffer> _st_stream, _st_state;
    winrt::com_ptr<ID3D11ShaderResourceView> _field_srv;
    winrt::com_ptr<ID3D11UnorderedAccessView> _uavs[11];
    std::vector<std::int32_t> _upload;

    // Round budgets from the previous frame (rounds used + 2 per phase)
    int _rounds_a = 24, _rounds_b = 24;
    bool _pending = false;

    // Stats
    std::uint64_t _frames = 0, _bytes = 0, _models = 0, _failed = 0;
    double _wait_ms = 0, _wait_max_ms = 0;
  };

}  // namespace platf::dxgi::game_capture
