/**
 * @file src/motion_sideband_wire.h
 * @brief The motion sideband on the wire: a game frame's motion field (MVC2,
 *        see game_capture/mvc2_encoder.h) appended to the frame's bitstream.
 *
 * Negotiated both ways: the host advertises SS_FF_MOTION_SIDEBAND in
 * x-ss-general.featureFlags, a client that wants it answers with
 * ML_FF_MOTION_SIDEBAND in x-ml-general.featureFlags. Only then do frames carry
 * it, each such frame marked with VIDEO_PACKET_EXTRA_FLAG_MOTION_SIDEBAND in
 * every datagram's NV_VIDEO_PACKET::extraFlags. The frame's payload is then
 *
 *     bitstream | 0xFF | escaped payload | length (5 bytes) | interval (5 bytes) | 'M' 'V' 'S' '2'
 *
 *   payload = varint mvc2 length | MVC2 stream | mask
 *   mask    = u16 cols | u16 rows (little-endian) | varint run lengths
 *
 * - The mask covers the encoded picture in 4x4 cells (cols = ceil(width / 4),
 *   rows = ceil(height / 4); 0 x 0 = none), row by row: a set cell did not
 *   change from the previous frame (the HUD, drawn after DLSS, and anything
 *   else standing still), so a client warping the frame along the field
 *   holds it still whatever its block's vector says. Runs alternate between
 *   clear and set cells, the first a clear run (possibly 0), and cover every
 *   cell. Varints are LEB128 (7 bits a byte, least significant first).
 *   ('M' 'V' 'S' '1', from earlier hosts: the payload is the MVC2 stream alone.)
 * - The MVC2 stream is escaped like an H.264/HEVC NAL payload (a 0x03 after
 *   every two zero bytes that precede a byte <= 3), so it never contains an
 *   Annex B start code; the 0xFF guard keeps the bitstream's last bytes from
 *   forming one with it. The depacketizer, which scans H.264/HEVC frames for
 *   start codes, sees nothing but slice data.
 * - length (of the escaped stream) and interval (microseconds between the
 *   field's two game frames on the host's capture timeline; 0 unknown) are
 *   unsigned values in five 7-bit groups, least significant first, each byte
 *   with its top bit set: no zero bytes.
 * - The client may find zero bytes after the magic (H.264/HEVC frames are
 *   padded to the packet size); the magic ends in a nonzero byte, so it skips
 *   them first.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace motion_sideband {
  constexpr std::uint32_t SS_FF_MOTION_SIDEBAND = 0x08000000;
  constexpr std::uint32_t ML_FF_MOTION_SIDEBAND = 0x08000000;
  constexpr std::uint8_t VIDEO_PACKET_EXTRA_FLAG_MOTION_SIDEBAND = 0x4;

  // Frame provenance, so a client can show what produced each frame. The host
  // advertises SS_FF_FRAME_SOURCE (no answer needed: old clients only bit-test
  // the extraFlags bits they know) and then marks every datagram's
  // NV_VIDEO_PACKET::extraFlags with
  // - GAME_FRAME: captured from the game's own Present (in-game capture),
  //   not the desktop;
  // - ENCODER_HINTS: encoded with the game's motion vectors as motion-search
  //   hints (nvenc_motion_hints).
  constexpr std::uint32_t SS_FF_FRAME_SOURCE = 0x04000000;
  constexpr std::uint8_t VIDEO_PACKET_EXTRA_FLAG_GAME_FRAME = 0x8;
  constexpr std::uint8_t VIDEO_PACKET_EXTRA_FLAG_ENCODER_HINTS = 0x10;
  constexpr std::uint8_t kGuard = 0xff;
  constexpr std::uint8_t kMagic[4] = {'M', 'V', 'S', '2'};
  constexpr std::size_t kMaxMaskBytes = 48 * 1024;  ///< a mask too fragmented to be worth it goes as none

  inline void put_varint(std::vector<std::uint8_t> &out, std::uint64_t v) {
    while (v >= 0x80) {
      out.push_back(static_cast<std::uint8_t>(0x80 | (v & 0x7f)));
      v >>= 7;
    }
    out.push_back(static_cast<std::uint8_t>(v));
  }

  /**
   * @brief Appends a mask section (the format above) for `cols` x `rows`
   *        cells, one bit each in `bits` (row by row, 32 a word, lowest
   *        first); an empty one when there is none or it would exceed
   *        kMaxMaskBytes.
   */
  inline void append_mask(std::vector<std::uint8_t> &out, const std::uint32_t *bits, std::uint32_t cols, std::uint32_t rows) {
    const auto start = out.size();
    const auto header = [&](std::uint32_t c, std::uint32_t r) {
      out.push_back(static_cast<std::uint8_t>(c));
      out.push_back(static_cast<std::uint8_t>(c >> 8));
      out.push_back(static_cast<std::uint8_t>(r));
      out.push_back(static_cast<std::uint8_t>(r >> 8));
    };
    if (!bits || !cols || !rows || cols > 0xffff || rows > 0xffff) {
      header(0, 0);
      return;
    }
    header(cols, rows);
    const std::uint64_t n = static_cast<std::uint64_t>(cols) * rows;
    std::uint64_t i = 0;
    bool set = false;
    while (i < n) {
      // The next cell after i that differs from the run's value, a word at a time
      std::uint64_t j = i;
      while (j < n) {
        std::uint32_t w = bits[j >> 5] ^ (set ? ~0u : 0u);
        w >>= (j & 31);
        if (w) {
          j += static_cast<std::uint64_t>(__builtin_ctz(w));
          break;
        }
        j = (j | 31) + 1;
      }
      j = std::min(j, n);
      put_varint(out, j - i);
      if (out.size() - start > kMaxMaskBytes) {
        out.resize(start);
        header(0, 0);
        return;
      }
      i = j;
      set = !set;
    }
  }

  inline void put_groups(std::vector<std::uint8_t> &out, std::uint32_t v) {
    for (int i = 0; i < 5; ++i) {
      out.push_back(static_cast<std::uint8_t>(0x80 | ((v >> (7 * i)) & 0x7f)));
    }
  }

  /**
   * @brief Appends the sideband (the format above) to a frame's payload.
   */
  inline void append(std::vector<std::uint8_t> &out, const std::vector<std::uint8_t> &mvc2, std::uint32_t interval_us) {
    out.push_back(kGuard);
    const std::size_t start = out.size();
    int zeros = 0;
    for (const std::uint8_t b : mvc2) {
      if (zeros >= 2 && b <= 3) {
        out.push_back(3);
        zeros = 0;
      }
      out.push_back(b);
      zeros = b == 0 ? zeros + 1 : 0;
    }
    put_groups(out, static_cast<std::uint32_t>(out.size() - start));
    put_groups(out, interval_us);
    out.insert(out.end(), kMagic, kMagic + 4);
  }
}  // namespace motion_sideband
