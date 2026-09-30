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
 *     bitstream | 0xFF | escaped MVC2 stream | length (5 bytes) | interval (5 bytes) | 'M' 'V' 'S' '1'
 *
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

#include <cstdint>
#include <vector>

namespace motion_sideband {
  constexpr std::uint32_t SS_FF_MOTION_SIDEBAND = 0x08000000;
  constexpr std::uint32_t ML_FF_MOTION_SIDEBAND = 0x08000000;
  constexpr std::uint8_t VIDEO_PACKET_EXTRA_FLAG_MOTION_SIDEBAND = 0x4;
  constexpr std::uint8_t kGuard = 0xff;
  constexpr std::uint8_t kMagic[4] = {'M', 'V', 'S', '1'};

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
