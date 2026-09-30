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
 *     bitstream | MVC2 stream | u32 MVC2 length | u32 kMotionSidebandMagic
 *
 * (little-endian), and the client strips the tail before decoding.
 */
#pragma once

#include <cstdint>

namespace motion_sideband {
  constexpr std::uint32_t SS_FF_MOTION_SIDEBAND = 0x08000000;
  constexpr std::uint32_t ML_FF_MOTION_SIDEBAND = 0x08000000;
  constexpr std::uint8_t VIDEO_PACKET_EXTRA_FLAG_MOTION_SIDEBAND = 0x4;
  constexpr std::uint32_t kMagic = 0x3153564du;  // "MVS1"
}  // namespace motion_sideband
