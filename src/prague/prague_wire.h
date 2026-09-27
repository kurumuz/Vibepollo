/**
 * @file src/prague/prague_wire.h
 * @brief Wire formats and negotiation flags for Prague congestion control
 *        (Vibepollo protocol extension; controller vendored from
 *        L4STeam/udp_prague, Apache 2.0).
 *
 * Mirrors the definitions in the client's moonlight-common-c Video.h.
 * All fields little-endian on the wire.
 */
#pragma once

#include <cstdint>

namespace prague {

  // Negotiation: the host ORs SS_FF_PRAGUE_CC into x-ss-general.featureFlags
  // when prague_cc is enabled in config; the client answers with
  // ML_FF_PRAGUE_CC in x-ml-general.featureFlags. Datagrams carry Prague
  // headers only when both happened. High bit deliberately far from
  // upstream's sequential feature flag allocation.
  constexpr std::uint32_t SS_FF_PRAGUE_CC = 0x40000000;
  constexpr std::uint32_t ML_FF_PRAGUE_CC = 0x40000000;

  // Prepended to EVERY video datagram (FEC parity shards included -- Prague
  // must account for every datagram that costs bandwidth), outside the FEC
  // and encryption envelopes, via the fec_t per-shard prefix region.
  struct data_hdr_t {
    std::int32_t timestamp;  ///< host send time, µs (host clock, wraps)
    std::int32_t echoed_timestamp;  ///< defrosted echo of client's last ACK timestamp
    std::uint32_t seq_nr;  ///< per-datagram monotonic counter
  };

  static_assert(sizeof(data_hdr_t) == 12, "data_hdr_t must be 12 bytes on the wire");

  // Sent client -> host on the video UDP socket. Distinguished from SS_PING
  // by exact size plus magic. Counters are cumulative, so ACK loss is
  // harmless; the next one supersedes it.
  constexpr std::uint32_t ACK_MAGIC = 0x4B434150;  // "PACK"

  struct ack_msg_t {
    std::uint32_t magic;
    char payload[16];  ///< client identifier, same value as SS_PING.payload
    std::int32_t timestamp;  ///< client send time, µs (client clock, wraps)
    std::int32_t echoed_timestamp;  ///< defrosted echo of host's last data timestamp
    std::uint32_t packets_received;  ///< cumulative
    std::uint32_t packets_CE;  ///< cumulative
    std::uint32_t packets_lost;  ///< cumulative
    std::uint32_t error_L4S;  ///< bool: client saw bleached/invalid ECN
  };

  static_assert(sizeof(ack_msg_t) == 44, "ack_msg_t must be 44 bytes on the wire");

  // Per-frame arrival reports for slo-bayes bitrate control, negotiated on top
  // of Prague (same feedback path): SS_FF_FRAME_REPORTS from the host when
  // slo_bayes is enabled, ML_FF_FRAME_REPORTS back.
  constexpr std::uint32_t SS_FF_FRAME_REPORTS = 0x20000000;
  constexpr std::uint32_t ML_FF_FRAME_REPORTS = 0x20000000;

  // Sent client -> host on the video UDP socket once per received frame. The
  // host keeps each frame's send span itself; this carries only the arrival
  // side, on the client's clock (used only as differences, so the clock
  // offset cancels). Distinguished from ack_msg_t by exact size plus magic.
  constexpr std::uint32_t FRAME_REPORT_MAGIC = 0x4D524650;  // "PFRM"

  struct frame_report_t {
    std::uint32_t magic;
    char payload[16];  ///< client identifier, same value as SS_PING.payload
    std::uint32_t frame_index;  ///< host frame index
    std::uint32_t packets;  ///< datagrams received for the frame, parity shards included
    std::uint32_t bytes;  ///< their size on the wire, Prague header included
    std::uint32_t first_arrival_us;  ///< client clock, µs, wraps
    std::uint32_t last_arrival_us;  ///< client clock, µs, wraps
  };

  static_assert(sizeof(frame_report_t) == 40, "frame_report_t must be 40 bytes on the wire");

  // Host -> client control message while slo-bayes drives the bitrate, a few
  // times a second, so the client can show what the controller is doing.
  // Little-endian, after the usual control header.
  constexpr std::uint16_t SS_RATE_STATUS_PTYPE = 0x5520;

  struct rate_status_t {
    std::uint32_t bitrate_kbps;  ///< encoder bitrate currently applied
    std::uint32_t ceiling_kbps;  ///< the client's configured bitrate
    std::uint32_t deadline_us;  ///< frame ready -> fully delivered target
    std::uint32_t capacity_kbps;  ///< link capacity estimate (posterior median)
    std::uint32_t sigma_permille;  ///< its log-space spread x1000 (0.15 = +-16%)
  };

  static_assert(sizeof(rate_status_t) == 20, "rate_status_t must be 20 bytes on the wire");

  // Capacity probe: padding datagrams sent right after a frame's data while
  // slo-bayes is unsure of the link (see slo_bayes::controller_t::
  // probe_bytes). Same size as a video datagram, Prague header first like
  // every datagram; then this marker, then zeros. The client counts it toward
  // the named frame's arrival report and drops it before decryption and the
  // RTP queue. Eight bytes of magic: after the Prague header an encrypted
  // video datagram starts with a random IV, so a 32-bit marker would collide
  // one packet in four billion.
  constexpr std::uint32_t PROBE_MAGIC0 = 0x45425250;  // "PRBE"
  constexpr std::uint32_t PROBE_MAGIC1 = 0x42525056;  // "VPRB"

  struct probe_hdr_t {
    std::uint32_t magic0;
    std::uint32_t magic1;
    std::uint32_t frame_index;  ///< the frame this padding extends
  };

  static_assert(sizeof(probe_hdr_t) == 12, "probe_hdr_t must be 12 bytes on the wire");

}  // namespace prague
