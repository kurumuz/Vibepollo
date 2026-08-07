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

}  // namespace prague
