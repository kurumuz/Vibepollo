/**
 * @file src/frame_timing.h
 * @brief Where a frame's time goes on the host, stage by stage: percentiles
 *        of each stage over 10 s, logged as one line. Stages are recorded
 *        from the capture and the video broadcast threads.
 */
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace frame_timing {

  enum stage_e {
    start_to_gpu,  ///< game frame: its start (limiter release) to the GPU finishing it
    gpu_to_pickup,  ///< game frame: GPU done to the host taking it for encoding
    pickup_to_encoded,  ///< taken to its packet leaving the encoder
    encoded_to_send,  ///< packet queued to its first datagram sent (includes the pacer finishing the previous frame)
    send_span,  ///< first to last datagram of the frame
    start_to_sent,  ///< frame timestamp to last datagram
    stage_count
  };

  inline constexpr const char *stage_names[stage_count] = {
    "start->gpu",
    "gpu->pickup",
    "pickup->encoded",
    "encoded->send",
    "send span",
    "start->sent",
  };

  struct state_t {
    std::mutex lock;
    std::array<std::vector<float>, stage_count> samples;
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
  };

  inline state_t &state() {
    static state_t s;
    return s;
  }

  inline void record(stage_e stage, std::chrono::steady_clock::duration d) {
    const float ms = std::chrono::duration<float, std::milli>(d).count();
    if (ms < 0 || ms > 1000) {
      return;  // (clocks disagreeing, or a stall: not a stage time)
    }
    auto &s = state();
    std::lock_guard lg {s.lock};
    auto &v = s.samples[stage];
    if (v.size() < 20000) {
      v.push_back(ms);
    }
  }

  /**
   * @brief Every 10 s: "p50/p95/p99 ms" per stage that has samples, then the
   *        window restarts. Empty when not yet due or nothing was recorded.
   */
  inline std::string take_report_if_due() {
    auto &s = state();
    std::lock_guard lg {s.lock};
    const auto now = std::chrono::steady_clock::now();
    if (now - s.since < std::chrono::seconds(10)) {
      return {};
    }
    s.since = now;
    std::string out;
    for (int i = 0; i < stage_count; ++i) {
      auto &v = s.samples[i];
      if (v.empty()) {
        continue;
      }
      std::sort(v.begin(), v.end());
      auto at = [&](double p) {
        return v[std::min(v.size() - 1, static_cast<std::size_t>(p * v.size()))];
      };
      char part[96];
      std::snprintf(part, sizeof(part), "%s%s %.2f/%.2f/%.2f", out.empty() ? "" : ", ", stage_names[i], at(0.5), at(0.95), at(0.99));
      out += part;
      v.clear();
    }
    return out.empty() ? out : "Frame timing last 10 s (p50/p95/p99 ms): " + out;
  }

}  // namespace frame_timing
