/**
 * @file src/rate/slo_bayes.h
 * @brief slo-bayes: deadline bitrate control from one Gaussian posterior over
 *        log link capacity.
 *
 * Port of the controller auditioned in moonlight-qt's tests/netsim
 * (slo_bayes_legacy.h, "slo-bayes"), algorithm and constants unchanged. The
 * one knob is the deadline D: a frame must be fully delivered D after it is
 * ready. Each frame is sized to what the link delivers within D with 0.1%
 * miss probability; bitrate is the outcome, capped by a ceiling.
 *
 *   exact obs     a frame whose arrival span exceeded its send span was
 *                 serialization-limited: bytes / service_time samples C
 *   censored obs  a frame that arrived as fast as it was sent only says
 *                 C >= that rate (truncate the posterior below it) -- the
 *                 upward probe, which is why frames go out as NIC-rate bursts
 *   robust        innovations outside the gate are held; consecutive
 *                 same-direction rejects are a capacity step, not jitter
 *
 *   budget = n_max * quantile_p(C) * J, the whole MAC service periods J that
 *            fit in D - owd - queue, clamped to [floor, ceiling]
 *
 * Differences from the simulator, all about real clocks:
 *  - Time is passed in; the host's steady clock in µs.
 *  - Frames are keyed by frame index rather than ready time.
 *  - The simulator's one-way delay was true. Here the first-packet delay is
 *    client arrival minus host send, which carries the clock offset between
 *    the machines; it is only ever used relative to its own running minimum
 *    (queue and period estimates), where the offset cancels. The deadline
 *    window needs the TRUE base one-way delay, which comes from half the
 *    minimum RTT (on_base_owd).
 *
 * Not thread-safe; the caller serializes.
 */
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace slo_bayes {

  class controller_t {
  public:
    /**
     * @brief Fixes found on the first real-link trace (2026-09-27, Wi-Fi,
     *        5 ms deadline: 8.9% of frames late against a 0.1% target).
     *        All off reproduces the simulator's controller.
     */
    struct options_t {
      // Size frames by the PREDICTIVE spread of one frame's service rate
      // (posterior + frame-to-frame variation), not the posterior alone. The
      // posterior says how well the average capacity is known and shrinks
      // with data; whether one frame is late depends on how much the rate
      // varies frame to frame (trace: posterior sigma 0.125, per-frame 0.36).
      bool predictive = false;
      // Queue a frame meets: this quantile of recent first-packet delays
      // (0 = the minimum, as before). The delay's tail, not its floor, is
      // what a 0.1% deadline has to absorb (trace: p90 3.3 ms on late frames).
      double wait_quantile = 0;
      // The encoder does not hit its per-frame target (trace: p90 1.37x,
      // p99 2.31x). Learn the size/target ratio and aim the target so this
      // quantile of frames still fits the budget (0 = trust the encoder).
      double encoder_quantile = 0;
      // Size frames from the measured distributions directly: recent
      // first-packet waits W and recent per-frame service rates r (only from
      // trains long enough to measure one), the largest S with
      //   P(W + dq_self + S / r > D - owd) <= p_target
      // over their empirical joint (independent) distribution. Replaces the
      // Gaussian quantile, which on the second trace mistook small frames'
      // MAC-batching noise for rate variation (predictive sigma 0.4-0.6
      // against 0.36 measured; half the deadline left unused).
      bool empirical = false;
    };

    /**
     * @param fps Stream frame rate.
     * @param max_rate_Bps Ceiling (and prior) link rate, bytes per second.
     * @param deadline_us Frame ready -> fully delivered.
     */
    controller_t(int fps, double max_rate_Bps, int64_t deadline_us, options_t options);

    controller_t(int fps, double max_rate_Bps, int64_t deadline_us):
        controller_t(fps, max_rate_Bps, deadline_us, options_t()) {
    }

    /**
     * @brief Factor to apply to the budget before turning it into an encoder
     *        target: 1 / (the encoder_quantile of size/target), at most 1.
     */
    double encoder_scale() const;

    /**
     * @brief A frame went on the wire: its first and last datagram send
     *        times (host µs) and total bytes on the wire, probe padding
     *        included.
     */
    void on_frame_sent(uint32_t frame_index, int64_t first_send_us, int64_t last_send_us, int64_t bytes, bool probe, int64_t frame_bytes = 0, int64_t target_bytes = 0);

    /**
     * @brief Padding bytes to send right after this frame's data, 0 for none.
     *
     * A frame that fits in one or two MAC service periods says almost nothing
     * about capacity (its arrival spread is the period, not the rate), and a
     * controller that is unsure sends exactly such frames: small ones. A
     * probe stretches the frame's burst into a train long enough to span
     * several periods, trailing the frame so its data is never delayed. It
     * is sized past the current belief (a probe that only confirms what is
     * believed teaches nothing about the upside), repeated often while the
     * posterior is wide and rarely once it is narrow, and skipped while the
     * budget sits at the ceiling (nothing to learn that would be used).
     *
     * @param frame_wire_bytes The frame's own bytes on the wire.
     * @param max_bytes What the sender can emit before the next frame is due.
     */
    int64_t probe_bytes(int64_t now_us, int64_t frame_wire_bytes, int64_t max_bytes);

    /**
     * @brief What one report did, for the per-frame trace.
     */
    struct report_trace_t {
      bool matched = false;  ///< a send record existed for the frame
      int64_t first_send_us = 0;
      int64_t last_send_us = 0;
      int64_t sent_bytes = 0;
      bool probe = false;
      int64_t w_us = 0;  ///< first-packet delay above base, net of self-queue
      const char *obs = "none";  ///< exact / cens / reject / JUMP / none
    };

    /**
     * @brief True one-way base delay sample (half an RTT), µs.
     */
    void on_base_owd(int64_t owd_us, int64_t now_us);

    /**
     * @brief The client's arrival report for a frame: first and last datagram
     *        arrival on the CLIENT clock (unwrapped, µs), datagrams and bytes
     *        received.
     */
    void on_frame_report(uint32_t frame_index, int64_t first_arrival_us, int64_t last_arrival_us, uint32_t packets, uint32_t bytes, int64_t now_us, report_trace_t *trace = nullptr);

    /**
     * @brief Bytes on the wire the next frame may use.
     */
    int64_t frame_budget(int64_t now_us);

    /**
     * @brief One line of controller state for logs.
     */
    std::string trace_state() const;

    /**
     * @brief Link capacity estimate (posterior median), bytes per second.
     */
    double capacity_Bps() const;

    int64_t deadline_us() const {
      return m_deadline;
    }

    struct state_t {
      double capacity_Bps, sigma, noise_k, base_owd_us, self_queue_bytes, predictive_sigma, encoder_scale;
      double rate_p10_Bps;  ///< empirical mode: slow-frame service rate (0 = not measured yet)
      int64_t period_us, queue_us;
      bool at_ceiling;
    };

    state_t state() const;

  private:
    struct held_t {
      double y, Rs;
    };

    struct sent_t {
      uint32_t frame_index;
      int64_t first_send_us;
      int64_t last_send_us;
      int64_t self_queue_us;  ///< our own queue this frame found, predicted
      int64_t bytes;
      bool probe;
    };

    // Time-windowed running minimum
    struct min_window_t {
      struct s_t {
        int64_t t, v;
      };

      std::deque<s_t> w;

      void push(int64_t t, int64_t v, int64_t window_us) {
        while (!w.empty() && w.back().v >= v) {
          w.pop_back();
        }
        w.push_back({t, v});
        while (!w.empty() && w.front().t < t - window_us) {
          w.pop_front();
        }
      }

      bool empty() const {
        return w.empty();
      }

      int64_t min() const {
        return w.front().v;
      }
    };

    double base_owd_us() const;
    void drain_virtual_queue(int64_t now_us);
    void kalman(double y, double Rs);
    void soft_truncate_below(double a, double Rs);
    void robust_update(double y, double Rs);

    // Policy
    static constexpr double kPTarget = 0.001;
    static constexpr double kFloor = 0.20;
    // Stated physical assumptions about the link
    static constexpr double kDriftPerSec = 2.0;  // capacity drifts by at most ~2x per second
    static constexpr double kIdleSpread = 1.5;  // after silence, capacity is known to within 1.5x
    static constexpr double kOutlierFrac = 0.02;  // fraction of service samples that are jitter, not capacity
    static constexpr double kStepRatePerFrame = 1.0 / 240;  // a capacity step every ~2 s
    static constexpr int64_t kOwdMinWindowUs = 10 * 1000 * 1000;  // base delay stable over 10 s
    // Numerical guards
    static constexpr double kVarFloor = 1e-4;
    static constexpr double kRFloor = 0.0024;  // (5%)^2: below this a rate sample is not more precise
    // Recent first-chunk delays for the queue/period estimate: 16 frames =
    // 133 ms at 120 fps, 1/16 quantile resolution
    static constexpr int kWRing = 16;
    static constexpr double kNoiseAlpha = 0.05, kNoiseKMin = 1.0 / 12, kNoiseKMax = 4.0;
    static constexpr double kLongRunAlpha = 1.0 / 1200;  // ~10 s at 120 fps
    // Probing: while the posterior is wider than +-15%, probe 4x a second;
    // once narrower, every 2 s to follow drift. A train spans at least 4
    // measured service periods (2 ms floor) at 1.5x the believed capacity.
    static constexpr double kProbeSigma = 0.15;
    static constexpr int64_t kProbeFastUs = 250000, kProbeSlowUs = 2000000;
    static constexpr int64_t kProbeMinTrainUs = 2000;
    static constexpr double kProbePeriods = 4.0, kProbeOvershoot = 1.5;

    double quantile_of(const std::deque<double> &ring, double q) const;

    options_t m_opt;
    double m_resid_e2 = -1;  // EWMA of squared log-rate innovations, all exact samples
    std::deque<double> m_wait_ring;  // recent first-packet delays (options.wait_quantile)
    std::deque<double> m_enc_ring;  // recent encoder size/target ratios (options.encoder_quantile)
    static constexpr double kResidAlpha = 0.02;
    static constexpr size_t kWaitRing = 512, kEncRing = 512;
    // Empirical mode: service rates from trains of >= 1 ms and >= 20
    // datagrams (a shorter one mostly measures the MAC batching)
    std::deque<double> m_rate_ring;  // bytes per µs
    int64_t m_last_informative_us = 0;
    static constexpr size_t kRateRing = 256, kRateMin = 32;
    static constexpr int64_t kInformativeSpanUs = 1000;
    static constexpr uint32_t kInformativePackets = 20;
    double empirical_budget(double self_queue_bytes) const;

    int m_fps;
    int64_t m_deadline;
    double m_max_rate_Bps;

    double m_z_lo, m_gate, m_q, m_var_cap, m_theta, m_long_run;
    int m_k_jump;
    double m_mu, m_var;
    std::vector<held_t> m_held;

    int64_t m_w[kWRing] = {};
    int64_t m_w_at = 0;
    int64_t m_dq = 0;
    int64_t m_last_j = 0;

    int64_t m_last_probe_us = 0;
    bool m_at_ceiling = false;  // last budget was clamped at the ceiling

    double m_v_bytes = 0;  // our bytes not yet served, per the capacity estimate
    int64_t m_v_last_us = 0;
    std::deque<sent_t> m_sent;  // recent frames awaiting their reports
    double m_last_r = 0, m_last_rho = 0;
    double m_noise_k = 1.0;  // starts at "one full period of ambiguity"; learned
    const char *m_last_obs = "-";

    min_window_t m_raw_owd;  // client arrival - host send: offset-contaminated
    min_window_t m_base_owd;  // true base one-way delay (half the RTT)
  };

}  // namespace slo_bayes
