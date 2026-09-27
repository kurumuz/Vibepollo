/**
 * @file src/rate/slo_bayes.cpp
 * @brief slo-bayes deadline bitrate control; see slo_bayes.h.
 */
#include "slo_bayes.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>

namespace slo_bayes {
  namespace {
    double norm_pdf(double x) {
      return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    }

    double norm_cdf(double x) {
      return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }

    double norm_quantile(double p) {
      double lo = -12, hi = 12;
      for (int i = 0; i < 200; i++) {
        double m = 0.5 * (lo + hi);
        (norm_cdf(m) < p ? lo : hi) = m;
      }
      return 0.5 * (lo + hi);
    }
  }  // namespace

  controller_t::controller_t(int fps, double max_rate_Bps, int64_t deadline_us, options_t options):
      m_opt(options),
      m_fps(fps),
      m_deadline(deadline_us),
      m_max_rate_Bps(max_rate_Bps) {
    m_z_lo = norm_quantile(kPTarget);
    m_gate = norm_quantile(1.0 - kOutlierFrac / 2);
    m_k_jump = (int) std::ceil(std::log(kStepRatePerFrame) / std::log(kOutlierFrac));
    m_q = std::pow(std::log(kDriftPerSec), 2) / fps;
    m_var_cap = std::pow(std::log(kIdleSpread), 2);
    m_theta = m_q / (2.0 * m_var_cap);  // stationary variance q/(2*theta) = var_cap
    m_mu = std::log(max_rate_Bps);  // prior: the ceiling is a plausible link
    m_long_run = m_mu;
    m_var = m_var_cap;
  }

  int64_t controller_t::frame_budget(int64_t now_us) {
    // Process model: capacity drifts, but toward the level exact measurements
    // last established, not on an unbounded random walk. A walk conditioned
    // each frame on "C >= my own lower quantile" (which is all a frame that
    // fits in one service period can say) ratchets upward forever with
    // nothing confirming it; a reverting process does not. Reversion rate is
    // derived so the stationary spread equals the stated idle spread.
    m_mu += m_theta * (m_long_run - m_mu);
    m_var = (1.0 - m_theta) * (1.0 - m_theta) * m_var + m_q;
    double sigma = std::sqrt(m_var);
    if (m_opt.predictive && m_resid_e2 > 0) {
      sigma = std::sqrt(std::max(m_var, m_resid_e2));
    }
    double c_lo = std::exp(m_mu + m_z_lo * sigma);  // B/s the link delivers w.p. 1-p

    // The MAC serves in whole service periods J: a frame that needs n periods
    // lands at owd + wait0 + (n-1)J, wait0 <= J, so the deadline admits
    // n_max = floor((D - owd - dq - J)/J) + 1 periods and a frame of
    // n_max * C * J bytes. As J -> 0 this is the continuous C * (D - owd - dq).
    // The queue a frame finds is our own earlier frames (predicted: a virtual
    // queue in bytes drained at the capacity estimate) or other traffic
    // (measured, net of our own contribution).
    drain_virtual_queue(now_us);
    double dq_self = m_v_bytes * 1e6 / c_lo;
    double deliverable;
    if (m_opt.empirical && m_rate_ring.size() >= kRateMin && m_wait_ring.size() >= 64) {
      deliverable = empirical_budget(m_v_bytes);
    } else if (m_opt.wait_quantile > 0 && m_wait_ring.size() >= 64) {
      // The delay quantile already contains the wait for the first service
      // opportunity, so the rest of the window is serialization
      double wait = quantile_of(m_wait_ring, m_opt.wait_quantile);
      double window = (double) m_deadline - base_owd_us() - std::max(wait, dq_self);
      deliverable = c_lo * std::max(window, 0.0) / 1e6;
    } else {
      double window = (double) m_deadline - base_owd_us() - std::max((double) m_dq, dq_self);
      window = std::max(window, 0.25 * (double) m_deadline);
      double J = (double) std::max<int64_t>(m_last_j, 100);
      double periods = std::floor((window - J) / J) + 1.0;
      deliverable = periods >= 1.0 ? c_lo * periods * J / 1e6 : c_lo * window / 1e6;
    }

    double ceiling = m_max_rate_Bps / m_fps;
    m_at_ceiling = deliverable >= ceiling;
    return (int64_t) std::clamp(deliverable, kFloor * ceiling, ceiling);
  }

  double controller_t::empirical_budget(double self_queue_bytes) const {
    std::vector<double> rates(m_rate_ring.begin(), m_rate_ring.end());
    std::sort(rates.begin(), rates.end());
    // Our own earlier bytes still ahead of this frame, drained at the median rate
    const double dq_self_us = self_queue_bytes / rates[rates.size() / 2];
    const double T = (double) m_deadline - base_owd_us() - dq_self_us;
    const double n_r = (double) rates.size();
    auto miss = [&](double S) {
      double sum = 0;
      for (double w : m_wait_ring) {
        const double left = T - w;
        if (left <= 0) {
          sum += 1;
          continue;
        }
        // Late iff r < S / left
        sum += (double) (std::lower_bound(rates.begin(), rates.end(), S / left) - rates.begin()) / n_r;
      }
      return sum / (double) m_wait_ring.size();
    };
    const double ceiling = m_max_rate_Bps / m_fps;
    double lo = 0, hi = ceiling;
    if (miss(hi) <= kPTarget) {
      return hi;
    }
    for (int i = 0; i < 22; i++) {
      const double mid = 0.5 * (lo + hi);
      (miss(mid) <= kPTarget ? lo : hi) = mid;
    }
    return lo;
  }

  int64_t controller_t::probe_bytes(int64_t now_us, int64_t frame_wire_bytes, int64_t max_bytes) {
    if (m_at_ceiling) {
      return 0;
    }
    // Empirical mode needs a supply of measurable trains, not a narrow
    // posterior: probe while natural frames have not provided one lately
    const bool unsure = m_opt.empirical ? (m_rate_ring.size() < kRateMin || now_us - m_last_informative_us > kProbeFastUs) :
                                          std::sqrt(m_var) > kProbeSigma;
    const int64_t interval = unsure ? kProbeFastUs : kProbeSlowUs;
    if (m_last_probe_us != 0 && now_us - m_last_probe_us < interval) {
      return 0;
    }
    const double J = (double) std::max<int64_t>(m_last_j, 100);
    const double train_us = std::max(kProbePeriods * J, (double) kProbeMinTrainUs);
    const double target = std::exp(m_mu) * kProbeOvershoot * train_us / 1e6;
    const int64_t extra = std::min((int64_t) target - frame_wire_bytes, max_bytes);
    if (extra <= 0) {
      return 0;  // the frame alone is a long enough train
    }
    m_last_probe_us = now_us;
    return extra;
  }

  controller_t::state_t controller_t::state() const {
    const double pred = m_resid_e2 > 0 ? std::sqrt(std::max(m_var, m_resid_e2)) : std::sqrt(m_var);
    double cap = std::exp(m_mu), p10 = 0;
    if (m_opt.empirical && m_rate_ring.size() >= kRateMin) {
      std::deque<double> r = m_rate_ring;
      cap = quantile_of(r, 0.5) * 1e6;
      p10 = quantile_of(r, 0.1) * 1e6;
    }
    return {cap, std::sqrt(m_var), m_noise_k, base_owd_us(), m_v_bytes, pred, encoder_scale(), p10, m_last_j, m_dq, m_at_ceiling};
  }

  double controller_t::quantile_of(const std::deque<double> &ring, double q) const {
    std::vector<double> v(ring.begin(), ring.end());
    size_t k = std::min(v.size() - 1, (size_t) (q * (double) v.size()));
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t) k, v.end());
    return v[k];
  }

  double controller_t::encoder_scale() const {
    if (m_opt.encoder_quantile <= 0 || m_enc_ring.size() < 64) {
      return 1.0;
    }
    return std::min(1.0, 1.0 / quantile_of(m_enc_ring, m_opt.encoder_quantile));
  }

  void controller_t::on_frame_sent(uint32_t frame_index, int64_t first_send_us, int64_t last_send_us, int64_t bytes, bool probe, int64_t frame_bytes, int64_t target_bytes) {
    if (target_bytes > 0 && frame_bytes > 0) {
      m_enc_ring.push_back((double) frame_bytes / (double) target_bytes);
      if (m_enc_ring.size() > kEncRing) {
        m_enc_ring.pop_front();
      }
    }
    // The queue this frame found when it went out, then this frame joins it
    drain_virtual_queue(first_send_us);
    int64_t self_us = (int64_t) (m_v_bytes * 1e6 / std::exp(m_mu));
    m_sent.push_back({frame_index, first_send_us, last_send_us, self_us, bytes, probe});
    if (m_sent.size() > 64) {
      m_sent.pop_front();
    }
    m_v_bytes += (double) bytes;
  }

  void controller_t::on_base_owd(int64_t owd_us, int64_t now_us) {
    m_base_owd.push(now_us, owd_us, kOwdMinWindowUs);
  }

  void controller_t::on_frame_report(uint32_t frame_index, int64_t first_arrival_us, int64_t last_arrival_us, uint32_t packets, uint32_t bytes, int64_t now_us, report_trace_t *trace) {
    while (!m_sent.empty() && (int32_t) (m_sent.front().frame_index - frame_index) < 0) {
      m_sent.pop_front();
    }
    if (m_sent.empty() || m_sent.front().frame_index != frame_index) {
      return;  // no send record (too old, or never sent by us)
    }
    const sent_t sent = m_sent.front();
    m_sent.pop_front();
    if (trace) {
      trace->matched = true;
      trace->first_send_us = sent.first_send_us;
      trace->last_send_us = sent.last_send_us;
      trace->sent_bytes = sent.bytes;
      trace->probe = sent.probe;
    }

    int64_t raw_owd = first_arrival_us - sent.first_send_us;
    m_raw_owd.push(now_us, raw_owd, kOwdMinWindowUs);
    if (packets < 2) {
      return;  // nothing to observe
    }
    // A frame sent as one burst has a send span of a few µs; the simulator
    // dropped spans under 100 µs, which on the real host discarded nearly
    // every frame. Arrival spread against a near-zero send span is the most
    // informative case there is, so it is kept.
    int64_t span_tx = std::max<int64_t>(sent.last_send_us - sent.first_send_us, 0);
    int64_t span_rx = last_arrival_us - first_arrival_us;
    if (span_rx >= kInformativeSpanUs && packets >= kInformativePackets) {
      // Same definition the budget predicts with: first to last arrival
      m_rate_ring.push_back((double) bytes / (double) span_rx);
      if (m_rate_ring.size() > kRateRing) {
        m_rate_ring.pop_front();
      }
      m_last_informative_us = now_us;
    }

    // The MAC serves the queue in aggregates: a frame spanning n service
    // opportunities lands as n clumps, and the first clump's own service time
    // is invisible (it is folded into the first packet's delay). Per frame:
    //   W      = first-packet delay above base = standing queue + wait for
    //            the first service opportunity
    //   span_rx = first clump -> last clump = (n-1) service periods
    // Under "uniform wait over one period plus a standing queue", recent W
    // give queue = min(W) and period J = 2*(median(W) - min(W)), and
    // s_pure = span_rx + J is the full serialization time. W is net of the
    // self-queue predicted for this frame at send time.
    int64_t W = std::max<int64_t>(raw_owd - m_raw_owd.min() - sent.self_queue_us, 0);
    m_w[m_w_at++ % kWRing] = W;
    m_wait_ring.push_back((double) W);
    if (m_wait_ring.size() > kWaitRing) {
      m_wait_ring.pop_front();
    }
    if (trace) {
      trace->w_us = W;
    }
    int64_t have = std::min<int64_t>(m_w_at, kWRing);
    if (have < 8) {
      return;  // no period estimate yet
    }
    int64_t tmp[kWRing];
    std::copy(m_w, m_w + have, tmp);
    std::nth_element(tmp, tmp + have / 2, tmp + have);
    int64_t w_med = tmp[have / 2];
    int64_t w_min = *std::min_element(tmp, tmp + have);
    int64_t J = std::max<int64_t>(2 * (w_med - w_min), 100);
    m_dq = w_min;
    m_last_j = J;

    int64_t s_pure = span_rx + J;
    double y = std::log((double) bytes * 1e6 / (double) s_pure);
    // Sample noise scales with the service-period ambiguity at this frame's
    // serialization time (short frames noisy, long ones sharp); the
    // dimensionless scale k is learned from accepted innovations.
    double rho = std::log(1.0 + (double) J / (double) s_pure);
    double Rs = std::max(kRFloor, m_noise_k * rho * rho);
    m_last_r = Rs;
    m_last_rho = rho;

    // Pacing-limited (link faster than our burst) when the arrival span
    // tracks the send span to within the measured wait/jitter envelope. A
    // percentage tolerance would let arrival jitter promote our own send rate
    // to a capacity sample -- a self-reinforcing underestimate.
    bool censored = span_rx <= span_tx + J;
    if (censored) {
      m_last_obs = "cens";
      soft_truncate_below(y, Rs);
    } else {
      m_last_obs = "exact";
      // Frame-to-frame spread, from every exact sample (rejects included:
      // for a 0.1% deadline the outliers are the point)
      const double e = y - m_mu;
      m_resid_e2 = m_resid_e2 < 0 ? e * e : m_resid_e2 + kResidAlpha * (e * e - m_resid_e2);
      robust_update(y, Rs);
    }
    if (trace) {
      trace->obs = m_last_obs;
    }
  }

  std::string controller_t::trace_state() const {
    char b[240];
    snprintf(b, sizeof(b), "mu=%.1fMb sig=%.3f R=%.4f k=%.2f J=%" PRId64 " owd=%.0f dq=%" PRId64 " V=%.0fKB last=%s held=%zu", std::exp(m_mu) * 8 / 1e6, std::sqrt(m_var), m_last_r, m_noise_k, m_last_j, base_owd_us(), m_dq, m_v_bytes / 1e3, m_last_obs, m_held.size());
    return b;
  }

  double controller_t::capacity_Bps() const {
    return std::exp(m_mu);
  }

  double controller_t::base_owd_us() const {
    return m_base_owd.empty() ? 0.0 : (double) m_base_owd.min();
  }

  void controller_t::drain_virtual_queue(int64_t now_us) {
    if (m_v_last_us != 0 && now_us > m_v_last_us) {
      m_v_bytes = std::max(0.0, m_v_bytes - std::exp(m_mu) * (double) (now_us - m_v_last_us) / 1e6);
    }
    if (now_us > m_v_last_us) {
      m_v_last_us = now_us;
    }
  }

  void controller_t::kalman(double y, double Rs) {
    double e = y - m_mu;
    double S = m_var + Rs;
    double K = m_var / S;
    // The level capacity reverts to: what exact measurements have shown
    m_long_run += kLongRunAlpha * (y - m_long_run);
    // Learn the noise scale: E[e^2] = var + k*rho^2 for accepted samples
    if (m_last_rho > 0) {
      double k_est = std::max(e * e - m_var, 0.0) / (m_last_rho * m_last_rho);
      m_noise_k = std::clamp(m_noise_k + kNoiseAlpha * (k_est - m_noise_k), kNoiseKMin, kNoiseKMax);
    }
    m_mu += K * e;
    m_var = std::max(m_var * (1.0 - K), kVarFloor);
  }

  // Noisy censored observation: the bound C >= exp(a) is itself uncertain by
  // Rs, so the likelihood is Phi((x - a)/sqrt(Rs)) rather than a step.
  // Closed-form moments of N(mu,var) * Phi(.) -- the variance shrinks only as
  // far as the bound's own precision justifies.
  void controller_t::soft_truncate_below(double a, double Rs) {
    double s2 = m_var + Rs;
    double t = (m_mu - a) / std::sqrt(s2);
    if (t > 8) {
      return;  // bound far below the posterior: no information
    }
    double Phi = std::max(norm_cdf(t), 1e-300);
    double lam = norm_pdf(t) / Phi;
    m_mu += m_var / std::sqrt(s2) * lam;
    m_var = std::max(m_var - (m_var * m_var / s2) * lam * (t + lam), kVarFloor);
  }

  void controller_t::robust_update(double y, double Rs) {
    double e = y - m_mu;
    double z = e / std::sqrt(m_var + Rs);
    if (std::fabs(z) <= m_gate) {
      m_held.clear();
      kalman(y, Rs);
      return;
    }
    m_last_obs = "reject";
    // Consecutive same-direction rejects are a regime change: the process
    // model cannot express a step, so widen to the idle prior and let the
    // held samples speak.
    if (!m_held.empty() && ((m_held.back().y - m_mu) > 0) != (e > 0)) {
      m_held.clear();
    }
    m_held.push_back({y, Rs});
    if ((int) m_held.size() >= m_k_jump) {
      m_last_obs = "JUMP";
      m_var = m_var_cap;
      for (const held_t &h : m_held) {
        kalman(h.y, h.Rs);
      }
      m_held.clear();
    }
  }

}  // namespace slo_bayes
