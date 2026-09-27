/**
 * @file src/pacing_lock.cpp
 * @brief See pacing_lock.h.
 */
#include "pacing_lock.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>

namespace pacing_lock {

  namespace {
    std::mutex g_lock;
    pacing_feedback_t g_latest {};
    std::chrono::steady_clock::time_point g_latest_at {};
    std::uint64_t g_serial = 0;  // bumped per report; 0 = none
  }  // namespace

  void submit(const pacing_feedback_t &feedback) {
    std::lock_guard lg(g_lock);
    g_latest = feedback;
    g_latest_at = std::chrono::steady_clock::now();
    ++g_serial;
  }

  void clear() {
    std::lock_guard lg(g_lock);
    g_latest = {};
    g_latest_at = {};
    ++g_serial;
  }

  void controller_t::set_nominal(double nominal_period_s) {
    if (nominal_period_s != _nominal) {
      _nominal = nominal_period_s;
      _period = nominal_period_s;
      _integral = 0;
      _locked = false;
    }
  }

  std::uint64_t controller_t::period_ps() {
    if (_nominal <= 0) {
      return 0;
    }
    pacing_feedback_t feedback;
    std::chrono::steady_clock::time_point at;
    std::uint64_t serial;
    {
      std::lock_guard lg(g_lock);
      feedback = g_latest;
      at = g_latest_at;
      serial = g_serial;
    }

    const auto now = std::chrono::steady_clock::now();
    if (serial == 0 || now - at > kFeedbackTimeout || feedback.version != kFeedbackVersion || !(feedback.flags & kFeedbackValid)) {
      _period = _nominal;
      _integral = 0;
      _locked = false;
    } else if (serial != _seen_serial) {
      _seen_serial = serial;
      ++_reports;
      // Everything below comes from the network: bound it before use
      const double refresh = static_cast<double>(feedback.refresh_period_ps) * 1e-12;
      const double skew = static_cast<double>(feedback.skew_ppb) * 1e-9;
      const double phase = static_cast<double>(feedback.phase_error_us) * 1e-6;
      const auto vblanks = feedback.vblanks_per_frame;
      const double feedforward = vblanks >= 1 && vblanks <= 8 && std::abs(skew) < 1e-3 ? vblanks * refresh / (1 + skew) : 0;
      if (feedforward <= 0 || std::abs(feedforward / _nominal - 1) > kMaxFeedforwardError || std::abs(phase) > refresh) {
        ++_rejected;
        _period = _nominal;
        _integral = 0;
        _locked = false;
      } else {
        // Reports arrive every ~250 ms; a gap (or the first report) counts as one interval
        double dt = _locked ? std::chrono::duration<double>(now - _last_report).count() : 0.25;
        dt = std::clamp(dt, 0.0, 0.5);
        _last_report = now;
        _integral = std::clamp(_integral + phase * dt / (4 * kPhaseTimeConstant * kPhaseTimeConstant), -kMaxIntegral, kMaxIntegral);
        const double adjust = std::clamp(phase / kPhaseTimeConstant + _integral, -kMaxAdjust, kMaxAdjust);
        _period = feedforward * (1 + adjust);
        _locked = true;
        _last_phase_us = phase * 1e6;
        _last_adjust = adjust;
        _last_feedforward_ppm = (feedforward / _nominal - 1) * 1e6;
      }
    }
    return static_cast<std::uint64_t>(std::llround(_period * 1e12));
  }

  std::string controller_t::stats() const {
    char buffer[256];
    if (!_locked) {
      std::snprintf(buffer, sizeof(buffer), "pacing lock off (period %.4f ms, reports=%llu rejected=%llu)", _period * 1e3, static_cast<unsigned long long>(_reports), static_cast<unsigned long long>(_rejected));
    } else {
      std::snprintf(buffer, sizeof(buffer), "pacing lock on (period %.4f ms, feedforward %+.1f ppm, phase %+.0f us, adjust %+.0f ppm, integral %+.0f ppm, reports=%llu)", _period * 1e3, _last_feedforward_ppm, _last_phase_us, _last_adjust * 1e6, _integral * 1e6, static_cast<unsigned long long>(_reports));
    }
    return buffer;
  }

}  // namespace pacing_lock
