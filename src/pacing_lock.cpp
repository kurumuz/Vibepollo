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

  void submit(const pacing_feedback_t &feedback, steady_t::time_point now) {
    std::lock_guard lg(g_lock);
    // Unsequenced delivery: a report overtaken by a newer one is stale
    if (g_serial != 0 && g_latest.version == feedback.version &&
        static_cast<std::int32_t>(feedback.sequence - g_latest.sequence) <= 0) {
      return;
    }
    g_latest = feedback;
    g_latest_at = now;
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
      _mode = mode_e::off;
    }
  }

  void controller_t::note_source(bool from_game, steady_t::time_point now) {
    if (from_game && !_from_game) {
      _game_since = now;
    }
    _from_game = from_game;
  }

  void controller_t::note_limiter(std::uint64_t waits, std::uint64_t late, std::uint32_t divisor, std::int32_t drift_ppm, std::uint64_t game_period_ps, steady_t::time_point now) {
    if (!_have_counters || waits < _window_waits || late < _window_late || divisor != _window_divisor) {
      if (_have_counters && divisor != _window_divisor) {
        _divisor_changed = true;
        _divisor_since = now;
      }
      _have_counters = true;
      _window_divisor = divisor;
      _window_waits = waits;
      _window_late = late;
      _window_start = now;
      return;
    }
    const auto frames = waits - _window_waits;
    if (frames < kLatenessMinFrames || now - _window_start < kLatenessWindow) {
      return;
    }
    if (_divisor_changed && now - _divisor_since < kDivisorSettle) {
      // Start over once the divisor has settled
      _window_waits = waits;
      _window_late = late;
      _window_start = now;
      return;
    }
    _late_fraction = static_cast<double>(std::min(late - _window_late, frames)) / static_cast<double>(frames);
    const double drift = std::clamp(static_cast<double>(drift_ppm) * 1e-6, -0.01, 0.01);
    // The game's own period, per base period (the hook paces divisor periods per frame)
    const double game_period = divisor >= 1 && game_period_ps > 0 ? static_cast<double>(game_period_ps) * 1e-12 / divisor : 0;
    const bool game_period_usable = _nominal > 0 && std::abs(game_period / _nominal - 1) < kMaxFeedforwardError;
    if (game_period_usable) {
      // Every measurement of the game's own rate refines the estimate
      const bool fresh = _game_rate > 0 && now - _game_rate_at < kGameRateValid;
      _game_rate = fresh ? 0.5 * (_game_rate + game_period) : game_period;
      _game_rate_at = now;
    }
    if (drift > kDriftGamePaced && _late_fraction <= kGamePacedAbove) {
      // Not late, yet the game sleeps longer after each release: its own
      // schedule is slower than our grid. Go to its rate, just above.
      _game_paced = true;
      _ever_game_paced = true;
      _game_paced_at = now;
      if (game_period_usable) {
        _floor = std::min(std::max(_floor, _game_rate * (1 + kDriftMargin)), _nominal * (1 + kMaxFeedforwardError));
      }
    } else if (drift < -kDriftGamePaced && _floor > 0 && game_period_usable && _late_fraction <= kGamePacedAbove) {
      // The game still sleeps after our releases, but less each frame: its
      // own schedule runs faster than our grid by -drift, so the floor can
      // come down to it (an escape from a phase coincidence overshoots)
      // Never below what the display needs, nor below the game's rate as last
      // measured (an estimate taken mid-transition is biased low)
      const double lowest = std::max(_feedforward, now - _game_rate_at < kGameRateValid ? _game_rate * (1 + kDriftMargin) : 0.0);
      _floor = std::max(std::min(_floor, _game_rate * (1 + kDriftMargin)), lowest);
    } else if (_late_fraction > kGamePacedAbove) {
      _game_paced = true;
      _ever_game_paced = true;
      _game_paced_at = now;
      // What we applied is too short to be in charge
      if (_period > 0) {
        _floor = std::min(std::max(_floor, _period * (1 + kFloorStep)), _nominal * (1 + kMaxFeedforwardError));
      }
    } else if (_late_fraction < kLimiterPacedBelow && _game_paced) {
      _game_paced = false;
      _limiter_since = now;
    }
    _window_waits = waits;
    _window_late = late;
    _window_start = now;
  }

  void controller_t::hold(double period, mode_e mode) {
    _period = period;
    _mode = mode;
  }

  std::uint64_t controller_t::period_ps(steady_t::time_point now) {
    if (_nominal <= 0) {
      return 0;
    }
    pacing_feedback_t feedback;
    steady_t::time_point at;
    std::uint64_t serial;
    {
      std::lock_guard lg(g_lock);
      feedback = g_latest;
      at = g_latest_at;
      serial = g_serial;
    }

    if (serial == 0 || now - at > kFeedbackTimeout || feedback.version != kFeedbackVersion || !(feedback.flags & kFeedbackValid)) {
      _integral = 0;
      hold(std::max(_nominal, _floor), mode_e::off);
    } else if (serial != _seen_serial) {
      _seen_serial = serial;
      ++_reports;
      // Everything below comes from the network: bound it before use
      const double refresh = static_cast<double>(feedback.refresh_period_ps) * 1e-12;
      const double skew = static_cast<double>(feedback.skew_ppb) * 1e-9;
      double phase = static_cast<double>(feedback.phase_error_us) * 1e-6;
      const auto vblanks = feedback.vblanks_per_frame;
      const double feedforward = vblanks >= 1 && vblanks <= 8 && std::abs(skew) < 1e-3 ? vblanks * refresh / (1 + skew) : 0;
      // Reports arrive every ~250 ms; a gap (or the first report) counts as one interval
      const double dt = _mode == mode_e::steering ? std::clamp(std::chrono::duration<double>(now - _last_report).count(), 0.0, 0.5) : 0.25;
      _last_report = now;
      _last_phase_us = phase * 1e6;

      if (feedforward <= 0 || std::abs(feedforward / _nominal - 1) > kMaxFeedforwardError || std::abs(phase) > refresh / 2 + 100e-6) {
        ++_rejected;
        _integral = 0;
        hold(std::max(_nominal, _floor), mode_e::off);
      } else {
        // The floor relaxes once the game has left our limiter in charge a while
        if (_floor > 0 && (!_ever_game_paced || now - _game_paced_at > kFloorHold)) {
          const double seconds = std::chrono::duration<double>(now - _floor_decayed_at).count();
          _floor *= 1 - kFloorDecay * std::clamp(seconds, 0.0, 1.0);
          // ...but not below a fresh measurement of the game's own rate
          if (_game_rate > 0 && now - _game_rate_at < kGameRateValid) {
            _floor = std::max(_floor, _game_rate * (1 + kDriftMargin));
          }
          if (_floor < feedforward * (1 - kMaxAdjust)) {
            _floor = 0;  // no longer constraining
          }
        }
        _floor_decayed_at = now;
        _feedforward = feedforward;
        _last_feedforward_ppm = (feedforward / _nominal - 1) * 1e6;
        if (!_from_game || now - _game_since < kSourceSettle) {
          // The reports describe frames the limiter does not produce: hold
          // the frequency, steer nothing
          _last_adjust = _integral;
          hold(feedforward * (1 + _integral), mode_e::holding_desktop);
        } else if (_game_paced || (_ever_game_paced && now - _limiter_since < kLimiterSettle)) {
          // The game paces itself: a shorter period would change nothing,
          // and an integral would only wind up. Hold on the floor (raised by
          // every such window) so our grid walks clear of the game's.
          _integral = 0;
          _last_adjust = 0;
          hold(std::max(feedforward, _floor), mode_e::holding_game_paced);
        } else if (_floor > feedforward * (1 + kUnlockableAbove)) {
          // The game cannot run as fast as the client needs: pace at its
          // rate, just above, and let the phase drift at the difference
          _integral = 0;
          _last_adjust = _floor / feedforward - 1;
          hold(_floor, mode_e::holding_unlockable);
        } else {
          const double floor_adjust = _floor > 0 ? _floor / feedforward - 1 : -kMaxAdjust;
          const double lowest = std::max(-kMaxAdjust, floor_adjust);
          // Late by more than a quarter refresh with the short way (a shorter
          // period) barred by the floor: go the long way round instead. The
          // frames move later until the phase wraps onto its target, one
          // slip once, rather than sitting a refresh from it for good.
          if (phase < -refresh / 4 && phase / kPhaseTimeConstant + _integral < lowest) {
            phase += refresh;
          }
          double integral = std::clamp(_integral + phase * dt / (4 * kPhaseTimeConstant * kPhaseTimeConstant), -kMaxIntegral, kMaxIntegral);
          const double wanted = phase / kPhaseTimeConstant + integral;
          // At the floor the integral must not keep winding downwards
          if (wanted < lowest && integral < _integral) {
            integral = _integral;
          }
          _integral = integral;
          const double adjust = std::clamp(phase / kPhaseTimeConstant + _integral, lowest, kMaxAdjust);
          _last_adjust = adjust;
          hold(feedforward * (1 + adjust), mode_e::steering);
        }
      }
    }
    return static_cast<std::uint64_t>(std::llround(_period * 1e12));
  }

  std::string controller_t::stats() const {
    static constexpr const char *kModes[] = {"off", "holding (desktop frames)", "holding (the game paces itself)", "steering", "holding (the game's cap is slower than the display needs)"};
    char buffer[320];
    std::snprintf(buffer, sizeof(buffer),
                  "pacing lock %s (period %.4f ms, feedforward %+.1f ppm, phase %+.0f us, adjust %+.0f ppm, integral %+.0f ppm, floor %+.0f ppm, paced frames late %.0f%%, reports=%llu rejected=%llu)",
                  kModes[static_cast<int>(_mode)], _period * 1e3, _last_feedforward_ppm, _last_phase_us, _last_adjust * 1e6, _integral * 1e6,
                  _floor > 0 && _nominal > 0 ? (_floor / _nominal - 1) * 1e6 : 0.0, _late_fraction * 100, static_cast<unsigned long long>(_reports), static_cast<unsigned long long>(_rejected));
    return buffer;
  }

}  // namespace pacing_lock
