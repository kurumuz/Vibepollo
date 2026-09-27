/**
 * @file src/platform/windows/game_capture/limiter_logic.h
 * @brief The front-edge frame limiter's timing decisions, free of clocks and
 *        Windows: the hook feeds it QPC readings, the pacing simulation
 *        (tests/pacing_sim) feeds it virtual time.
 *
 * Per paced frame, after the game's real Present returned:
 *   plan(now, period)     where the next release goes (the grid), and whether
 *                         this frame arrived late / so late the grid restarts
 *   (the hook waits until the planned release)
 *   released(at, wait, …) records the frame and adapts the divisor
 *
 * Grid: releases sit on next_release + divisor * period. A frame up to half a
 * frame period late releases at once and keeps the grid (the next interval is
 * then shorter: limited recovery); a longer stall restarts the grid from now.
 *
 * Divisor: a game that cannot keep up with the period (its own frame cap, or
 * too slow) is paced at a whole multiple of it, so its frames still land on a
 * regular grid instead of restarting it every frame. Decided on lateness and
 * slack, never on the game's work time: a game with its own cap shortens its
 * sleep by exactly as long as we wait, so work time would oscillate.
 *   up:   most of the last kHistory frames were late, and their median
 *         interval needs more periods than the current divisor
 *   down: every one of them left at least 0.9 period of wait (so one period
 *         fewer is still met)
 * The up rule needs an interval above (divisor + 0.2) periods, the down rule
 * one below (divisor - 0.9); between the two nothing changes, so a game near
 * a boundary does not flap.
 *
 * Slack does not prove a game could go faster: one with its own cap never
 * sleeps while our (longer) grid is in charge, so a 60 fps-capped game with
 * 6 ms of work looks able to run at 120. A step down is therefore a probe:
 * kProbeAbortLate late frames within kProbeFrames of it abort it at once:
 * back to the previous divisor, and the grid snapped back onto its pre-probe
 * phase (the probe's start plus whole old frame periods), so a failed probe
 * costs a few frames rather than a phase shift the lock must work off. The
 * next probe waits kProbeBackoffFrames, doubling per failure (up to 16x).
 *
 * Drift: a game with its own cap slower than our grid is in charge even when
 * no frame arrives late -- with little work per frame it just sleeps longer
 * after each release, until the growing gap wraps a whole frame. The time
 * from a release to the game's next Present ("turnaround") shows it: flat
 * while our limiter paces (render time only), rising steadily while the
 * game sleeps on its own schedule. drift() is the slope of turnaround over
 * kDriftFrames frames -- the difference of the medians of the window's two
 * halves, so a stall or a hitch does not read as a trend -- as a fraction of
 * the frame period: how much slower than our grid the game's own schedule
 * runs. It is reported only when it clears 4 standard errors, estimated
 * from the turnaround's own spread (MAD), so render-time noise never reads
 * as a trend. A grid restart clears the window.
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace game_capture {

  class limiter_logic_t {
  public:
    static constexpr int kHistory = 32;
    static constexpr int kMaxDivisor = 4;
    static constexpr long kProbeFrames = 2 * kHistory;
    static constexpr long kProbeBackoffFrames = 3600;  // 30 s at 120 fps, 60 s at 60
    static constexpr int kProbeAbortLate = 3;
    static constexpr int kDriftFrames = 256;
    static constexpr int kDriftEvery = 64;
    static constexpr double kDriftSigmas = 4;

    struct plan_t {
      double release;  ///< when to release the game (never more than two frame periods out)
      bool late;  ///< arrived at or after its release point
      bool reset;  ///< ... more than half a frame period after: the grid restarted
    };

    plan_t plan(double now, double period) {
      const double frame_period = period * _divisor;
      double target = _next_release == 0 ? now : _next_release + frame_period;
      plan_t p {};
      if (now - target > frame_period / 2) {
        target = now;
        p.late = p.reset = true;
        _turn_count = 0;
        _turn_head = 0;
        _turn_since = 0;
      } else if (now >= target) {
        p.late = true;
      }
      _next_release = target;
      p.release = std::min(target, now + 2 * frame_period);
      return p;
    }

    /**
     * @return true when the divisor changed
     */
    bool released(double released_at, double wait, bool late, double period, double turnaround = -1) {
      bool changed = false;
      ++_frame;
      // A probe the game cannot follow ends at its first few late frames
      if (_probe_at >= 0 && _frame - _probe_at <= kProbeFrames) {
        _probe_late += late ? 1 : 0;
        if (_probe_late >= kProbeAbortLate) {
          _divisor = _probe_from;
          const double frame_period = period * _divisor;
          if (_probe_anchor != 0 && _next_release > _probe_anchor) {
            _next_release = _probe_anchor + std::ceil((_next_release - _probe_anchor) / frame_period) * frame_period;
          }
          _probe_at = -1;
          _no_probe_until = _frame + _backoff;
          _backoff = std::min(_backoff * 2, 16 * kProbeBackoffFrames);
          _count = 0;
          _head = 0;
          _last_release = released_at;
          return true;
        }
      } else if (_probe_at >= 0) {
        _probe_at = -1;  // held: the game follows the shorter frame period
        _backoff = kProbeBackoffFrames;
      }
      if (turnaround >= 0) {
        note_turnaround(turnaround, released_at, period);
      }
      if (_last_release != 0) {
        _history[_head] = {released_at - _last_release, wait, late};
        _head = (_head + 1) % kHistory;
        _count = std::min(_count + 1, kHistory);
        changed = adapt(period);
      }
      _last_release = released_at;
      return changed;
    }

    void forget() {
      _turn_count = 0;
      _turn_head = 0;
      _turn_since = 0;
      _drift = 0;
      _game_period = 0;
      _next_release = 0;
      _last_release = 0;
      _divisor = 1;
      _frame = 0;
      _probe_at = -1;
      _no_probe_until = 0;
      _backoff = kProbeBackoffFrames;
      forget_history();
    }

    int divisor() const {
      return _divisor;
    }

    /**
     * @return how much slower than our frame period the game's own schedule
     *         runs (fraction; ~0 while our limiter paces), from the last
     *         complete window; 0 before one
     */
    double drift() const {
      return _drift;
    }

    /**
     * @return the game's own frame period measured over the drift window
     *         (our mean release interval there plus the turnaround slope), in
     *         the caller's time unit, whenever a drift was detected; 0 otherwise
     */
    double game_period() const {
      return _game_period;
    }

    // For the log line when the divisor changes
    int last_late_count() const {
      return _last_late;
    }

    double last_min_wait() const {
      return _last_min_wait;
    }

  private:
    struct frame_t {
      double interval;  // release to release
      double wait;  // our wait before the release
      bool late;
    };

    void forget_history() {
      _count = 0;
      _head = 0;
      // A new frame period: turnaround trends from before say nothing
      _turn_count = 0;
      _turn_head = 0;
      _turn_since = 0;
      _drift = 0;
      _game_period = 0;
    }

    void note_turnaround(double turnaround, double released_at, double period) {
      _turn[_turn_head] = turnaround;
      _released[_turn_head] = released_at;
      _turn_head = (_turn_head + 1) % kDriftFrames;
      _turn_count = std::min(_turn_count + 1, kDriftFrames);
      if (_turn_count < kDriftFrames || ++_turn_since < kDriftEvery) {
        return;
      }
      _turn_since = 0;
      // Median of each half, oldest half first; their centres are n/2 frames apart
      constexpr int half = kDriftFrames / 2;
      double first[half], second[half];
      for (int i = 0; i < half; ++i) {
        first[i] = _turn[(_turn_head + i) % kDriftFrames];
        second[i] = _turn[(_turn_head + half + i) % kDriftFrames];
      }
      std::nth_element(first, first + half / 2, first + half);
      std::nth_element(second, second + half / 2, second + half);
      const double m1 = first[half / 2];
      const double m2 = second[half / 2];
      // Spread of turnaround about its half's median (MAD -> sigma), and the
      // standard error of a difference of two half medians
      double deviations[kDriftFrames];
      for (int i = 0; i < half; ++i) {
        deviations[i] = std::abs(_turn[(_turn_head + i) % kDriftFrames] - m1);
        deviations[half + i] = std::abs(_turn[(_turn_head + half + i) % kDriftFrames] - m2);
      }
      std::nth_element(deviations, deviations + kDriftFrames / 2, deviations + kDriftFrames);
      const double sigma = 1.4826 * deviations[kDriftFrames / 2];
      const double error = std::sqrt(2.0) * 1.2533 * sigma / std::sqrt(static_cast<double>(half));
      const double difference = m2 - m1;
      const bool detected = std::abs(difference) > kDriftSigmas * error;
      _drift = detected ? difference / half / (period * _divisor) : 0;
      // Oldest and newest release in the window: our mean frame period there
      const double span = _released[(_turn_head + kDriftFrames - 1) % kDriftFrames] - _released[_turn_head];
      _game_period = detected && span > 0 ? span / (kDriftFrames - 1) + difference / half : 0;
    }

    bool adapt(double period) {
      if (_count < kHistory) {
        return false;
      }
      int late = 0;
      double min_wait = 1e300;
      double intervals[kHistory];
      for (int i = 0; i < kHistory; ++i) {
        late += _history[i].late ? 1 : 0;
        min_wait = std::min(min_wait, _history[i].wait);
        intervals[i] = _history[i].interval;
      }
      _last_late = late;
      _last_min_wait = min_wait;
      int next = _divisor;
      if (late > kHistory / 2) {
        std::nth_element(intervals, intervals + kHistory / 2, intervals + kHistory);
        const int need = static_cast<int>(std::ceil(intervals[kHistory / 2] / period - 0.2));
        if (need > _divisor) {
          next = std::min(need, kMaxDivisor);
          if (_probe_at >= 0 && _frame - _probe_at <= kProbeFrames + kHistory) {
            // The last step down did not hold: back off before the next
            _no_probe_until = _frame + _backoff;
            _backoff = std::min(_backoff * 2, 16 * kProbeBackoffFrames);
          }
          _probe_at = -1;
        }
      } else if (_divisor > 1 && min_wait > period * 0.9 && _frame >= _no_probe_until) {
        next = _divisor - 1;
        _probe_at = _frame;
        _probe_from = _divisor;
        _probe_late = 0;
        _probe_anchor = _next_release;
      }
      if (next == _divisor) {
        return false;
      }
      _divisor = next;
      forget_history();
      return true;
    }

    double _next_release = 0;  // 0 = no grid yet
    double _last_release = 0;
    int _divisor = 1;
    frame_t _history[kHistory] = {};
    int _count = 0;
    int _head = 0;
    int _last_late = 0;
    double _last_min_wait = 0;
    long _frame = 0;
    long _probe_at = -1;  // frame of the last step down; -1 = none pending
    int _probe_from = 1;
    double _probe_anchor = 0;  // the grid point the probe started from
    int _probe_late = 0;
    long _no_probe_until = 0;
    long _backoff = kProbeBackoffFrames;
    double _turn[kDriftFrames] = {};
    double _released[kDriftFrames] = {};
    double _game_period = 0;
    int _turn_count = 0;
    int _turn_head = 0;
    int _turn_since = 0;
    double _drift = 0;
  };

}  // namespace game_capture
