/**
 * @file src/platform/windows/game_capture/limiter_logic.h
 * @brief The front-edge frame limiter's timing decisions, free of clocks and
 *        Windows (the hook feeds it QPC readings; any time unit works).
 *
 * Per paced frame, after the game's real Present returned:
 *   plan(now, period)      where the next release goes (the grid), and whether
 *                          this frame arrived late / so late the grid restarts
 *   (the hook waits until the planned release)
 *   released(at, wait, …)  records the frame and adapts the divisor
 *
 * Grid: releases sit on next_release + divisor * period. A frame up to half a
 * frame period late releases at once and keeps the grid (the next interval is
 * then shorter: limited recovery); a longer stall restarts the grid from now.
 *
 * Divisor: a game locked to a sub-multiple of the stream rate (its own 60 fps
 * cap in a 120 fps stream reset the grid on 7499 of 7500 frames) is paced at
 * that multiple of the period, so its frames land on a regular grid. Only a
 * lock earns it: most of the last kHistory frames late, their median interval
 * within kLockTolerance of a whole number of periods above the divisor, and
 * their interquartile spread within kLockSpread of a period. A game that is
 * merely slow or fluctuating (45-55 fps in a 60 fps stream, a heavy scene, a
 * loading hitch) matches none of that and runs at its own rate: halving it
 * to 30 fps for a stumble (Stellar Blade, 30 s at 30 fps) is worse than
 * letting it run.
 *   down: every one of the last frames left at least 0.9 period of wait
 *
 * Slack does not prove a game could go faster: one with its own cap never
 * sleeps while our (longer) grid is in charge, so a 60 fps-capped game with
 * 6 ms of work looks able to run at 120. A step down is therefore a probe:
 * kProbeAbortLate late frames within kProbeFrames of it abort it at once --
 * back to the previous divisor, with the grid put back on its pre-probe phase
 * -- and the next probe waits kProbeBackoffSeconds, doubling per failure up
 * to kProbeBackoffMaxSeconds. A probe that holds resets the backoff.
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
    static constexpr int kProbeAbortLate = 3;
    static constexpr double kProbeBackoffSeconds = 30;
    static constexpr double kProbeBackoffMaxSeconds = 240;
    static constexpr double kLockTolerance = 0.08;
    static constexpr double kLockSpread = 0.1;

    struct plan_t {
      double release;  ///< when to release the game (never more than two frame periods out)
      bool late;  ///< arrived at or after its release point
      bool reset;  ///< ... more than half a frame period after: the grid restarted
    };

    /**
     * @param units_per_second the time unit of every argument (QPC frequency)
     */
    explicit limiter_logic_t(double units_per_second = 1):
        _units_per_second(units_per_second) {
    }

    plan_t plan(double now, double period) {
      const double frame_period = period * _divisor;
      double target = _next_release == 0 ? now : _next_release + frame_period;
      plan_t p {};
      if (now - target > frame_period / 2) {
        target = now;
        p.late = p.reset = true;
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
    bool released(double released_at, double wait, bool late, double period) {
      ++_frame;
      if (_probe_at >= 0 && _frame - _probe_at <= kProbeFrames) {
        // A probe the game cannot follow ends at its first few late frames
        _probe_late += late ? 1 : 0;
        if (_probe_late >= kProbeAbortLate) {
          _divisor = _probe_from;
          // Back onto the old grid: the next plan() adds one frame period, so
          // set the point before the first old-grid point at or after this
          // release -- the next frame then targets that one (earlier points
          // would read as a stall, later ones would skip a usable slot)
          const double frame_period = period * _divisor;
          if (_probe_anchor != 0 && released_at > _probe_anchor) {
            _next_release = _probe_anchor + (std::ceil((released_at - _probe_anchor) / frame_period) - 1) * frame_period;
          }
          _probe_at = -1;
          _no_probe_until = released_at + _backoff * _units_per_second;
          _backoff = std::min(_backoff * 2, kProbeBackoffMaxSeconds);
          forget_history();
          _last_release = released_at;
          return true;
        }
      } else if (_probe_at >= 0) {
        _probe_at = -1;  // held: the game follows the shorter frame period
        _backoff = kProbeBackoffSeconds;
      }

      bool changed = false;
      if (_last_release != 0) {
        _history[_head] = {released_at - _last_release, wait, late};
        _head = (_head + 1) % kHistory;
        _count = std::min(_count + 1, kHistory);
        changed = adapt(released_at, period);
      }
      _last_release = released_at;
      return changed;
    }

    void forget() {
      _next_release = 0;
      _last_release = 0;
      _divisor = 1;
      _frame = 0;
      _probe_at = -1;
      _no_probe_until = 0;
      _backoff = kProbeBackoffSeconds;
      forget_history();
    }

    int divisor() const {
      return _divisor;
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
    }

    bool adapt(double now, double period) {
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
        std::sort(intervals, intervals + kHistory);
        const double ratio = intervals[kHistory / 2] / period;
        const double spread = (intervals[kHistory * 3 / 4] - intervals[kHistory / 4]) / period;
        const int multiple = static_cast<int>(std::lround(ratio));
        if (multiple > _divisor && std::abs(ratio - multiple) < kLockTolerance && spread < kLockSpread) {
          next = std::min(multiple, kMaxDivisor);
        }
      } else if (_divisor > 1 && min_wait > period * 0.9 && now >= _no_probe_until) {
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

    double _units_per_second;
    double _next_release = 0;  // 0 = no grid yet
    double _last_release = 0;
    int _divisor = 1;
    frame_t _history[kHistory] = {};
    int _count = 0;
    int _head = 0;
    int _last_late = 0;
    double _last_min_wait = 0;
    long _frame = 0;
    long _probe_at = -1;  // frame of the pending step down; -1 = none
    int _probe_from = 1;
    int _probe_late = 0;
    double _probe_anchor = 0;  // the grid point the probe started from
    double _no_probe_until = 0;  // time, in the caller's unit
    double _backoff = kProbeBackoffSeconds;  // seconds
  };

}  // namespace game_capture
