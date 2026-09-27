/**
 * @file src/platform/windows/game_capture/limiter_logic.h
 * @brief The front-edge frame limiter's timing decision, free of clocks and
 *        Windows (the hook feeds it QPC readings; any time unit works).
 *
 * After the game's real Present returned, plan(now, period) says when to
 * release the game into its next frame. Releases sit on a grid,
 * next_release + period:
 *  - a frame that arrives before its grid point waits for it;
 *  - one up to half a period late is released at once and the grid is kept
 *    (the next interval is then shorter: limited recovery);
 *  - one later than that restarts the grid from now, so a stall never turns
 *    into a burst of catch-up frames.
 * A game that cannot keep up with the period, or caps itself lower, simply
 * runs at its own rate: it is never waited on.
 */
#pragma once

#include <algorithm>

namespace game_capture {

  class limiter_logic_t {
  public:
    struct plan_t {
      double release;  ///< when to release the game (never more than two periods out)
      bool late;  ///< arrived at or after its release point
      bool reset;  ///< ... more than half a period after: the grid restarted
    };

    plan_t plan(double now, double period) {
      double target = _next_release == 0 ? now : _next_release + period;
      plan_t p {};
      if (now - target > period / 2) {
        target = now;
        p.late = p.reset = true;
      } else if (now >= target) {
        p.late = true;
      }
      _next_release = target;
      p.release = std::min(target, now + 2 * period);
      return p;
    }

    void forget() {
      _next_release = 0;
    }

  private:
    double _next_release = 0;  // 0 = no grid yet
  };

}  // namespace game_capture
