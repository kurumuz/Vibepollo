/**
 * @file tests/game_capture/limiter_logic_test.cpp
 * @brief Checks of the in-game frame limiter's release grid (limiter_logic.h)
 *        against a modelled game. Standalone (no Windows, no framework):
 *   g++ -std=c++20 -O2 -I. tests/game_capture/limiter_logic_test.cpp -o limiter_logic_test && ./limiter_logic_test
 */
#include "src/platform/windows/game_capture/limiter_logic.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace {

  int failures = 0;

#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures; \
    } \
  } while (0)

  // A game: `work` seconds from its release to its Present, optionally capped
  // by its own absolute-schedule limiter at `cap` seconds per frame.
  struct game_t {
    double work = 0.006;
    double jitter = 0.001;
    double cap = 0;
  };

  struct run_t {
    int frames = 0;
    int late = 0;
    int resets = 0;
    double mean_interval = 0;  // release to release, over the second half
    double max_wait = 0;
  };

  run_t simulate(const game_t &g, double period, double seconds, double stall_at = -1, double stall = 0) {
    game_capture::limiter_logic_t logic;
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u01(0, 1);
    run_t r;
    double release = 0.1, cap_target = 0, sum = 0;
    int n = 0;
    bool stalled = false;
    while (release < seconds) {
      double start = release;
      if (g.cap > 0) {
        cap_target = cap_target == 0 ? start : cap_target + g.cap;
        if (cap_target < start - g.cap) {
          cap_target = start;
        }
        start = std::max(start, cap_target);
      }
      double work = g.work + (u01(rng) - 0.5) * g.jitter;
      if (!stalled && stall_at >= 0 && start >= stall_at) {
        work += stall;
        stalled = true;
      }
      const double present = start + work;
      const auto plan = logic.plan(present, period);
      r.late += plan.late ? 1 : 0;
      r.resets += plan.reset ? 1 : 0;
      const double next = std::max(plan.release, present);
      r.max_wait = std::max(r.max_wait, next - present);
      if (release > seconds / 2) {
        sum += next - release;
        ++n;
      }
      release = next;
      ++r.frames;
    }
    r.mean_interval = n ? sum / n : 0;
    return r;
  }

}  // namespace

int main() {
  const double p60 = 1.0 / 60, p120 = 1.0 / 120;

  std::printf("uncapped game at 60: paced on the grid\n");
  {
    const auto r = simulate({}, p60, 60);
    CHECK(r.resets == 0);
    CHECK(r.late <= 1);  // the first frame: no grid yet
    CHECK(std::abs(r.mean_interval - p60) < 1e-6);
    CHECK(r.max_wait <= 2 * p60);
  }

  std::printf("game capped at 60 in a 120 fps stream: runs at its own 60, never waited on\n");
  {
    game_t g;
    g.cap = p60;
    const auto r = simulate(g, p120, 60);
    CHECK(std::abs(r.mean_interval - p60) < 50e-6);
    CHECK(r.max_wait < 1e-9);
  }

  std::printf("game at ~50 fps in a 60 fps stream: runs at its own rate\n");
  {
    game_t g;
    g.work = 1.0 / 50;
    g.jitter = 0.004;
    const auto r = simulate(g, p60, 60);
    CHECK(std::abs(r.mean_interval - 1.0 / 50) < 0.001);
    CHECK(r.max_wait < 1e-9);
  }

  std::printf("loading stall: one reset, then back on the grid\n");
  {
    const auto r = simulate({}, p60, 60, 20, 0.4);
    CHECK(r.resets == 1);
    CHECK(std::abs(r.mean_interval - p60) < 1e-6);
  }

  std::printf("slightly late frame: released at once, grid kept\n");
  {
    game_capture::limiter_logic_t logic;
    auto p = logic.plan(1.0, p60);  // first: grid at 1.0
    CHECK(p.release == 1.0);
    p = logic.plan(1.0 + p60 + 0.003, p60);  // 3 ms late
    CHECK(p.late && !p.reset);
    CHECK(p.release == 1.0 + p60);  // already past: release now
    p = logic.plan(1.0 + p60 + 0.009, p60);  // early for the next point
    CHECK(!p.late);
    CHECK(std::abs(p.release - (1.0 + 2 * p60)) < 1e-12);
  }

  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
