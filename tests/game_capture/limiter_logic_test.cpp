/**
 * @file tests/game_capture/limiter_logic_test.cpp
 * @brief Checks of the in-game frame limiter's decisions (limiter_logic.h)
 *        against a modelled game. Standalone (no Windows, no framework):
 *   g++ -std=c++20 -O2 -I. tests/game_capture/limiter_logic_test.cpp -o limiter_logic_test && ./limiter_logic_test
 */
#include "src/platform/windows/game_capture/limiter_logic.h"

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

  // A game: `work` seconds from its start to its Present, optionally capped by
  // its own absolute-schedule limiter at `cap` seconds per frame.
  struct game_t {
    double work = 0.006;
    double jitter = 0.001;
    double cap = 0;
    double overshoot = 0.001;
  };

  struct run_t {
    int frames = 0;
    int late = 0;
    int resets = 0;
    int divisor_changes = 0;
    int final_divisor = 1;
    double mean_interval = 0;  // release to release, over the second half
  };

  run_t simulate(const game_t &g, double period, double seconds, unsigned seed = 1, double stall_at = -1, double stall = 0) {
    game_capture::limiter_logic_t logic(1.0);  // seconds
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0, 1);
    run_t r;
    double release = 0.1, cap_target = 0;
    bool stalled = false;
    double first_half_end = seconds / 2, sum = 0;
    int n = 0;
    while (release < seconds) {
      double start = release;
      if (g.cap > 0) {
        cap_target = cap_target == 0 ? start : cap_target + g.cap;
        if (cap_target < start - g.cap) {
          cap_target = start;
        }
        if (start < cap_target) {
          start = cap_target + u01(rng) * g.overshoot;
        }
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
      r.divisor_changes += logic.released(next, next - present, plan.late, period) ? 1 : 0;
      if (release > first_half_end) {
        sum += next - release;
        ++n;
      }
      release = next;
      ++r.frames;
    }
    r.final_divisor = logic.divisor();
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
    CHECK(r.final_divisor == 1);
    CHECK(std::abs(r.mean_interval - p60) < 1e-6);
  }

  std::printf("60 fps-capped game in a 120 fps stream: two periods per frame, few resets\n");
  {
    game_t g;
    g.cap = p60;
    const auto r = simulate(g, p120, 120);
    std::printf("  resets=%d late=%d divisor changes=%d mean interval=%.4f ms\n", r.resets, r.late, r.divisor_changes, r.mean_interval * 1e3);
    CHECK(r.final_divisor == 2);
    CHECK(r.resets < 80);  // every frame until a settled history shows the lock (under a second), then only at failed probes
    CHECK(std::abs(r.mean_interval - p60) < 20e-6);
    // Its step-down probes fail and back off: a handful in two minutes, not one per history
    CHECK(r.divisor_changes <= 8);
  }

  std::printf("95 fps game in a 120 fps stream: runs at its own rate (no lock)\n");
  {
    game_t g;
    g.work = 1.0 / 95;
    g.jitter = 0.0005;
    const auto r = simulate(g, p120, 60);
    CHECK(r.final_divisor == 1);
    CHECK(r.divisor_changes == 0);
  }

  std::printf("game fluctuating at 45-55 fps in a 60 fps stream: never halved to 30\n");
  {
    game_t g;
    g.work = 1.0 / 50;
    g.jitter = 0.004;
    const auto r = simulate(g, p60, 60);
    CHECK(r.final_divisor == 1);
    CHECK(r.divisor_changes == 0);
  }

  std::printf("heavy stretch (35 ms frames for 1 s) in a 60 fps stream: never halved\n");
  {
    const auto r = simulate({}, p60, 30, 1, 10, 0);  // warm up
    CHECK(r.final_divisor == 1);
    game_capture::limiter_logic_t logic(1.0);
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u01(0, 1);
    double release = 0.1;
    int changes = 0;
    while (release < 20) {
      const bool heavy = release > 10 && release < 11;
      const double work = heavy ? 0.030 + u01(rng) * 0.010 : 0.006 + u01(rng) * 0.001;
      const double present = release + work;
      const auto plan = logic.plan(present, p60);
      const double next = std::max(plan.release, present);
      changes += logic.released(next, next - present, plan.late, p60) ? 1 : 0;
      release = next;
    }
    CHECK(changes == 0);
    CHECK(logic.divisor() == 1);
  }

  std::printf("game that fits after all (6 ms at 120): back to one period\n");
  {
    const auto r = simulate({}, p120, 30);
    CHECK(r.final_divisor == 1);
    CHECK(r.resets == 0);
  }

  std::printf("loading stall: one reset, then back on the grid\n");
  {
    const auto r = simulate({}, p60, 60, 1, 20, 0.4);
    CHECK(r.resets == 1);
    CHECK(r.final_divisor == 1);
    CHECK(std::abs(r.mean_interval - p60) < 1e-6);
  }

  std::printf("failed probes back off in time and keep the grid's phase\n");
  {
    game_capture::limiter_logic_t logic(1.0);
    // A game capped at 60 in a 120 fps stream, our grid a hair slower than
    // its cap (so ours is in charge and the game shows spare time): it
    // settles at two periods, and every step down fails
    double release = 0.1, cap_target = 0;
    double probes[16];
    int probe_count = 0;
    int resets_after_settling = 0;
    while (release < 400) {
      double start = release;
      cap_target = cap_target == 0 ? start : cap_target + p60;
      if (cap_target < start - p60) {
        cap_target = start;
      }
      start = std::max(start, cap_target);
      const double present = start + 0.006;
      const double period = p120 * 1.001;
      const bool probing = logic.divisor() == 1 && release > 5;  // the probe's own frames run late by design
      const auto plan = logic.plan(present, period);
      resets_after_settling += (plan.reset && release > 5 && !probing) ? 1 : 0;
      const double next = std::max(plan.release, present);
      const int before = logic.divisor();
      if (logic.released(next, next - present, plan.late, period) && logic.divisor() < before && probe_count < 16) {
        probes[probe_count++] = next;
      }
      release = next;
    }
    std::printf("  probes at:");
    for (int i = 0; i < probe_count; ++i) {
      std::printf(" %.0f", probes[i]);
    }
    std::printf(" s; resets after settling: %d\n", resets_after_settling);
    CHECK(logic.divisor() == 2);
    CHECK(probe_count >= 2);
    // Each wait at least doubles until the cap: 30, 60, 120, 240, 240 s
    for (int i = 1; i < probe_count; ++i) {
      const double wanted = std::min(30.0 * (1 << (i - 1)), 240.0);
      CHECK(probes[i] - probes[i - 1] >= wanted - 1);
    }
    // A failed probe puts the grid back on its old phase. The game kept its
    // own schedule meanwhile, so the first frame back may still restart the
    // grid once: at most one restart per probe outside the probe's frames
    CHECK(resets_after_settling <= probe_count);
  }

  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
