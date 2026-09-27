/**
 * @file tests/pacing_sim/pacing_sim.cpp
 * @brief Closed-loop simulation of the in-game frame limiter and the pacing
 *        lock: the real controller (src/pacing_lock.cpp) and limiter
 *        decisions (limiter_logic.h) against models of a game, the client's
 *        clock and vblanks, and the client's phase measurement
 *        (moonlight PacingLock).
 *
 * Standalone (no Windows, no test framework):
 *   g++ -std=c++20 -O2 -I. tests/pacing_sim/pacing_sim.cpp src/pacing_lock.cpp -o pacing_sim && ./pacing_sim
 * Exits non-zero when a scenario misses its expectations.
 */
#include "src/pacing_lock.h"
#include "src/platform/windows/game_capture/limiter_logic.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

  using steady = pacing_lock::steady_t;

  steady::time_point at(double seconds) {
    return steady::time_point(std::chrono::duration_cast<steady::duration>(std::chrono::duration<double>(seconds)));
  }

  struct game_t {
    double work_ms = 5;  // render time after the frame starts
    double work_jitter_ms = 1;
    double cap_ms = 0;  // the game's own limiter (absolute schedule); 0 = none
    double cap_overshoot_ms = 1;  // its sleep lands up to this late
    double stall_at_s = -1;  // one long frame (loading)
    double stall_ms = 0;
  };

  struct client_t {
    double refresh_hz = 120;
    double skew_ppm = 40;  // client clock rate / host clock rate - 1
    double due_delay_ms = 11;  // mapped host timeline -> due
    double margin_ms = 1;  // PacingLock::MARGIN
    double report_s = 0.25;
    double report_delay_ms = 5;
  };

  struct scenario_t {
    std::string name;
    double stream_fps = 60;
    game_t game;
    client_t client;
    double seconds = 120;
    double measure_from_s = 30;
    std::function<bool(double)> from_game = [](double) {
      return true;
    };
    // expectations (after measure_from_s)
    int max_slips = 0;  // display intervals other than the expected refresh count
    double max_mean_abs_phase_us = 1e9;
    int max_resets = 1000000;
    int expect_divisor = 1;
    bool expect_steering = true;  // steering for most of the measured time
  };

  struct result_t {
    int slips = 0;
    int frames = 0;
    double mean_abs_phase_us = 0;
    int resets = 0;
    int late = 0;
    int divisor = 1;
    double steering_fraction = 0;
    double min_adjust_ppm = 0, max_adjust_ppm = 0;
    std::string last_stats;
  };

  result_t run(const scenario_t &sc, bool verbose, unsigned seed = 12345) {
    pacing_lock::clear();
    pacing_lock::controller_t ctl;
    const double nominal = 1.0 / sc.stream_fps;
    ctl.set_nominal(nominal);
    game_capture::limiter_logic_t logic;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0, 1);

    const double R = 1.0 / sc.client.refresh_hz;  // client seconds
    const double skew = sc.client.skew_ppm * 1e-6;
    const int vblanks_per_frame = std::max(1, static_cast<int>(std::lround(nominal / R)));

    result_t r;
    std::uint64_t waits = 0, late_total = 0;
    double t = 0.1;  // host time
    double release = 0;  // release that started the frame being rendered
    double cap_target = 0;
    double period = nominal;

    // client reporting
    double report_start = -1, ref_err = 0, sum_err = 0;
    int samples = 0;
    std::uint32_t sequence = 0;
    struct pending_t {
      double deliver_at;
      pacing_lock::pacing_feedback_t fb;
    };
    std::vector<pending_t> pending;
    long last_vblank = -1;
    double phase_abs_sum = 0;
    int phase_n = 0;
    double steering_time = 0, measured_time = 0;
    bool stalled = false;
    long last_print = -1;

    while (t < sc.seconds) {
      // --- the game renders one frame, starting at the release ---
      double start = std::max(t, release);
      if (sc.game.cap_ms > 0) {
        const double cap = sc.game.cap_ms * 1e-3;
        cap_target = cap_target == 0 ? start : cap_target + cap;
        if (cap_target < start - cap) {
          cap_target = start;  // far behind: its schedule restarts
        }
        if (start < cap_target) {
          start = cap_target + u01(rng) * sc.game.cap_overshoot_ms * 1e-3;
        }
      }
      double work = (sc.game.work_ms + (u01(rng) - 0.5) * sc.game.work_jitter_ms) * 1e-3;
      if (!stalled && sc.game.stall_at_s >= 0 && start >= sc.game.stall_at_s) {
        work += sc.game.stall_ms * 1e-3;
        stalled = true;
      }
      const double present = start + work;
      const double pts = release != 0 ? release : present;  // what the host sends

      // --- the host: capture at Present, limiter after it ---
      const bool from_game = sc.from_game(present);
      ctl.note_source(from_game, at(present));
      ctl.note_limiter(waits, late_total, static_cast<std::uint32_t>(logic.divisor()),
                       static_cast<std::int32_t>(std::lround(logic.drift() * 1e6)),
                       static_cast<std::uint64_t>(std::llround(logic.game_period() * 1e12)), at(present));
      period = static_cast<double>(ctl.period_ps(at(present))) * 1e-12;
      const auto plan = logic.plan(present, period);
      late_total += plan.late ? 1 : 0;
      r.resets += (plan.reset && present > sc.measure_from_s) ? 1 : 0;
      r.late += (plan.late && present > sc.measure_from_s) ? 1 : 0;
      const double next_release = std::max(plan.release, present);
      logic.released(next_release, next_release - present, plan.late, period, release != 0 ? present - release : -1);
      ++waits;
      release = next_release;
      t = next_release;

      // deliver due reports
      for (auto it = pending.begin(); it != pending.end();) {
        if (it->deliver_at <= present) {
          pacing_lock::submit(it->fb, at(it->deliver_at));
          it = pending.erase(it);
        } else {
          ++it;
        }
      }

      // --- the client (only game frames carry the paced timeline; desktop
      // frames are modelled as the same timeline, which is what makes
      // steering on them wrong) ---
      const double c_pts = pts * (1 + skew);
      const double latch = c_pts + sc.client.due_delay_ms * 1e-3;
      const long vblank = static_cast<long>(std::ceil(latch / R));
      const double waste = vblank * R - latch;
      double err = waste - sc.client.margin_ms * 1e-3;
      err -= std::floor(err / R + 0.5) * R;
      if (present > sc.measure_from_s) {
        ++r.frames;
        if (last_vblank >= 0 && vblank - last_vblank != vblanks_per_frame * logic.divisor()) {
          ++r.slips;
          if (verbose) {
            std::printf("      slip at %.2f s: %ld refreshes (divisor %d, late %d, reset %d)\n", present, vblank - last_vblank, logic.divisor(), plan.late, plan.reset);
          }
        }
        phase_abs_sum += std::abs(err);
        ++phase_n;
      }
      last_vblank = vblank;

      if (report_start < 0) {
        report_start = c_pts;
      }
      if (samples == 0) {
        ref_err = err;
        sum_err = 0;
      }
      double e = err;
      while (e - ref_err > R / 2) {
        e -= R;
      }
      while (e - ref_err < -R / 2) {
        e += R;
      }
      sum_err += e;
      ++samples;
      if (c_pts - report_start >= sc.client.report_s) {
        double mean = sum_err / samples;
        mean -= std::floor(mean / R + 0.5) * R;
        pacing_lock::pacing_feedback_t fb {};
        fb.version = pacing_lock::kFeedbackVersion;
        fb.flags = pacing_lock::kFeedbackValid;
        fb.refresh_period_ps = static_cast<std::uint64_t>(std::llround(R * 1e12));
        fb.skew_ppb = static_cast<std::int32_t>(std::lround(skew * 1e9));
        fb.vblanks_per_frame = static_cast<std::uint32_t>(vblanks_per_frame);
        fb.phase_error_us = static_cast<std::int32_t>(std::lround(mean * 1e6));
        fb.samples = static_cast<std::uint32_t>(samples);
        fb.sequence = ++sequence;
        pending.push_back({present + sc.client.report_delay_ms * 1e-3, fb});
        samples = 0;
        report_start = c_pts;
      }

      if (present > sc.measure_from_s) {
        const double dt = std::max(0.0, next_release - present) + work;
        measured_time += dt;
        if (ctl.mode() == pacing_lock::controller_t::mode_e::steering ||
            ctl.mode() == pacing_lock::controller_t::mode_e::holding_unlockable) {
          steering_time += dt;
        }
        const double adj = (period / (vblanks_per_frame * R / (1 + skew)) - 1) * 1e6;
        r.min_adjust_ppm = std::min(r.min_adjust_ppm, adj);
        r.max_adjust_ppm = std::max(r.max_adjust_ppm, adj);
      }
      if (verbose && static_cast<long>(present / 2) != last_print) {
        last_print = static_cast<long>(present / 2);
        std::printf("    t=%6.1f err=%+6.0fus lag=%5.2fms drift=%+5.0fppm game=%+5.0fppm %s div=%d\n", present, err * 1e6,
                    (start - (pts)) * 1e3, logic.drift() * 1e6, logic.game_period() > 0 ? (logic.game_period() / logic.divisor() / nominal - 1) * 1e6 : 0.0, ctl.stats().c_str(), logic.divisor());
      }
    }
    r.divisor = logic.divisor();
    r.mean_abs_phase_us = phase_n ? phase_abs_sum / phase_n * 1e6 : 0;
    r.steering_fraction = measured_time > 0 ? steering_time / measured_time : 0;
    r.last_stats = ctl.stats();
    return r;
  }

}  // namespace

// Self-capped game at 60 across client clock offsets and seeds, harsher
// jitter: slips against the physical minimum (0 when the client needs a
// slower rate than the cap, one per refresh of drift otherwise)
int sweep() {
  int bad = 0;
  std::printf("skew_ppm  jitter  seeds  slips(min..max)  floor-of-physics  verdict\n");
  for (double jitter : {1.0, 3.0}) {
    for (int skew = -300; skew <= 300; skew += 50) {
      scenario_t sc;
      sc.name = "sweep";
      sc.game.cap_ms = 1000.0 / 60;
      sc.game.work_ms = 6;
      sc.game.work_jitter_ms = jitter;
      sc.game.cap_overshoot_ms = jitter > 1 ? 2 : 1;
      sc.client.skew_ppm = skew;
      sc.seconds = 240;
      sc.measure_from_s = 40;
      // The client needs host frames every 2R/(1+skew); the game delivers every
      // cap. Needing faster than the cap (skew > 0 here) drifts at skew ppm.
      const double measured_s = sc.seconds - sc.measure_from_s;
      const double drift = skew > 0 ? skew * 1e-6 : 0;
      const double physics = drift * measured_s / (1.0 / 120);  // refreshes of drift
      int lo = 1 << 30, hi = 0;
      for (unsigned seed = 1; seed <= 5; ++seed) {
        const auto r = run(sc, false, seed);
        lo = std::min(lo, r.slips);
        hi = std::max(hi, r.slips);
      }
      const bool ok = hi <= std::ceil(physics * 1.5) + 3;
      bad += ok ? 0 : 1;
      std::printf("%+8d  %4.1fms  %5d  %5d..%-5d      %6.1f          %s\n", skew, jitter, 5, lo, hi, physics, ok ? "ok" : "EXCESS");
    }
  }
  std::printf("%d sweep point(s) with excess slips\n", bad);
  return bad;
}

int main(int argc, char **argv) {
  const std::string only = argc > 1 ? argv[1] : "";
  if (only == "sweep") {
    return sweep() ? 1 : 0;
  }
  if (only == "one" && argc >= 5) {
    scenario_t sc;
    sc.name = "one";
    sc.game.cap_ms = 1000.0 / 60;
    sc.game.work_ms = 6;
    sc.game.work_jitter_ms = std::atof(argv[3]);
    sc.game.cap_overshoot_ms = sc.game.work_jitter_ms > 1 ? 2 : 1;
    sc.client.skew_ppm = std::atof(argv[2]);
    sc.seconds = 240;
    sc.measure_from_s = 40;
    const auto r = run(sc, true, static_cast<unsigned>(std::atoi(argv[4])));
    std::printf("slips=%d\n", r.slips);
    return 0;
  }
  const bool verbose = !only.empty();
  std::vector<scenario_t> scenarios;

  {
    scenario_t s;
    s.name = "uncapped game, 60 fps on 120 Hz, +40 ppm";
    s.max_slips = 0;
    s.max_mean_abs_phase_us = 300;
    s.max_resets = 0;
    scenarios.push_back(s);
  }
  {
    scenario_t s;
    s.name = "uncapped game, -80 ppm, 1 ms report delay jitterless";
    s.client.skew_ppm = -80;
    s.max_slips = 0;
    s.max_mean_abs_phase_us = 300;
    s.max_resets = 0;
    scenarios.push_back(s);
  }
  {
    // NieR: its own 60 fps cap, the client wants a hair slower than 60.000
    scenario_t s;
    s.name = "self-capped at 60, client slower (+33 ppm)";
    s.game.cap_ms = 1000.0 / 60;
    s.game.work_ms = 6;
    s.client.skew_ppm = -33;  // client clock slower -> host must run slower
    s.max_slips = 2;
    s.max_mean_abs_phase_us = 1500;
    scenarios.push_back(s);
  }
  {
    // Same game, the client needs faster than it can go: no lock possible,
    // but no runaway either: slips only at the drift rate
    scenario_t s;
    s.name = "self-capped at 60, client faster (-150 ppm)";
    s.game.cap_ms = 1000.0 / 60;
    s.game.work_ms = 6;
    s.client.skew_ppm = 150;
    s.seconds = 200;
    s.max_slips = 6;  // one per ~55 s of 150 ppm drift, give or take
    s.expect_steering = false;
    scenarios.push_back(s);
  }
  {
    scenario_t s;
    s.name = "60 fps cap in a 120 fps stream on 120 Hz";
    s.stream_fps = 120;
    s.game.cap_ms = 1000.0 / 60;
    s.game.work_ms = 6;
    s.client.skew_ppm = -20;
    s.expect_divisor = 2;
    s.max_slips = 4;  // its failed step-down probe costs two, and one catching the phase up
    s.max_mean_abs_phase_us = 1500;
    scenarios.push_back(s);
  }
  {
    scenario_t s;
    s.name = "uncapped, 400 ms loading stall at 40 s";
    s.game.stall_at_s = 40;
    s.game.stall_ms = 400;
    s.measure_from_s = 50;
    s.max_slips = 0;
    s.max_mean_abs_phase_us = 400;
    s.max_resets = 0;
    scenarios.push_back(s);
  }
  {
    scenario_t s;
    s.name = "uncapped, desktop frames 40-50 s";
    s.from_game = [](double t) {
      return t < 40 || t > 50;
    };
    s.measure_from_s = 65;
    s.max_slips = 0;
    s.max_mean_abs_phase_us = 400;
    s.max_resets = 0;
    scenarios.push_back(s);
  }
  {
    scenario_t s;
    s.name = "slow game (95 fps) in a 120 fps stream: paced at 60, no flapping";
    s.stream_fps = 120;
    s.game.work_ms = 1000.0 / 95;
    s.game.work_jitter_ms = 0.5;
    s.max_slips = 2;
    s.expect_divisor = 2;
    scenarios.push_back(s);
  }

  int failures = 0;
  for (const auto &sc : scenarios) {
    if (!only.empty() && sc.name.find(only) == std::string::npos) {
      continue;
    }
    const auto r = run(sc, verbose);
    std::vector<std::string> why;
    if (r.slips > sc.max_slips) {
      why.push_back("slips " + std::to_string(r.slips) + " > " + std::to_string(sc.max_slips));
    }
    if (r.mean_abs_phase_us > sc.max_mean_abs_phase_us) {
      why.push_back("mean |phase| " + std::to_string(static_cast<int>(r.mean_abs_phase_us)) + " us");
    }
    if (r.resets > sc.max_resets) {
      why.push_back("resets " + std::to_string(r.resets));
    }
    if (r.divisor != sc.expect_divisor) {
      why.push_back("divisor " + std::to_string(r.divisor));
    }
    if (sc.expect_steering && r.steering_fraction < 0.9) {
      why.push_back("steering only " + std::to_string(static_cast<int>(r.steering_fraction * 100)) + "%");
    }
    if (r.min_adjust_ppm < -5100 || r.max_adjust_ppm > 5100) {
      why.push_back("period outside the clamp");
    }
    std::printf("%s  %-62s frames=%d slips=%d |phase|=%.0fus late=%d resets=%d divisor=%d steering=%.0f%% adjust=[%.0f,%.0f]ppm\n",
                why.empty() ? "PASS" : "FAIL", sc.name.c_str(), r.frames, r.slips, r.mean_abs_phase_us, r.late, r.resets, r.divisor,
                r.steering_fraction * 100, r.min_adjust_ppm, r.max_adjust_ppm);
    if (!why.empty()) {
      ++failures;
      for (const auto &w : why) {
        std::printf("      %s\n", w.c_str());
      }
      std::printf("      last: %s\n", r.last_stats.c_str());
    }
  }
  std::printf("%d scenario(s) failed\n", failures);
  return failures ? 1 : 0;
}
