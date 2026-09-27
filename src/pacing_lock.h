/**
 * @file src/pacing_lock.h
 * @brief Locks the game's frame rate to the client's display (Vibepollo
 *        extension). The client reports how its refresh relates to the host
 *        clock and where frames land relative to its vblanks; the host steers
 *        the in-game frame limiter's period so frames arrive at a constant
 *        phase, just before a vblank, with no drops or repeats from clock
 *        drift.
 *
 * Loop: the client sends pacing_feedback_t a few times a second over the
 * control stream (SS_PACING_FEEDBACK_PTYPE). controller_t turns the latest
 * report into a limiter period, a second-order phase-locked loop:
 *
 *   feedforward = vblanks_per_frame * refresh_period / (1 + skew)   (host clock)
 *   period      = feedforward * (1 + phase_error / tau + integral)
 *   integral   += phase_error * dt / (4 tau^2)
 *
 * A positive phase error means frames are ready earlier than they need to be
 * (they wait for their vblank): a slightly longer period moves them later,
 * shaving that wait off the latency. With the phase error e obeying
 * de/dt = -adjust + (frequency error), those gains make the loop critically
 * damped (a double pole at 1 / (2 tau): ~4 s to settle a step); the integral removes what the feedforward
 * gets wrong (clock slews it cannot see), so the phase settles on the margin
 * rather than beside it.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace pacing_lock {

  // Client -> host control message; little-endian, after the control header
  constexpr std::uint16_t SS_PACING_FEEDBACK_PTYPE = 0x5521;
  // Host feature flag (x-ss-general.featureFlags): the host paces the game and
  // accepts pacing feedback
  constexpr std::uint32_t SS_FF_PACING_LOCK = 0x10000000;

  constexpr std::uint32_t kFeedbackVersion = 2;
  constexpr std::uint32_t kFeedbackValid = 0x1;  ///< the client measured a lockable relation

#pragma pack(push, 1)
  struct pacing_feedback_t {
    std::uint32_t version;  ///< kFeedbackVersion
    std::uint32_t flags;  ///< kFeedback*
    std::uint64_t refresh_period_ps;  ///< the client display's refresh period, client clock
    std::int32_t skew_ppb;  ///< client clock rate relative to the host's, minus 1, in 1e-9 (1 host s = 1 + skew client s)
    std::uint32_t vblanks_per_frame;  ///< client refreshes per stream frame (1, 2, 3, ...)
    std::int32_t phase_error_us;  ///< mean (vblank - scheduled present - margin), wrapped to +-half a refresh
    std::uint32_t samples;  ///< frames behind this report
    std::uint32_t sequence;  ///< increments per report; an older one arriving late is dropped
    std::uint32_t reserved;
  };
#pragma pack(pop)

  static_assert(sizeof(pacing_feedback_t) == 40, "pacing_feedback_t must be 40 bytes on the wire");

  using steady_t = std::chrono::steady_clock;

  /**
   * @brief Record a report from the client (control stream thread).
   */
  void submit(const pacing_feedback_t &feedback, steady_t::time_point now = steady_t::now());

  /**
   * @brief Forget reports (session start / end).
   */
  void clear();

  /**
   * Turns the latest report into the limiter period (see the file comment).
   *
   * Steering needs the stream's frames to come from the paced game (not the
   * desktop) and our limiter to be what paces it. A game with its own frame
   * cap at (or below) the stream rate paces itself: its frames arrive at or
   * after our release points, and a shorter period changes nothing -- the
   * loop would only wind up (NieR:Automata: -2100 ppm and falling while the
   * phase kept slipping). The hook's counters show which: most paced frames
   * arriving at or after their release point.
   *
   * The same counters teach a floor: the shortest period known to keep our
   * limiter in charge. Each window in which the game paces itself raises it
   * above the period that was applied (kFloorStep), and the period is held
   * there, so our grid walks clear of the game's own schedule within
   * seconds. (Two grids at the same rate lock in phase: anchored on the
   * game's first Present, ours sits exactly where the game's frames end, and
   * every frame is "late" by its timer's overshoot, handing the stream the
   * game's jitter.) Steering never goes below the floor, and the integral
   * does not wind against it; a phase that needs a shorter period than the
   * floor allows is corrected the long way round (a longer period: the
   * frames move later until the phase wraps onto its target, one slip once).
   * While the game still sleeps after our releases
   * but less each frame (negative drift), its schedule is faster than our
   * grid by that much and the floor comes straight down to it. Otherwise the
   * floor relaxes (kFloorDecay) once the game
   * has not paced itself for kFloorHold: slowly enough that the drift
   * detector (~4 s) catches an undercut long before its lag shows, so a game
   * that got faster, or a floor raised too far, is found without slips.
   *
   * A client needing a faster rate than the game's cap cannot be locked (the
   * floor sits above the feedforward): the period rests on the floor and the
   * phase drifts at the difference, one slip per refresh of drift. Steering
   * there would only speed the drift up (a positive phase lengthens the
   * period further), so the controller holds instead.
   */
  class controller_t {
  public:
    // Phase error decays with this time constant
    static constexpr double kPhaseTimeConstant = 2.0;
    // Bound on the integral (frequency the feedforward misses)
    static constexpr double kMaxIntegral = 0.002;
    // Largest period change the loop applies, either way
    static constexpr double kMaxAdjust = 0.005;
    // A feedforward further than this from the nominal period is not a lock
    // the client can hold (its refresh is not a multiple of the frame rate)
    static constexpr double kMaxFeedforwardError = 0.01;
    // Reports older than this no longer steer: back to the nominal period
    static constexpr auto kFeedbackTimeout = std::chrono::milliseconds(1500);
    // After the stream switches back to the game's frames, reports still
    // describe the desktop's for a while (a report window plus transit)
    static constexpr auto kSourceSettle = std::chrono::milliseconds(1000);
    // Late fraction of paced frames above which the game paces itself, and
    // below which our limiter is back in charge (hysteresis). While our grid
    // is in charge a frame is essentially never late (the game does not
    // sleep and its work fits the period), so sustained lateness at all --
    // a self-capped game that saturated its lag, or one too heavy for the
    // period -- means we are not.
    static constexpr double kGamePacedAbove = 0.05;
    static constexpr double kLimiterPacedBelow = 0.01;
    // A late fraction is measured over at least this many frames and this long
    static constexpr std::uint64_t kLatenessMinFrames = 60;
    static constexpr auto kLatenessWindow = std::chrono::milliseconds(500);
    // Our limiter must have been in charge this long before steering resumes
    static constexpr auto kLimiterSettle = std::chrono::milliseconds(1000);
    // Lateness says nothing about our period this long after a divisor change
    // (the hook's step-down probes run late on purpose)
    static constexpr auto kDivisorSettle = std::chrono::milliseconds(2000);
    // Floor: raised by this each time the game paces itself, relaxed by this
    // per second once it has not for kFloorHold, never above nominal + 1%
    static constexpr double kFloorStep = 300e-6;
    // The game's own schedule running this much slower than our grid (the
    // hook's turnaround drift) also means it paces itself; the floor then goes
    // to its measured rate plus kDriftMargin
    static constexpr double kDriftGamePaced = 60e-6;
    static constexpr double kDriftMargin = 50e-6;
    static constexpr double kFloorDecay = 20e-6;
    static constexpr auto kFloorHold = std::chrono::seconds(10);
    // A floor this far above the feedforward means the lock is out of reach
    static constexpr double kUnlockableAbove = 50e-6;
    // A game rate measured from positive drift bounds the floor from below
    // this long; afterwards the floor may relax past it (the game may have
    // got faster)
    static constexpr auto kGameRateValid = std::chrono::seconds(60);

    enum class mode_e {
      off,  ///< no usable report: nominal period
      holding_desktop,  ///< the stream shows desktop frames: frequency held
      holding_game_paced,  ///< the game paces itself: feedforward held
      steering,  ///< phase-locking
      holding_unlockable,  ///< the game's own cap is slower than the client needs: resting on the floor
    };

    /**
     * @param nominal_period_s the stream's frame period (1 / fps)
     */
    void set_nominal(double nominal_period_s);

    /**
     * @brief Whether the stream's frames currently come from the paced game.
     */
    void note_source(bool from_game, steady_t::time_point now = steady_t::now());

    /**
     * @brief The hook's cumulative limiter counters: paced frames, and how
     *        many of them arrived at or after their release point (late or
     *        reset), and its current periods-per-frame divisor. Counters that
     *        go backwards restart the measurement; so does a divisor change
     *        (the lateness around it says nothing about our period).
     */
    void note_limiter(std::uint64_t waits, std::uint64_t late, std::uint32_t divisor = 1, std::int32_t drift_ppm = 0, std::uint64_t game_period_ps = 0, steady_t::time_point now = steady_t::now());

    /**
     * @brief The counters now come from another game: measure afresh.
     */
    void forget_limiter() {
      _have_counters = false;
    }

    /**
     * @brief The limiter period to apply now, in picoseconds (0 when there is
     *        no nominal period).
     */
    std::uint64_t period_ps(steady_t::time_point now = steady_t::now());

    mode_e mode() const {
      return _mode;
    }

    double last_phase_us() const {
      return _last_phase_us;
    }

    /**
     * @brief One line for the periodic log.
     */
    std::string stats() const;

  private:
    void hold(double period, mode_e mode);

    double _nominal = 0;
    double _period = 0;
    std::uint64_t _seen_serial = 0;
    steady_t::time_point _last_report {};
    double _integral = 0;
    mode_e _mode = mode_e::off;
    bool _from_game = false;
    steady_t::time_point _game_since {};

    // Who paces: our limiter or the game
    bool _have_counters = false;
    std::uint32_t _window_divisor = 1;
    bool _divisor_changed = false;
    steady_t::time_point _divisor_since {};
    std::uint64_t _window_waits = 0;
    std::uint64_t _window_late = 0;
    steady_t::time_point _window_start {};
    double _late_fraction = 0;
    bool _game_paced = false;
    bool _ever_game_paced = false;
    steady_t::time_point _game_paced_at {};
    steady_t::time_point _limiter_since {};
    double _floor = 0;  // seconds; 0 = none
    double _game_rate = 0;  // the game's own frame period, as last measured from positive drift; 0 = unknown
    steady_t::time_point _game_rate_at {};
    double _feedforward = 0;  // from the latest valid report
    steady_t::time_point _floor_decayed_at {};

    double _last_phase_us = 0;
    double _last_adjust = 0;
    double _last_feedforward_ppm = 0;
    std::uint64_t _reports = 0;
    std::uint64_t _rejected = 0;
  };

}  // namespace pacing_lock
