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
 * report into a limiter period:
 *
 *   feedforward = vblanks_per_frame * refresh_period / (1 + skew)   (host clock)
 *   period      = feedforward * (1 + phase_error / kPhaseTimeConstant)
 *
 * A positive phase error means frames are ready earlier than they need to be
 * (they wait for their vblank): a slightly longer period moves them later,
 * shaving that wait off the latency. Proportional only: the feedforward
 * carries the frequency, so a residual frequency error of e leaves a phase
 * offset of e * kPhaseTimeConstant (20 ppm -> 40 us).
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

  constexpr std::uint32_t kFeedbackVersion = 1;
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
  };
#pragma pack(pop)

  static_assert(sizeof(pacing_feedback_t) == 32, "pacing_feedback_t must be 32 bytes on the wire");

  /**
   * @brief Record a report from the client (control stream thread).
   */
  void submit(const pacing_feedback_t &feedback);

  /**
   * @brief Forget reports (session start / end).
   */
  void clear();

  class controller_t {
  public:
    // Phase error decays with this time constant
    static constexpr double kPhaseTimeConstant = 2.0;
    // Largest period change the loop applies, either way
    static constexpr double kMaxAdjust = 0.005;
    // A feedforward further than this from the nominal period is not a lock
    // the client can hold (its refresh is not a multiple of the frame rate)
    static constexpr double kMaxFeedforwardError = 0.01;
    // Reports older than this no longer steer: back to the nominal period
    static constexpr auto kFeedbackTimeout = std::chrono::milliseconds(1500);

    /**
     * @param nominal_period_s the stream's frame period (1 / fps)
     */
    void set_nominal(double nominal_period_s);

    /**
     * @brief The limiter period to apply now, in picoseconds (0 when there is
     *        no nominal period).
     */
    std::uint64_t period_ps();

    /**
     * @brief One line for the periodic log.
     */
    std::string stats() const;

  private:
    double _nominal = 0;
    double _period = 0;
    std::uint64_t _seen_serial = 0;
    bool _locked = false;
    double _last_phase_us = 0;
    double _last_adjust = 0;
    double _last_feedforward_ppm = 0;
    std::uint64_t _reports = 0;
    std::uint64_t _rejected = 0;
  };

}  // namespace pacing_lock
