## Why

Most systems are seen only in part and through noise: a pendulum through the horizontal position of its bob, an
outbreak through reported cases, a climate through sparse stations, a neuron through its spikes. Estimating the hidden
state of such a system from its observations is filtering, smoothing, and data assimilation, a field of its own in the
references (Särkkä 2013, Cheng et al. 2303.10462). The library can simulate these systems and fit their parameters, but
it cannot track their states, forecast from the latest estimate, or give the likelihood of a stochastic system seen
through noise, which particle methods give and which the inference of stochastic epidemic and ecological models needs.

`add-learning-foundation` adds `ObservationSpec`, which says what a sensor sees and with what noise, and observed runs.
The observation spec is already the measurement model of a filter and the system already holds its process noise, so
state estimation needs one experiment and no new concept.

## What Changes

- New experiment `assimilate(system, data, *, method='ukf', ...)` in `skcomplex.core.experiments`, which estimates the
  states of a system from one observed run, reading the `ObservationSpec` of the data as the measurement model and the
  noise laws of the system as the process noise. Methods `'kalman'`, `'ekf'`, `'ukf'`, `'enkf'`, and `'particle'`,
  with `smoother=True` for the Rauch-Tung-Striebel smoothers of the Gaussian filters and the backward simulation
  smoother of the particle filter. `augmented` names parameters appended to the state, for joint state and parameter
  estimation. It returns a frozen `AssimilationResult` with the filtered or smoothed trajectory, covariances or
  ensembles, innovations, the log-likelihood of the run, the estimates of augmented parameters, and `system`, restarted
  from the last estimate so that `simulate` forecasts from it.
- `calibrate` gains `method='pmmh'`, particle marginal Metropolis-Hastings, whose likelihood is that of the particle
  filter, so stochastic systems seen through noise, jump laws included, are calibrated by their exact posterior.
- `forecast_skill` gains probabilistic scores of ensemble forecasts: the continuous ranked probability score and the
  coverage of central intervals.
- `PendulumSystem` gains `noise`, the spectral density of a white noise on its angular acceleration, the noisy pendulum
  of Särkkä (Example 3.7) that every filter of the book tracks.
- Dependencies: filterpy 2.0.0 and scoringrules 0.12.0 become runtime dependencies.

## Capabilities

### New Capabilities

- `state-estimation`: filters, smoothers, and joint state and parameter estimation of any system from observed runs.

### Modified Capabilities

- `experiments`: calibration by particle marginal Metropolis-Hastings.
- `trajectory-analysis`: probabilistic scores of ensemble forecasts.
- `model-library`: the noisy pendulum.

## Impact

- Core: a new module `core/experiments/_assimilation.py`, the method `'pmmh'` in `_calibration.py`, and
  `core/analysis/_forecast_skill.py`.
- Systems: `physics/mechanics/_mechanics.py` gains the noise of the pendulum.
- Dependencies: `filterpy>=2.0.0` (2.0.0, released 2026-10-26) and `scoringrules>=0.11.0` (0.12.0, released
  2026-11-08, and 0.11.0 of 2026-06-06 as the lower bound).
- Docs: an assimilation section in the calibrate page of the user guide and gallery examples of the noisy pendulum
  tracked by every filter and of Lorenz-96 assimilated by the ensemble Kalman filter.
- Depends on `add-learning-foundation` for `ObservationSpec`, observed runs, and `forecast_skill`. The probabilistic
  scores also read the replicates of `identify` that `add-learning-bridges` adds, but do not need them.
