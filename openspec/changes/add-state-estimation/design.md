## Context

- `add-learning-foundation` adds `ObservationSpec(paths, noise, scale)`, the paths or observable readouts a sensor sees
  and the noise of each, and `ObservedRunSpec` and `ObservedRunsSpec`, runs with their own times, missing values,
  inputs, and conditions. `forecast_skill` scores deterministic forecasts by lead time.
- Systems hold their process noise as laws: SDE terms, noise of maps, and jump laws. `simulate` runs any of them on its
  backend, the JAX backend traces deterministic and SDE steps, and the runtime advances a system step by step.
- `calibrate` (`core/experiments/_calibration.py`) has the methods `local`, `global`, `abc_smc`, `mcmc`, and
  `variational`, with priors, and its likelihoods need every random draw observed or the system deterministic.
- `Lorenz96System` is in `physics/climate/_low_order.py`. `PendulumSystem` is in `physics/mechanics/_mechanics.py`.
- The gap report of the learning drafts (2026-11-10) found filtering, smoothing, ensemble and particle methods, joint
  state and parameter estimation, and particle likelihoods missing, and the user approved this change the same day.

## Goals / Non-Goals

**Goals:**
- The Kalman, extended, unscented, ensemble Kalman, and bootstrap particle filters, and smoothers, on any system whose
  observations are described by an `ObservationSpec`.
- Joint state and parameter estimation by augmented states.
- The likelihood of a stochastic system seen through noise, and a calibration on it.
- Probabilistic scores of the ensemble forecasts that filters and replicated fits give.

**Non-Goals:**
- 3D-Var and 4D-Var. Strong constraint 4D-Var is the fit of initial states by `identify(initial=...)` in the
  foundation, and weak constraint 4D-Var can follow as an option of it.
- Learning model error or observation operators from assimilated states (Cheng et al. 2303.10462, sect. III and IV).
  It composes `assimilate` with `identify` and can be a gallery example once both exist.
- Localisation of ensemble filters for large grids, iterated filtering, and expectation maximisation of noise
  parameters.
- GPU filters and JAX filtering libraries, rejected below.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `assimilate`, `AssimilationResult` | 2, an experiment over any system that reads its compiled step and the `ObservationSpec` of the data by type | `skcomplex.core.experiments` | climate and weather (Lorenz-96, station data), epidemiology (nowcasting from reported cases), neuroscience (hidden membrane states), engineering (tracking, sensor fusion), ecology (populations from counts), pharmacology (concentrations of individuals) |
| `calibrate(method='pmmh')` | 2, an option of an experiment over any system | `core/experiments/_calibration.py` | epidemiology (stochastic outbreaks), ecology (stochastic populations from counts), finance (stochastic volatility), systems biology (stochastic gene expression) |
| Probabilistic scores of `forecast_skill` | 2, an option of an analysis that reads series by their tags | `core/analysis/_forecast_skill.py` | climate and weather (ensemble forecasts), epidemiology (forecast hubs), ecology, finance, and every replicated fit |
| `noise` of `PendulumSystem` | 3, a model parameter of one system | `physics/mechanics/_mechanics.py` | physics |

No new package and no new concept. The measurement model is the `ObservationSpec` of the foundation and the process
model is the system with its own noise laws, so a filter is a procedure applied to a system, as calibration and
identification are, and its result holds a system.

### Dependencies

Checked on PyPI on 2026-11-10.

| Library | Version | Released | Status |
| --- | --- | --- | --- |
| filterpy | 2.0.0 | 2026-10-26 | Added. MIT, needs numpy, scipy, and matplotlib (already installed through other dependencies). Its `KalmanFilter`, `ExtendedKalmanFilter`, `UnscentedKalmanFilter` with Merwe sigma points, `EnsembleKalmanFilter`, `rts_smoother`, and the resampling of `filterpy.monte_carlo` take the step of the system as a function |
| scoringrules | 0.12.0 | 2026-11-08 | Added with the lower bound 0.11.0 (2026-06-06), since 0.12.0 is two days old. Needs numpy, scipy, and array-api-compat. Its `crps_ensemble` and `interval_score` score ensemble forecasts |

Rejected: dynamax 1.0.3 (2026-10-13), the JAX filtering library, which requires `tfp-nightly`, a nightly build as a hard
dependency, for the same reason the foundation rejects rlax. particles 0.4 (2023-11-06), past the 24 month limit.
DAPPER 1.8.1 (2026-06-23) pins `dill` and needs `notebook`, so it is a source of benchmark numbers only. cuthbertlib
0.1.1 (2026-10-02) is too young. properscoring 0.1 (2015-11-12) is unmaintained. The bootstrap particle filter and the
backward simulation smoother are a few lines on numpy and the resampling of filterpy.

### Assimilation

- `assimilate(system, data, *, method='ukf', smoother=False, run=None, process_noise=None, augmented=(),
  augmented_noise=0.0, ensemble=50, particles=1000, inflation=1.0, resampling='systematic', random_state=None)` in
  `core/experiments/_assimilation.py`.
- **Data.** `data` is an `ObservedRunsSpec` whose observation model is the measurement model, or a mapping that the
  foundation reads as one run. With several runs, `run` names the label to filter, and without it `ValidationError`
  names `run`. Missing values skip the rows of the update they belong to, and observation times may be irregular.
- **Process model.** Every member, sigma point, or particle is advanced from one observation time to the next by the
  compiled step of the system on its backend, vectorised over the members, with the inputs and conditions of the run
  set. The Gaussian filters take the process noise covariance from the SDE terms and map noise of the system, `L L^T`
  times the time between observations to first order, or from `process_noise`, a matrix or one variance per state,
  which overrides it. A deterministic system has none unless `process_noise` is given.
- **Measurement model.** The observation function is the readout of the paths of the `ObservationSpec`, compiled as
  observables of the system. The Gaussian filters take `R` from Gaussian noise, and filter the logarithm of the
  observations for log-normal noise. Count noise needs the particle filter, and the other methods raise
  `ValidationError` naming the noise and `'particle'`.
- **Methods.**
  - `'kalman'` needs linear laws, checked by a Jacobian of the compiled rates that holds no state, and raises
    `ValidationError` naming the first nonlinear rate otherwise.
  - `'ekf'` linearises the step and the observation function with `jax.jacfwd` of the traced step, so it needs the
    differentiable tag.
  - `'ukf'` propagates the Merwe sigma points of filterpy through the step and the observation function, so it needs
    no derivative.
  - `'enkf'` is the stochastic ensemble Kalman filter of filterpy with perturbed observations, `ensemble` members, and
    multiplicative `inflation` of the anomalies.
  - `'particle'` is the bootstrap filter: every particle runs the system's own stochastic step, jump laws included,
    and is weighted by the likelihood of the observation under any of the six noise kinds of the observation model,
    with `resampling` among `'systematic'`, `'stratified'`, `'residual'`, and `'multinomial'` when the effective sample
    size falls below half.
- **Smoothers.** `smoother=True` runs the Rauch-Tung-Striebel smoother after `'kalman'`, `'ekf'` (with the Jacobians of
  the forward pass), and `'ukf'` through filterpy, and the backward simulation smoother of Godsill, Doucet, and West
  (2004) after `'particle'`, as Särkkä (ch. 8 to 11) presents them. `'enkf'` raises `ValidationError` that names the
  smoothers available.
- **Joint estimation.** `augmented` names parameters appended to the state with a random walk of variance
  `augmented_noise` per unit time, and their final means and variances are the `estimates` of the result. This is the
  augmented state method of Särkkä (ch. 12).
- **Result.** `AssimilationResult` holds `system`, the system with its states set to the last estimate and its
  augmented parameters to their estimates, `trajectory`, the filtered or smoothed means at the observation times,
  `covariances` for the Gaussian filters, `ensemble` for `'enkf'` and `'particle'` (members by times by states, after
  resampling), `innovations`, the observation minus its prediction at every time, `log_likelihood`, the sum of the
  predictive log densities of the observations, `estimates`, and `effective_sizes` for `'particle'`.
- **Forecasts.** `simulate(result.system, ...)` forecasts from the last estimate, and an ensemble forecast runs the
  members of `ensemble` through `ensemble` of the experiments with their states as initial states.

### Particle marginal Metropolis-Hastings

- `calibrate(method='pmmh', particles=1000, ...)` runs a random walk Metropolis-Hastings chain on the parameters within
  their priors, with the log-likelihood of each proposal from a bootstrap particle filter over every run of the data,
  after Andrieu, Doucet, and Holenstein (2010) and Särkkä (sect. 12.3). The particle filter is the private filter of
  `_assimilation.py`, imported downward within the experiments package. The random walk scale adapts during warmup to
  an acceptance rate near 0.15, the advice for noisy likelihoods. The result holds the draws, the acceptance rate, and
  the posterior summaries that `'mcmc'` reports.
- It is the method for stochastic systems seen through noise, such as an outbreak with jump laws observed through
  negative binomial counts, which `'mcmc'` cannot score without observing every draw.

### Probabilistic scores

- `forecast_skill` accepts an ensemble forecast: an array with a leading member axis, or a sequence of trajectories.
  `metric='crps'` gives the continuous ranked probability score of the ensemble at every lead time with
  `scoringrules.crps_ensemble`, and every result of an ensemble gains `coverage`, the share of true values inside the
  central intervals of `levels=(0.5, 0.9)`, and `interval_scores` from `scoringrules.interval_score`. The skill
  against persistence uses the absolute error of persistence, to which the CRPS reduces for one member.

### Probes

- **Noisy pendulum.** `PendulumSystem(noise=qc)` adds a white noise of spectral density `qc` to the angular
  acceleration, the model of Särkkä (Example 3.7), and stays deterministic at the default `noise=0`. Its angle is seen
  through `sin(theta)` with Gaussian noise, Example 5.1. The book reports the root mean squared error of the angle on
  one simulated run: 0.12 for the extended Kalman filter, 0.11 for the unscented, Gauss-Hermite, and cubature filters,
  0.12 for the bootstrap particle filter with 10 000 particles, 0.033 for the extended RTS smoother, and 0.028 for the
  unscented RTS smoother and the particle smoother (Examples 5.1, 5.3, 7.1, 9.1, 9.3, and 11.1, checked in
  `references/textbooks/Särkkä - 2013 - Bayesian Filtering and Smoothing.pdf`). The settings of the run, the step 0.01,
  `qc = 0.01`, `g = 9.81`, the measurement variance 0.1, the start `(1.5, 0)`, and the horizon of 5, are not printed
  in the book. They are those of its companion code and must be checked against it before the tests are written. The
  scenarios average 20 seeded runs and allow 0.02, since the book reports one run.
- **Noise variance of the pendulum.** Särkkä (Example 12.2) samples the posterior of the measurement variance of the
  same model by particle MCMC, with the true value near 0.1 inside its high density region. The PMMH scenario checks
  that the true variance lies inside the 90% interval.
- **Car tracking.** The linear Wiener velocity model of Särkkä (Example 4.3) checks `'kalman'` and its smoother:
  position RMSE 0.77 from the measurements alone, 0.43 for the Kalman filter, and 0.27 for the RTS smoother (Figure 8.3).
  Its settings also come from the companion code and must be checked. It is built in the test, not added as a system.
- **Lorenz-96.** `Lorenz96System` with 40 variables and forcing 8, every variable observed every 0.05 with unit
  Gaussian noise, the setting of Sakov and Oke (2008) that DAPPER benchmarks. The ensemble Kalman filter with 40
  members should reach an RMSE near 0.18 after spin-up. That value is from memory of the DAPPER benchmark table and
  must be checked against it, with the matching inflation and filter variant, before it is used as a reference value.
  Until then the scenario checks only that the filter RMSE stays well below the observation noise.

## Risks / Trade-offs

- [The reference RMSE values of the book come from one run] → The scenarios average seeded runs and use a tolerance of
  0.02, and the settings are checked against the companion code first.
- [Particle filters degenerate with many observed variables] → The docstring says so, `effective_sizes` reports it, and
  the ensemble Kalman filter is the method for large states.
- [The particle likelihood is noisy, so PMMH chains stick] → The adaptive random walk targets a low acceptance rate,
  `particles` is an argument, and the scenario uses enough particles for a stable likelihood.
- [filterpy is a numpy library, so filters do not run on the JAX backend] → The step of the system is vectorised over
  members on its own backend, and only the filter algebra is numpy, which is small next to the steps.
- [filterpy 2.0.0 is a fresh major release after years of 1.4.5] → The wrapper uses the classes and functions that
  both versions share, and the tests pin the reference values.

## Later

- Weak constraint 4D-Var as an option of `identify`, and localisation of ensemble filters.
- Learning model error from assimilated states, a gallery example of `assimilate` with `identify`.
- Particle filters with optimal or auxiliary proposals.
