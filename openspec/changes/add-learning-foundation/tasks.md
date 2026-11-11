# Tasks

## 1. Dependencies and placement

- [ ] 1.1 Create the package `src/skcomplex/core/data` with its `__init__.py` and the mirrored `tests/core/data`, add
  `skcomplex.core.data` to the layers contract of `pyproject.toml` between `skcomplex.core.analysis` and
  `skcomplex.core.engine`, add a forbidden contract that keeps it from `skcomplex.core.components` and every engine
  package but `runtime`, and verify the import contracts pass and fail on a test import of `skcomplex.core.experiments`
- [ ] 1.2 Add `skcomplex.core.data` to the package list of the user guide and the API reference, and verify the strict
  docs build
- [ ] 1.3 Add `equinox>=0.13.8` and `xarray>=2026.9.0` to the runtime dependencies in `pyproject.toml` and the Project
  Profile of the constitution, with their release dates, and verify `pdm install` resolves and both import

## 2. Observed data

- [ ] 2.1 Add `ObservationSpec` in `src/skcomplex/core/specs/_observation_spec.py` with its `_check`, export it, and
  verify the unknown-noise scenario and its doctest in `tests/core/specs/test_observation_spec.py`
- [ ] 2.2 Add `ObservedRunSpec` and `ObservedRunsSpec` with their checks of paths and times, export them, and verify the
  paths-with-their-own-times and times-that-decrease scenarios and their doctests
- [ ] 2.3 Add `read_runs` for pandas data frames in long and wide form in `src/skcomplex/core/data/_read.py`, export it
  from `skcomplex.core.data`, and verify the long-frame and missing-column scenarios in `tests/core/data/test_read.py`
- [ ] 2.4 Extend `read_runs` to xarray datasets with the dimensions `time` and `run` and conditions as coordinates, and
  verify the dataset-with-conditions scenario
- [ ] 2.5 Add `observe` on trajectories with the six noise models and readout observables in
  `src/skcomplex/core/data/_observe.py`, export it, and verify the readout-of-a-compartment, under-reported-counts, and
  same-seed scenarios
- [ ] 2.6 Extend `observe` to `ExperimentResult` sources, read through their run result specs, with conditions per
  design point and `replicates`, and verify the sweep-as-runs and experiment-results-read-without-the-experiments-package
  scenarios
- [ ] 2.7 Add `split_runs` in `src/skcomplex/core/data/_split.py` with `kind='runs'` and `kind='time'`, export it,
  and verify the no-run-in-both-sets, too-few-runs, and held-out-years-of-one-run scenarios
- [ ] 2.8 Add `make_arrays` with `kind='transitions'` in `src/skcomplex/core/data/_arrays.py`, export it, and verify the
  states-to-next-states, unknown-path, and pipeline-trained-on-runs scenarios
- [ ] 2.9 Add `kind='runs'` with the summaries and the scalar outputs of experiment results, and verify the
  parameters-to-summaries scenario
- [ ] 2.10 Document observation models, observed runs, and `make_arrays` in the calibrate page of the user guide,
  labelling simulated data as such, with a runnable example that trains a scikit-learn pipeline on the arrays of a
  sweep with `GroupKFold`, and verify its code blocks run

## 3. Learned functions

- [ ] 3.1 Accept positive whole numbers in `ParameterSpec.shape`, lay out and record such parameters, and verify the
  weight-vector and shape-not-positive scenarios
- [ ] 3.2 Add `learned_function` and `learned_weights` for Equinox modules in
  `src/skcomplex/core/components/laws/utils/_learned.py` with the registry of static parts, export them, and verify
  their doctests and a pickle round trip of a system with a learned law
- [ ] 3.3 Lower learned calls in `JaxBackend`, vectorised over members and pairs with `jax.vmap`, and verify the
  network-in-a-rate-law scenario on JAX
- [ ] 3.4 Lower learned calls in the NumPy backend through the jitted JAX function, and verify the network and
  weights-changed scenarios on NumPy
- [ ] 3.5 Report `learned-weights` and `learned-inputs` in the compiler, and verify both scenarios
- [ ] 3.6 Make the Numba path fall back to Python with the existing warning, and verify the 2000-member scenario
- [ ] 3.7 Accept scikit-learn regressors and pipelines with one `predict` per evaluation, refuse unfitted ones at
  `prepare`, report `not-differentiable` on JAX, and verify the fitted-regressor, fitted-pipeline, and
  not-differentiable scenarios
- [ ] 3.8 Mark systems whose learned models are all Equinox modules as differentiable, and verify the
  gradient-of-the-weights scenario against finite differences
- [ ] 3.9 Verify the hybrid-rate and learned-pair-potential scenarios on the NumPy and JAX backends
- [ ] 3.10 Document learned functions, hybrid laws, and learned interactions in the build page of the user guide with
  runnable examples

## 4. One fit loop

- [ ] 4.1 Lift the epochs, batches, optimizer state, and update of `GradientDescentSolver` into the private
  `_OptaxLoop` of the same module, and verify the existing learning-dynamics tests and the same-optimizer-path scenario
- [ ] 4.2 Replace `DatasetSpec` by `ObservedRunsSpec` in `TrainingInteraction` and the solver, unrolling every run on its
  own times with inputs held and conditions set, and verify the inputs-and-targets and unknown-path scenarios
- [ ] 4.3 Pad and mask runs of different lengths and missing values, and verify the masked-steps, shape-mismatch, and
  runs-of-different-lengths scenarios
- [ ] 4.4 Move the Elman network, its gallery example, the user guide, and every test from `DatasetSpec` to observed
  runs, remove `DatasetSpec` with no alias, and add the breaking changelog line with the migration
- [ ] 4.5 Add the loss names `'squares'`, `'log'`, and `'likelihood'` with the six noise models, and verify the
  Poisson-likelihood and likelihood-without-noise scenarios and a binomial likelihood against `scipy.stats.binom` at
  whole counts
- [ ] 4.6 Add `validation` and `patience` with the recorded validation loss and the best parameters, and verify the
  early-stop scenario
- [ ] 4.7 Lift the Diffrax solver, tolerances, and adjoint to module constants read by the solver and
  `calibrate(method='variational')`, and verify the calibration tests still pass

## 5. Identification

- [ ] 5.1 Add `identify` and `IdentificationResult` in `src/skcomplex/core/experiments/_identification.py` with
  `method='gradient'` on the training system, export them, and verify the universal-differential-equation,
  rates-fitted-with-the-network, reproducible-fit, and fitted-system-is-a-system scenarios
- [ ] 5.2 Add the checks of trained names, adaptivity, and stochastic law kinds, and verify the not-adaptive and
  jump-laws scenarios
- [ ] 5.3 Add `initial` and `noise` with the fitted initial states per run and the noise scales per path, add
  `initial_states` and `noise_scales` to the result, and verify the initial-states-of-a-run and noise-scale scenarios
- [ ] 5.4 Add `horizon` with rollouts from every fully observed time and `windows` with trained window starts and the
  `continuity` penalty, and verify the chaotic-run, one-step-prediction, and horizon-with-unobserved-states scenarios
- [ ] 5.5 Accept SDE terms in `method='gradient'` with `paths` Brownian paths per run from
  `diffrax.VirtualBrownianTree` and the loss of their mean and variance, and verify the Ornstein-Uhlenbeck scenario
- [ ] 5.6 Add `method='derivatives'` with pysindy's smoothed differences and the fit of regressors on residual rates,
  and verify the regressor-of-a-missing-term and unobserved-state scenarios
- [ ] 5.7 Make `calibrate` read `ObservedRunsSpec`, and verify the one-run scenario and the existing calibration tests
- [ ] 5.8 Make `_read_samples` read `ObservedRunsSpec` for `sparse_identification`, `mode_decomposition`, and
  `reservoir`, and verify the runs-from-a-reader and uneven-times scenarios
- [ ] 5.9 Document `identify`, its two methods, initial states and noise, prediction error and windows, SDE fits, and
  the difference from `calibrate` in the calibrate page of the user guide with runnable examples

## 6. Reinforcement learning

- [ ] 6.1 Add `EnvironmentSpec` in `src/skcomplex/core/specs/_environment_spec.py` with its checks, export it, and verify
  the action-not-adaptive scenario and its doctest
- [ ] 6.2 Add `PolicyInteraction` in `src/skcomplex/core/components/interactions/_policy.py` with categorical, normal,
  and deterministic actions as named draws, export it, and verify its doctest
- [ ] 6.3 Make the runtime split runs at decision times and hold actions as the schedule holds parameters, and verify
  the actions-held-between-decisions and replay-of-actions scenarios
- [ ] 6.4 Act for every member of a group with shared weights, and verify the shared-weights-over-a-group scenario
- [ ] 6.5 Lower the deterministic policy on JAX in the traced function, and verify the
  deterministic-policy-is-differentiable scenario
- [ ] 6.6 Add the agent form of the policy on the Mesa backend through the system view, and verify a policy of a group
  of agents acts in their step
- [ ] 6.7 Add `train_policy` and `PolicyTrainingResult` in `src/skcomplex/core/experiments/_policy_training.py` with
  `'reinforce'` on rollouts of the `ensemble` runner cut at the terminal observable, export them, and verify the
  learning-improves-the-return, same-seed, and no-policy scenarios
- [ ] 6.8 Add `'ppo'` with the critic, generalised advantage estimates, and the clipped surrogate objective on the shared
  loop, and verify the cart-pole-balanced scenario, marked slow
- [ ] 6.9 Add `'pathwise'` through `GradientDescentSolver` with a reward loss, and verify the
  pathwise-needs-a-differentiable-system scenario and a pendulum held upright by a trained deterministic policy
- [ ] 6.10 Record the training of every method as the `trajectory` of `PolicyTrainingResult`, with the epochs as times
  and the weights, mean return, and loss as series, and verify the training-as-a-trajectory scenario for the three
  methods
- [ ] 6.11 Document environments, policies, and `train_policy` in the experiments page of the user guide, with the
  design of off-policy and multi-agent methods named as follow-ups, and verify its code blocks run

## 7. Forecast skill

- [ ] 7.1 Add `forecast_skill` and `ForecastSkillResult` in `src/skcomplex/core/analysis/_forecast_skill.py` with both
  metrics, the valid time, and the persistence skill, export them with their analysis tags, and verify the
  perfect-forecast, persistence, and lengths-differ scenarios and that `applicable_analyses` lists it
- [ ] 7.2 Add a paragraph and a runnable example scoring the forecast of `reservoir` in the analysis page of the user
  guide

## 8. Probes

- [ ] 8.1 Add `UniversalLotkaVolterraSystem` in `src/skcomplex/systems/biology/ecology/_universal_lotka_volterra.py`
  with both forms, export it, and verify the untrained-network scenario in `tests/systems/biology/ecology/`
- [ ] 8.2 Read the posterior means of the initial populations and noise scales from Carpenter (2018), and verify the
  mechanistic-fit scenario with `initial` and `noise` and the held-out-years scenario with `split_runs(kind='time')` on
  the pelt records, read from the same source as the build tutorial
- [ ] 8.3 Fetch the 1978 boarding school counts from the note of the British Medical Journal, check them against the
  R package `outbreaks`, and store them with their provenance as a data file of the epidemiology package read with
  `importlib.resources.files`
- [ ] 8.4 Read the posterior intervals of `beta` and `gamma` from Grinsztajn et al. (2021), add
  `LearnedContactSIRSystem` in `src/skcomplex/systems/biology/epidemiology/_learned_contact.py` with both forms, export
  it, and verify the constant-form and learned-contact-rate scenarios
- [ ] 8.5 Add `CartPoleSystem` in `src/skcomplex/systems/engineering/control/_cart_pole.py` with its environment and
  policy, export it, and verify the pole-falls and equations-of-Florian scenarios
- [ ] 8.6 Add the gallery examples `docs/examples/biology/ecology/plot_universal_lotka_volterra.py`,
  `docs/examples/biology/epidemiology/plot_learned_contact_sir.py`, and
  `docs/examples/engineering/control/plot_cart_pole.py`, and verify the gallery build runs them
- [ ] 8.7 Add the inventory entries `universal_lotka_volterra`, `learned_contact_sir`, and `cart_pole` with their
  sources, and the primitives `observed-data`, `learned-functions`, and `policies` with
  `change = "add-learning-foundation"`, and verify the inventory check passes

## 9. Checks

- [ ] 9.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules with the thread
  limits set to 1, doctests, the inventory check, and the strict docs build) and verify every session passes
