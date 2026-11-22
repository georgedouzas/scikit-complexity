# Tasks

## 1. Dependencies and placement

- [ ] 1.1 Create the package `src/skcomplex/core/data` with its `__init__.py` and the mirrored `tests/core/data`, add
  `skcomplex.core.data` to the layers contract of `pyproject.toml` between `skcomplex.core.analysis` and
  `skcomplex.core.systems`, add a forbidden contract that keeps it from the subpackages of
  `skcomplex.core.components`, from `skcomplex.core.systems`, and from every engine package but `runtime`, and verify
  the import contracts pass and fail on a test import of `skcomplex.core.experiments`
- [ ] 1.2 Add `skcomplex.core.data` to the package list of the user guide and the API reference, and verify the strict
  docs build
- [ ] 1.3 Add `equinox>=0.13.8` and `xarray>=2026.9.0` to the runtime dependencies in `pyproject.toml` and the Project
  Profile of the constitution, with their release dates, and verify `pdm install` resolves and both import

## 2. Observed data

- [ ] 2.1 Add `ObservationSpec` in `src/skcomplex/core/components/_observation_spec.py` with its `_check`, export it
  from `skcomplex.core.components`, and verify the unknown-noise scenario and its doctest in
  `tests/core/components/test_observation_spec.py`
- [ ] 2.2 Add `ObservedRunSpec` in `src/skcomplex/core/components/_observed_run_spec.py` and `ObservedRunsSpec` in
  `src/skcomplex/core/components/_observed_runs_spec.py` with their checks of paths and times, export them from
  `skcomplex.core.components`, and verify the paths-with-their-own-times and times-that-decrease scenarios and their
  doctests in `tests/core/components/test_observed_run_spec.py` and `tests/core/components/test_observed_runs_spec.py`
- [ ] 2.3 Add `read_runs` for pandas data frames in long and wide form in `src/skcomplex/core/data/_read.py`, export it
  from `skcomplex.core.data`, and verify the long-frame and missing-column scenarios in `tests/core/data/test_read.py`
- [ ] 2.4 Extend `read_runs` to xarray datasets with the dimensions `time` and `run` and conditions as coordinates, and
  verify the dataset-with-conditions scenario
- [ ] 2.5 Add `observe` on trajectories with the six noise models and readout observables in
  `src/skcomplex/core/data/_observe.py`, export it, and verify the readout-of-a-compartment, under-reported-counts, and
  same-seed scenarios in `tests/core/data/test_observe.py`
- [ ] 2.6 Extend `observe` to `ExperimentResult` sources, read through their `runs` and their fields without importing
  `skcomplex.core.experiments`, with conditions per design point and `replicates`, and verify the sweep-as-runs and experiment-results-read-without-the-experiments-package
  scenarios
- [ ] 2.7 Add `split_runs` in `src/skcomplex/core/data/_split.py` with `kind='runs'` and `kind='time'`, export it,
  and verify the no-run-in-both-sets, too-few-runs, and held-out-years-of-one-run scenarios in
  `tests/core/data/test_split.py`
- [ ] 2.8 Add `make_arrays` with `kind='transitions'` in `src/skcomplex/core/data/_arrays.py`, export it, and verify the
  states-to-next-states, unknown-path, and pipeline-trained-on-runs scenarios in `tests/core/data/test_arrays.py`
- [ ] 2.9 Add `kind='runs'` with the summaries and the scalar outputs of experiment results, and verify the
  parameters-to-summaries scenario
- [ ] 2.10 Document observation models, observed runs, and `make_arrays` in the calibrate page of the user guide,
  labelling simulated data as such, with a runnable example that trains a scikit-learn pipeline on the arrays of a
  sweep with `GroupKFold`, and verify its code blocks run

## 3. Learned functions

- [ ] 3.1 Add `learned_function` and `learned_weights` for Equinox modules in
  `src/skcomplex/core/components/laws/utils/_learned.py`, with a private `BuiltinFunction` subclass and the registry of
  static parts, export them from `skcomplex.core.components.laws.utils`, and verify their doctests, a pickle round trip
  of a system with a learned law, and the weights-recorded-with-their-shape scenario in
  `tests/core/components/laws/utils/test_learned.py`
- [ ] 3.2 Lower learned calls in `JaxBackend`, vectorised over members and pairs with `jax.vmap`, and verify the
  network-in-a-rate-law scenario on JAX
- [ ] 3.3 Lower learned calls in the NumPy backend through the jitted JAX function, and verify the network and
  weights-changed scenarios on NumPy
- [ ] 3.4 Report `learned-weights` and `learned-inputs` in the compiler, and verify both scenarios
- [ ] 3.5 Make the Numba path fall back to Python with the existing warning, and verify the 2000-member scenario
- [ ] 3.6 Accept scikit-learn regressors and pipelines with one `predict` per evaluation, refuse unfitted ones at
  `prepare`, report `not-differentiable` on JAX, and verify the fitted-regressor, fitted-pipeline, unfitted-regressor,
  and not-differentiable scenarios
- [ ] 3.7 Give the tag `differentiable` to systems whose learned models are all Equinox modules, and verify the
  gradient-of-the-weights scenario against finite differences
- [ ] 3.8 Verify the hybrid-rate and learned-pair-energy scenarios on the NumPy and JAX backends
- [ ] 3.9 Document learned functions, hybrid laws, and learned interactions in the build page of the user guide with
  runnable examples

## 4. One fit loop

- [ ] 4.1 Lift the epochs, batches, optimizer state, and update of `GradientDescentSolver` into the private
  `_OptaxLoop` of the same module, and verify the existing tests in `tests/core/engine/solvers/test_gradient_descent.py`
  and the same-optimizer-path scenario
- [ ] 4.2 Replace `DatasetSpec` by `ObservedRunsSpec` in `TrainingInteraction`, the learning compiler, and the solver,
  unrolling every run on its own times with inputs held and conditions set, and verify the inputs-and-targets and
  unknown-path scenarios in `tests/core/components/interactions/test_training.py` and
  `tests/core/engine/compile/test_learning_compiler.py`
- [ ] 4.3 Pad and mask runs of different lengths and missing values, and verify the masked-steps, shape-mismatch, and
  runs-of-different-lengths scenarios
- [ ] 4.4 Move the Elman network of `src/skcomplex/systems/computer_science/neural_networks/_elman.py`, its gallery
  example, the user guide, and every test from `DatasetSpec` to observed runs, remove
  `src/skcomplex/core/components/laws/_dataset_spec.py` and its export with no alias, and add the breaking changelog
  line with the migration
- [ ] 4.5 Add the loss names `'squares'`, `'log'`, and `'likelihood'` with the six noise models, and verify the
  Poisson-likelihood and likelihood-without-noise scenarios and a binomial likelihood against `scipy.stats.binom` at
  whole counts
- [ ] 4.6 Add `validation` and `patience` with the recorded validation loss and the best parameters, and verify the
  early-stop scenario
- [ ] 4.7 Lift the Diffrax solver, tolerances, and adjoint to public module constants of
  `src/skcomplex/core/engine/solvers/_gradient_descent.py` read by the solver and `calibrate(method='variational')`,
  and verify the tests in `tests/core/experiments/test_calibration.py` still pass

## 5. Identification

- [ ] 5.1 Add `identify` and `IdentificationResult` in `src/skcomplex/core/experiments/_identification.py` with
  `method='gradient'` on the training system, which clones the system, nests it in a `System` with a
  `TrainingInteraction`, and prepares it with `GradientDescentSolver`, export them, and verify the
  universal-differential-equation, rates-fitted-with-the-network, reproducible-fit, and fitted-system-is-a-system
  scenarios in `tests/core/experiments/test_identification.py`
- [ ] 5.2 Add the checks of trained names against `get_params` and of random law kinds, and verify the
  unknown-parameter and jump-laws scenarios
- [ ] 5.3 Add `initial` and `noise` with the fitted initial states per run and the noise scales per path, add
  `initial_states` and `noise_scales` to the result, and verify the initial-states-of-a-run and noise-scale scenarios
- [ ] 5.4 Add `horizon` with rollouts from every fully observed time and `windows` with trained window starts and the
  `continuity` penalty, and verify the chaotic-run, one-step-prediction, and horizon-with-unobserved-states scenarios
- [ ] 5.5 Accept noise laws in `method='gradient'` with `paths` Brownian paths per run from
  `diffrax.VirtualBrownianTree` and the loss of their mean and variance, and verify the Ornstein-Uhlenbeck scenario
- [ ] 5.6 Add `method='derivatives'` with pysindy's smoothed differences and the fit of regressors on residual rates,
  and verify the regressor-of-a-missing-term and unobserved-state scenarios
- [ ] 5.7 Make `calibrate` read `ObservedRunsSpec`, and verify the one-run scenario and the existing tests in
  `tests/core/experiments/test_calibration.py`
- [ ] 5.8 Make `_read_samples` read `ObservedRunsSpec` for `sparse_identification`, `mode_decomposition`, and
  `reservoir`, and verify the runs-from-a-reader and uneven-times scenarios in `tests/core/analysis/test_base.py`
- [ ] 5.9 Document `identify`, its two methods, initial states and noise, prediction error and windows, fits of noise laws, and
  the difference from `calibrate` in the calibrate page of the user guide with runnable examples

## 6. Reinforcement learning

- [ ] 6.1 Add `EnvironmentSpec` in `src/skcomplex/core/components/interactions/_environment_spec.py` with its checks,
  export it from `skcomplex.core.components.interactions`, and verify its doctest in
  `tests/core/components/interactions/test_environment_spec.py`
- [ ] 6.2 Add `PolicyInteraction` in `src/skcomplex/core/components/interactions/_policy.py`, whose `_prepare` sets one
  `UpdateLaw` per action with the named draws `categorical` or `normal` of `laws.utils` or the deterministic mean,
  export it, and verify its doctest and the unknown-action and bounds-outside-the-allowed-values scenarios in
  `tests/core/components/interactions/test_policy.py`
- [ ] 6.3 Make the runtime split runs at decision times, apply the laws of the policy there, and hold actions as the
  schedule holds parameters, and verify the actions-held-between-decisions and replay-of-actions scenarios
- [ ] 6.4 Act for every member of a group with shared weights, and verify the shared-weights-over-a-group scenario and
  that agents of the agent solver read the actions of their group
- [ ] 6.5 Lower the deterministic policy on JAX in the traced function, and verify the
  deterministic-policy-is-differentiable scenario
- [ ] 6.6 Add `train_policy` and `PolicyTrainingResult` in `src/skcomplex/core/experiments/_policy_training.py` with
  `'reinforce'` on rollouts of the `ensemble` runner cut at the terminal observable, export them, and verify the
  learning-improves-the-return, same-seed, and no-policy scenarios in `tests/core/experiments/test_policy_training.py`
- [ ] 6.7 Add `'ppo'` with the critic, generalised advantage estimates, and the clipped surrogate objective on the shared
  loop, and verify the cart-pole-balanced scenario, marked slow
- [ ] 6.8 Add `'pathwise'` through `GradientDescentSolver` with a reward loss, and verify the
  pathwise-needs-a-differentiable-system scenario and a pendulum held upright by a trained deterministic policy
- [ ] 6.9 Record the training of every method as the `trajectory` of `PolicyTrainingResult`, with the epochs as times
  and the weights, mean return, and loss as series, and verify the training-as-a-trajectory scenario for the three
  methods
- [ ] 6.10 Document environments, policies, and `train_policy` in the experiments page of the user guide, with the
  design of off-policy and multi-agent methods named as follow-ups, and verify its code blocks run

## 7. Forecast skill

- [ ] 7.1 Add `forecast_skill` and `ForecastSkillResult` in `src/skcomplex/core/analysis/_forecast_skill.py` with both
  metrics, the valid time, and the persistence skill, export them with their analysis tags, and verify the
  perfect-forecast, persistence, and lengths-differ scenarios and that `applicable_analyses` lists it in
  `tests/core/analysis/test_forecast_skill.py`
- [ ] 7.2 Add a paragraph and a runnable example scoring the forecast of `reservoir` in the analysis page of the user
  guide

## 8. Probes

- [ ] 8.1 Add `UniversalLotkaVolterraSystem` in `src/skcomplex/systems/biology/ecology/_universal_lotka_volterra.py`
  with both forms, export it, and verify the untrained-network scenario in
  `tests/systems/biology/ecology/test_universal_lotka_volterra.py`
- [ ] 8.2 Read the posterior means of the initial populations and noise scales from Carpenter (2018), and verify the
  mechanistic-fit scenario with `initial` and `noise` and the held-out-years scenario with `split_runs(kind='time')` on
  the pelt records, read from the same source as the build tutorial
- [ ] 8.3 Fetch the 1978 boarding school counts from the note of the British Medical Journal, check them against the
  R package `outbreaks`, and store them with their provenance as a data file of the epidemiology package read with
  `importlib.resources.files`
- [ ] 8.4 Read the posterior intervals of `beta` and `gamma` from Grinsztajn et al. (2021), add
  `LearnedContactSIRSystem` in `src/skcomplex/systems/biology/epidemiology/_learned_contact.py` with both forms, export
  it, and verify the constant-form and learned-contact-rate scenarios in
  `tests/systems/biology/epidemiology/test_learned_contact.py`
- [ ] 8.5 Add `CartPoleSystem` in `src/skcomplex/systems/engineering/control/_cart_pole.py` with its environment,
  policy, and tuned Euler solver, export it, and verify the pole-falls and equations-of-Florian scenarios in
  `tests/systems/engineering/control/test_cart_pole.py`
- [ ] 8.6 Add the gallery examples `docs/examples/biology/ecology/plot_universal_lotka_volterra.py`,
  `docs/examples/biology/epidemiology/plot_learned_contact_sir.py`, and
  `docs/examples/engineering/control/plot_cart_pole.py`, and verify the gallery build runs them
- [ ] 8.7 Add the inventory entries `universal_lotka_volterra`, `learned_contact_sir`, and `cart_pole` with their
  sources, and the primitives `observed-data`, `learned-functions`, and `policies` with
  `change = "add-learning-foundation"`, and verify the inventory check passes

## 9. Checks

- [ ] 9.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules with the thread
  limits set to 1, doctests, the inventory check, and the strict docs build) and verify every session passes
