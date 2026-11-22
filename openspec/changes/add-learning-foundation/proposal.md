## Why

The library exists to build and simulate systems, and in system theory a learner, a controller, a data set, and an
environment are systems too. Data-driven and machine-learning methods should therefore enter as building blocks of the
same kind: a law whose term is learned, a policy that is an interaction of a system, a fit that is an experiment whose
result holds a system. Today the pieces exist but do not meet. `TrainingInteraction` trains parameters of a nested
system through an unrolled run, but its `DatasetSpec` reads only regular sequences of parameter values. `calibrate`
fits known parameters, but its differentiable run is private. `sparse_identification`, `mode_decomposition`, and
`reservoir` learn systems, but each reads trajectories its own way. No law can call a network or a regressor, no
interaction can choose actions, and no experiment can train a policy.

The drafted changes, from hybrid laws and learned indicators to torch surrogates and language-model agents, each
plan their own network primitive, their own training loop, their own data reading, or their own dependency on Equinox.
This change defines those shared blocks once, so every draft becomes a realization on top of them and new methods,
from system identification to reinforcement learning, plug into the same parts.

## What Changes

- **Observed data.** New frozen specs `ObservationSpec`, the observation model of which states or observables are
  seen and with what noise, and `ObservedRunSpec` and `ObservedRunsSpec`, observed runs with their own irregular
  times, missing values, inputs, and conditions. They live in `skcomplex.core.components` beside `Observable`, since
  `TrainingInteraction` and `PolicyInteraction` read them. A new package `skcomplex.core.data`, approved on
  2026-11-10, holds the functions that make and reshape data: `read_runs`, which reads observed runs from a pandas
  data frame or an xarray dataset, `observe`, which turns trajectories and experiment results into observed runs
  through an observation model, `split_runs`, which splits whole runs, or the times of every run, into training and
  validation sets, and `make_arrays`, which turns runs, trajectories, ensembles, or sweeps into the arrays `X` and `y`
  of scikit-learn, from parameters to summaries or from states to later states with lags, so any scikit-learn pipeline
  trains on simulated or observed data. Every fit reads observed runs: `calibrate`, `identify`, the surrogate
  analyses, and `TrainingInteraction`.
- **Learned functions in laws.** New law function `learned_function(model, *inputs, weights=...)` that calls a learned
  model inside any law an interaction states. The model is an Equinox module, whose weights are one array parameter of
  the interaction that the JAX and NumPy backends lower and differentiate, or a scikit-learn regressor or pipeline,
  which the NumPy backend calls. A learned term added to a known term is a hybrid law, a learned pair energy is a
  learned interaction, and a symbolic term with trained coefficients needs nothing new.
- **One fit loop.** The Optax loop of `GradientDescentSolver` becomes the one loop of the library. It takes batches of
  runs and a gradient from one of two estimators, pathwise through a differentiable run or by the score of recorded
  draws, with validation runs and early stopping. Losses follow from the noise of the observation model: squares,
  squares of logarithms, or the likelihood of Gaussian, log-normal, Poisson, negative binomial, or binomial noise, the
  last for counts reported with a probability, as under-reported cases are.
- **BREAKING: `ObservedRunsSpec` replaces `DatasetSpec`.** Decided on 2026-11-10. `DatasetSpec` is removed from
  `skcomplex.core.components.laws` with no alias. A data set of inputs and targets is a set of observed runs whose
  inputs are the parameter series. `TrainingInteraction` takes it as `data`.
- **scikit-learn inside, not around.** Systems are not scikit-learn estimators, and no class gains `fit` or
  `predict`. scikit-learn enters as learned terms of laws and through `make_arrays`, and its splitters, scorers, and
  parameter grids enter `identify`, `calibrate`, and `train_policy` in `add-learning-bridges`. Systems keep `prepare`
  and `simulate`, engine classes `prepare` and `run`, and the other parts `prepare` alone.
- **Identification as an experiment.** New experiment `identify(system, data, *, trained, method, ...)` that clones a
  system, fits the parameters named by `trained`, learned weights included, to observed runs, and returns a frozen
  `IdentificationResult` whose `system` is the clone with the fitted values set through `set_params`.
  `method='gradient'` trains through the run with the shared loop, and `method='derivatives'` fits scikit-learn
  regressors to the residual rates of observed states. `identify` also fits the initial states of every run and the
  scale of the observation noise, as published fits of real data do, fits systems whose stochastic laws are noise laws
  through Diffrax, and replaces the error of the whole run by the error of short predictions or of windows (multiple
  shooting) for long or chaotic runs.
- **Reinforcement learning.** New frozen spec `EnvironmentSpec`, which reads any system as an environment: its
  observation, its actions as parameters with bounds or choices, its reward and terminal observables, and the time
  between decisions. New core interaction `PolicyInteraction`, a policy as a part of the system, whose laws set the
  actions from a learned function of the observation by a named categorical, normal, or deterministic draw. New
  experiment `train_policy(system, *, method, ...)` with `'reinforce'` and `'ppo'`, on-policy by the score of recorded
  draws, and `'pathwise'`, through the differentiable run of a system. It returns a frozen `PolicyTrainingResult` whose
  `system` holds the trained policy and whose `trajectory` records the training, epoch by epoch, for every method. A
  policy bound to a group acts for every member with shared weights. Off-policy and multi-agent methods are designed
  here and specified by follow-up changes.
- **Forecast skill.** New analysis `forecast_skill(forecast, truth, ...)` that scores a forecast against the truth by
  lead time, with the valid prediction time and the skill against persistence, for every learned or pretrained
  forecaster.
- **Probes.** New systems, each with model parameters, reference values, and a gallery example:
  - `UniversalLotkaVolterraSystem` in `biology/ecology`, the predator and prey model with a learned predation term,
    identified from the hare and lynx pelt records that the user guide already uses
  - `LearnedContactSIRSystem` in `biology/epidemiology`, an outbreak whose contact rate is a learned function of time,
    identified from the bed-confined counts of the 1978 boarding school influenza outbreak, observed partially and with
    count noise
  - `CartPoleSystem` in `engineering/control`, the cart and pole of Barto, Sutton, and Anderson (1983), balanced by a
    policy trained with `train_policy`
- **Dependencies.** Equinox 0.13.8 and xarray 2026.9.0 become runtime dependencies.

## Capabilities

### New Capabilities

- `observed-data`: observation models, observed runs with irregular times, missing values, inputs, and conditions,
  their readers from pandas and xarray, simulated observations, splits of runs, and the arrays of scikit-learn made
  from runs.
- `learned-functions`: Equinox modules and scikit-learn regressors and pipelines called from laws, weights as array
  parameters of interactions, hybrid laws, and learned interactions.
- `reinforcement-learning`: systems read as environments, policies as interactions, and the training of policies as an
  experiment.

### Modified Capabilities

- `learning-dynamics`: observed runs replace data sets, losses follow from observation models, validation runs stop
  training early, and the Optax loop is shared with every training.
- `experiments`: the experiment `identify`, and `calibrate` reads observed runs.
- `data-driven-dynamics`: the surrogate analyses read observed runs.
- `trajectory-analysis`: the forecast skill of learned and pretrained forecasters.
- `model-library`: the universal Lotka-Volterra, learned contact SIR, and cart-pole systems.
- `package-layout`: the package `skcomplex.core.data` and its place between analysis and the systems of the core.

## Impact

- Core: new modules `_observation_spec.py`, `_observed_run_spec.py`, and `_observed_runs_spec.py` in
  `core/components`, a new module `components/laws/utils/_learned.py`, the JAX and NumPy backends and the Numba
  fallback, `components/interactions` with new modules `_policy.py` and `_environment_spec.py`,
  `core/engine/solvers/_gradient_descent.py`, `core/engine/compile/_learning_compiler.py`,
  `components/interactions/_training.py`, new modules `_identification.py` and `_policy_training.py` in
  `core/experiments`, `_calibration.py`, `core/analysis/_base.py`, a new `core/analysis/_forecast_skill.py`, and the
  new package `core/data` with its layer in the import contracts of `pyproject.toml`.
- Systems: new modules in `biology/ecology`, `biology/epidemiology`, and `engineering/control`, and the Elman network
  of `computer_science/neural_networks`, which moves to observed runs.
- API: **BREAKING** `DatasetSpec` is removed with no alias. The Elman network, the user guide, and the tests move to
  `ObservedRunsSpec`.
- Dependencies: `equinox>=0.13.8` (0.13.8, released 2026-05-05) and `xarray>=2026.9.0` (2026.9.0, released
  2026-09-29). JAX, Optax, Diffrax, numpyro, pandas, pysindy, and scikit-learn are already runtime dependencies.
- Docs: sections in the build, calibrate, experiments, and analysis pages of the user guide, three gallery examples,
  and the inventory primitives `observed-data`, `learned-functions`, and `policies`.
- Follow-up drafts, in order: `add-learning-bridges` (scikit-learn splitters, scorers, and parameter grids inside
  `identify`, `calibrate`, and `train_policy`, information criteria and residual tests, and Gymnasium and PettingZoo
  environments), `add-off-policy-learning`, and `add-multi-agent-learning`. `add-state-estimation` (filters,
  smoothers, and data assimilation on the observation model) and `add-experimental-design` (optimal designs and design
  optimisation) also build on it.
- Drafts that build on it: `add-hybrid-neural-laws`, `extend-law-discovery`, `add-learned-indicators`,
  `add-generative-samplers`, `add-neural-particle-simulators`, `add-equilibrium-census`, `add-torch-surrogates`,
  `extend-network-and-emergence-measures`, `add-diversity-search`, and `add-language-model-agents`.
