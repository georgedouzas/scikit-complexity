## Context

- `GradientStepLaw`, returned by `TrainingInteraction` (`core/components/interactions/_training.py`), trains adaptive
  parameters of a nested system over epochs, which are the time of the outer system. `GradientDescentSolver`
  (`core/engine/solvers/_gradient_descent.py`) unrolls the nested system on the pure function `traced_` of `JaxBackend`,
  with one `diffrax.diffeqsolve` per inner step for ODEs, and updates the parameters with an Optax optimizer. Its data
  is a `DatasetSpec` of parameter values of shape `(sequences, steps, ...)` on one shared grid of times.
- `calibrate` (`core/experiments/_calibration.py`) fits model parameters within bounds with SciPy, Optuna, pyabc, and
  numpyro. Its private `_CalibrationObjective.get_traced` compiles a system on JAX and returns the differentiable map
  from a parameter vector to the simulated values at the data times. Its data is one mapping from paths to times and
  values.
- `sparse_identification`, `mode_decomposition`, and `reservoir` read trajectories through `_read_samples` in
  `core/analysis/_base.py` and return results whose `system` runs through `simulate`. `reservoir` runs its fitted map
  through a `CallSpec`.
- Bounded adaptive parameters are controls (`control` spec), set by the `schedule` of `simulate`, by optimal control
  solvers, or by a controller. `simulate(record_draws=True)` keeps named draws, and the event likelihood of `calibrate`
  scores them.
- Law functions such as `series` and `lookup` live in `components/laws/utils` and return Sage expressions that every
  backend lowers. `ParameterSpec.shape` holds role names only.
- The drafted changes plan their own learning blocks: `neural_function` and `neural_identification` in
  `add-hybrid-neural-laws`, Equinox training loops in `add-learned-indicators`, `add-generative-samplers`, and
  `add-neural-particle-simulators`, private readers of experiment results in `extend-law-discovery`, and forecast
  errors in `add-torch-surrogates`.

## Goals / Non-Goals

**Goals:**
- Express every data-driven and learning method with the existing concepts: a learned function is a term of a law, a
  policy is an interaction, a data set is observed runs, identification and training are experiments whose results hold
  a system, and the record of a training is a trajectory.
- One observation model and one data spec, read by every fit.
- One Optax loop, shared by identification, policy training, and the training of learned analyses.
- Reinforcement learning on any system, with on-policy methods implemented and off-policy and multi-agent methods
  designed so their follow-ups add estimators, not concepts.
- Probes in three fields, on real data where it exists.

**Non-Goals:**
- scikit-learn splitters, scorers, and parameter grids inside `identify`, `calibrate`, and `train_policy`, and the
  Gymnasium and PettingZoo environments. They are the follow-up `add-learning-bridges`. scikit-learn regressors and
  pipelines as learned models and `make_arrays` are part of this change.
- Adapter classes with `fit` and `predict` around systems or experiments. They are rejected below.
- Off-policy methods (Q-learning, soft actor-critic, fitted Q iteration) and multi-agent methods (independent learners,
  centralised critics, self-play). They are designed below and specified by `add-off-policy-learning` and
  `add-multi-agent-learning`.
- Torch models. `add-torch-surrogates` keeps them, trained by their own libraries.
- Mixed effects, covariate maps, and dosing on JAX, which stay in `add-hybrid-neural-laws` on top of these blocks.
- GPU tuning. Everything runs on the CPU in double precision.
- Filters, smoothers, and the likelihood of partially observed stochastic systems by particle filtering. They are
  `add-state-estimation`, which reads the same observation model.
- The spread of learned weights over seeds, information criteria, and residual tests. They are `add-learning-bridges`.

## Decisions

### Everything is a system

| Machine-learning notion | Building block |
| --- | --- |
| Data set | `ObservedRunsSpec`, observed runs of states or observables through an `ObservationSpec` |
| Observation model, sensor | `ObservationSpec`, which paths are seen and with what noise |
| Learned model, network, regressor | `learned_function`, a term of a law, with weights as one adaptive parameter |
| Hybrid model, universal differential equation | a law with a known term and a learned term |
| Learned interaction, learned potential | an interaction whose law calls `learned_function` on the states of its roles |
| Loss | the negative log-likelihood of the observation model, or a loss observable |
| Training | the system of `TrainingInteraction`, whose time is epochs and whose states are the trained parameters |
| System identification, system from data | the experiment `identify`, whose result holds a system |
| Environment | any system read through an `EnvironmentSpec` |
| Agent, controller, policy | `PolicyInteraction`, a component that sets bounded adaptive parameters from its observation |
| Episode, rollout | a run of `simulate` with recorded draws |
| Reward, return | an observable of the system, summed over decisions with a discount |
| Reinforcement learning | the experiment `train_policy`, whose result holds a system with the trained policy |
| Forecast evaluation | the analysis `forecast_skill` |
| Feature and target arrays | `make_arrays`, from runs to the `X` and `y` of any scikit-learn pipeline |

No machine-learning package, base class, or verb is added. The one new package, `skcomplex.core.data`, holds functions
that make and reshape data. Learned components are configured with `prepare` and run with `run`, like every component.

### scikit-learn inside, not around

Decided on 2026-11-10. A system is the model being simulated, not an estimator, so no system and no experiment is
wrapped in a scikit-learn estimator. scikit-learn enters in three places instead:

- **Learned terms.** A fitted regressor or pipeline is the model of `learned_function`, a term of a law.
- **Model selection inside the experiments.** `identify`, `calibrate`, and `train_policy` accept scikit-learn
  splitters as `cv`, scorers as `scoring`, and parameter grids as `param_grid`, and their frozen results report the
  score of every fold and the selected settings. This is specified by `add-learning-bridges`, since it adds arguments
  and result fields to three experiments and `identify` works without it.
- **Arrays for any pipeline.** `make_arrays` of `skcomplex.core.data` turns runs into `X`, `y`, and `groups`, so any
  pipeline trains on simulated or observed data and any splitter keeps runs whole.

`prepare` and `run` stay the only verbs of the classes of the library.

Alternatives considered:

- Adapter classes such as `IdentificationEstimator` and `SurrogateEstimator`, with `fit`, `predict`, and `score` around
  `identify` and the surrogate analyses. Rejected: they are a second way of doing what `identify` does, and they make
  a system look like an estimator when it is the model being simulated.
- Systems with `fit` and `predict` themselves. Rejected for the same reasons, and because it would add a second verb
  pair to every class.

The accepted cost: a fitted system cannot sit inside a scikit-learn `Pipeline` or `GridSearchCV`. Model selection
runs inside the experiments, and pipelines train on the arrays of `make_arrays` or sit inside laws.

### Placement

| Component | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `ObservationSpec`, `ObservedRunSpec`, `ObservedRunsSpec` | 2, the losses, `identify`, `calibrate`, and the surrogate analyses read them by type | `skcomplex.core.specs` | ecology (pelt and census counts), epidemiology (reported cases), pharmacology (sparse samples), climate (station series), engineering (sensors), neuroscience (recordings) |
| `read_runs`, `observe`, `split_runs`, `make_arrays` | 2, functions that build the spec the fits read and the arrays any pipeline reads, as `sklearn.datasets` and `train_test_split` do | `skcomplex.core.data`, approved on 2026-11-10 | as above |
| Whole-number sizes of `ParameterSpec.shape` | 1, the structural spec of every parameter | `skcomplex.core.specs` | every learned function, and fixed-size vectors in any field |
| `learned_function`, `learned_weights` | 2, a law function the backends lower by the type of its model | `skcomplex.core.components.laws.utils` | ecology and epidemiology (unknown rates), pharmacology (covariate maps), physics (learned potentials, neural cellular automata), engineering (learned controllers), neuroscience (learned couplings) |
| Lowering of learned functions | 2, backend lowering | `core/engine/backends/_jax.py`, `_numpy.py`, and the Numba fallback | as above |
| `EnvironmentSpec` | 2, `PolicyInteraction`, `train_policy`, and the bridges read it by type | `skcomplex.core.specs` | engineering (control), epidemiology (interventions), ecology (harvests), economics (decisions of agents), social systems (learning agents), computer science (games) |
| `PolicyInteraction` | 2, the runtime holds its actions between decisions and the training finds it by type | `skcomplex.core.components.interactions` | as above |
| The shared Optax loop and the score estimator | 2, a solver and its estimators | `core/engine/solvers/_gradient_descent.py` | every training |
| `identify`, `IdentificationResult` | 2, an experiment over any system | `skcomplex.core.experiments` | every field with data |
| `train_policy`, `PolicyTrainingResult` | 2, an experiment over any system with a policy | `skcomplex.core.experiments` | as for `PolicyInteraction` |
| `forecast_skill`, `ForecastSkillResult` | 2, an analysis that reads series by their tags | `skcomplex.core.analysis` | climate, ecology, epidemiology, finance, fluids, and every surrogate |
| `initial`, `noise`, `horizon`, `windows`, and SDE fits of `identify` | 2, options of an experiment over any system | `core/experiments/_identification.py` | ecology and epidemiology (unknown initial populations and reporting noise), neuroscience and climate (long chaotic runs), finance and cell biology (SDE models) |
| `kind='time'` of `split_runs`, binomial noise of `ObservationSpec` | 2, options of a data function and a spec that every fit reads | `skcomplex.core.data`, `skcomplex.core.specs` | every field whose data are one run (pelts, outbreaks, climate records), and epidemiology and ecology (under-reported counts) |
| `UniversalLotkaVolterraSystem` | 3, one system | `biology/ecology/_universal_lotka_volterra.py` | biology |
| `LearnedContactSIRSystem` | 3, one system | `biology/epidemiology/_learned_contact.py` | biology |
| `CartPoleSystem` | 3, one system | `engineering/control/_cart_pole.py` | engineering |

The core never imports `skcomplex.systems` (step 5). The policy is a core interaction and not a recipe because no
core component can hold an action between decisions and expose its draws to a training (step 4). The general
`ChoiceInteraction` of `systems.general.agents` stays a recipe: it chooses by scores the user writes, and a learned
policy is not a case of it.

### Dependencies

Checked on PyPI on 2026-11-10.

| Library | Version | Released | Status |
| --- | --- | --- | --- |
| equinox | 0.13.8 | 2026-05-05 | Added, the network library of the Diffrax authors, whose modules are pytrees that `jax.grad` and Diffrax take directly |
| xarray | 2026.9.0 | 2026-09-29 | Added, the reader of labelled arrays of runs, which needs only numpy, packaging, and pandas |
| jax | 0.12.0.1 | 2026-10-21 | Present |
| optax | 0.2.8 | 2026-03-20 | Present, every optimizer |
| diffrax | 0.7.2 | 2026-02-18 | Present, every differentiable ODE solve |
| numpyro | 0.22.0 | 2026-09 | Present, the categorical and normal distributions of policies, their samples and log densities |
| pandas | 3.1.1 | 2026-11-04 | Present, the reader of data frames |
| pysindy | 2.1.0 | 2026-01 | Present, the smoothed derivatives of `method='derivatives'` |
| scikit-learn | 1.9.1 | 2026-09-10 | Present, regressors and pipelines as learned models |

Rejected: Flax, whose modules carry state outside the pytree. Haiku, no longer developed. rlax 0.1.9 (2026-06-12),
the JAX library of reinforcement learning losses, because it requires distrax, which requires `tfp-nightly`, a nightly
build as a hard dependency, and `dm_env` 1.6 of 2022-12-21, past the 24-month limit. The losses of REINFORCE, PPO, and
generalised advantage estimation are a few lines each on Optax and numpyro. stable-baselines3 2.9.2 (2026-10-26) and
skrl 2.1.1 (2026-10-21) train on torch and read Gymnasium environments only, so they are reached through the
Gymnasium bridge of the follow-up, not wrapped. brax 0.14.2 and gymnax 1.0.1 hold their own environments and cannot
step a system of this library.

### Observed data

- `ObservationSpec(paths, noise='none', scale=0.0)`. `paths` are dotted paths of states or observables, or
  `ObservableSpec` readouts such as a reported fraction of a compartment. `noise` is `'none'`, `'gaussian'`,
  `'log-normal'`, `'poisson'`, `'negative-binomial'`, or `'binomial'`, and `scale` is the standard deviation, the log
  standard deviation, the dispersion, or the reporting probability, one value or one per path. Binomial noise draws
  the reported count from the true count, rounded, with the reporting probability, the under-reporting model of
  epidemiology, and its likelihood uses the continuous extension of the binomial coefficient by the gamma function. The spec says what a sensor sees, so the same spec builds
  simulated data, scores fits, and gives the observation of an environment.
- `ObservedRunSpec(series, inputs=None, conditions=None, label='')`. `series` maps a path to its times and values, one
  column per member for a group, each path with its own times, and NaN marks a missing value. `inputs` maps parameter
  paths to times and values held between their times, the control inputs and exogenous series of the run.
  `conditions` maps model parameters, initial states, or per-member covariates to the values of this run, set through
  `set_params` before the run.
- `ObservedRunsSpec(runs, observation=None)` holds the runs and the observation model they were seen through. Its
  `_check` refuses empty runs, paths that differ between runs, and times that are not increasing.
- `read_runs(frame, *, time='time', run=None, paths=None, inputs=(), conditions=(), observation=None)` reads a pandas
  data frame in long form (one row per time and run) or wide form, or an xarray dataset with the dimension `time` and an
  optional dimension `run`. Conditions are columns or coordinates constant within a run.
- `observe(source, observation, *, times=None, replicates='each', random_state=None)` reads trajectories or the
  `ExperimentResult` of `ensemble` and `sweep`, samples them at `times` or at their frames, applies the noise of the
  observation with a seeded generator, and records the swept values of every design point as conditions. With
  `replicates='mean'` the replicates of a design point are averaged frame by frame before the noise. The result is
  labelled simulated data in every docstring and example that uses it.
- `split_runs(data, *, validation=0.2, kind='runs', random_state=None)` returns two `ObservedRunsSpec`, as
  `train_test_split` does. `kind='runs'` splits whole runs so no run leaks into both. `kind='time'` holds out the last
  share of the observation times of every run, as `TimeSeriesSplit` keeps time order, since most real data sets are
  one run, such as the pelt records and the boarding school outbreak of the probes. Its validation runs keep the
  labels, conditions, and inputs of their training runs and hold the later times only, and a fit scores them by
  running from the start of the run, so the held-out values are forecasts of the fitted system. The seed does not
  change a split by time.
- `make_arrays(data, *, kind='transitions', features=None, targets=None, lags=1, horizon=1, summary='last')` reads
  observed runs, one trajectory or several, or the `ExperimentResult` of `ensemble` or `sweep`, read as runs as
  `observe` reads them but without noise. It returns `X`, `y`, and `groups`, the index of the run of every row, which
  `GroupKFold` and the other group splitters take so no run is split across training and test rows.
  - `kind='transitions'` gives one row per run and observation time. `X` holds the `features` paths (all observed
    paths by default) at the last `lags` observation times, oldest first, and the inputs of the run at the last of
    them. `y` holds the `targets` paths (the features by default) `horizon` observation times later. So `lags=1` and
    `horizon=1` map states to next states, as the data of a learned map or a reservoir.
  - `kind='runs'` gives one row per run. `X` holds the conditions of the run, the swept parameters of a sweep, and `y`
    holds the `summary` of every target path over the run: `'last'`, `'mean'`, `'min'`, `'max'`, or a callable of
    times and values. Scalar outputs of an `ExperimentResult` are targets as they are. So a pipeline learns the map
    from parameters to outcomes, as an emulator of the system.
  - Rows with a missing value are left out, and paths with their own times enter a row only at the times they share.
  - The name is a verb, as `make_pipeline` and `make_scorer` are, and says the result is new arrays made from the
    runs.
- Every fit reads the spec. `calibrate` accepts it for `data`, its former mapping being one run. `_read_samples`
  accepts it, so `sparse_identification`, `mode_decomposition`, and `reservoir` take observed runs, with conditions as
  the parameters of a parametric library and inputs as their control inputs. `TrainingInteraction` takes it as `data`.

### Learned functions

- `learned_function(model, *inputs, weights=None)` in `components/laws/utils/_learned.py` returns one Sage expression
  per output of the model, a call of a symbolic function that holds the static part of the model in a registry keyed by
  its structure, as `series` holds its table. The inputs are expressions of the states and parameters the law reads, so
  a pair law passes the distance and a group law passes member states.
- An Equinox module is split with `equinox.partition(model, equinox.is_array)`. Its arrays become `weights`, an
  expression of one adaptive vector parameter declared `ParameterSpec(shape=(size,), adaptive=True)`, whose initial
  value `learned_weights(model)` returns with `jax.flatten_util.ravel_pytree`. The JAX backend lowers the call to
  `equinox.combine(unravel(weights), static)(inputs)[k]`, vectorised over members and pairs with `jax.vmap`, traced and
  differentiable. The NumPy backend calls the same function compiled with `jax.jit`. The Numba backend falls back to
  Python with its existing warning.
- A scikit-learn regressor, or a pipeline that ends in one, is cloned when the law is built and called through
  `predict` on the NumPy backend, once per evaluation over all members. It takes no `weights`. An unfitted regressor
  raises `ValidationError` at `prepare` unless the system is passed to `identify(method='derivatives')`, which fits
  it. The JAX backend has no form for it, so the compiler reports `not-differentiable`, the issue `calibrate` already
  reads.
- The compiler reports `learned-weights` when the weight vector does not have the length of the module, and
  `learned-inputs` when the number of inputs is not the input size of the model.
- A symbolic learned term is an ordinary expression whose coefficients are adaptive parameters, such as the laws
  `sparse_identification` returns. `identify` refines it like any weights, so no third model kind is needed.
- Weights are parameters, so `set_params`, `schedule`, trajectories, `sweep`, and `calibrate` read them like any
  parameter. A system with learned functions validates as differentiable exactly when every model is an Equinox module.

### One fit loop

- The loop of `GradientDescentSolver` becomes a private class `_OptaxLoop` of the same module, holding the optimizer,
  its state, the shuffling of batches with the run's random stream, the validation loss, and early stopping. It takes a
  function from parameters and a batch to a loss and its gradient. The solver gives it the pathwise gradient of the
  unrolled run. `train_policy` gives it the score gradient of recorded rollouts. The learned analyses of the drafts that
  build on this change give it their own losses. The engine order holds, since the experiments pass the batches as a
  function and the solver never imports the runtime.
- Observed runs replace data sets. Each run is unrolled on the union of its observation times and its input times,
  inputs held between their times, conditions set before the run, and missing values masked out of the loss. Runs with
  different times are padded and masked, so a batch compiles once.
- The loss of `TrainingInteraction` is an `ObservableSpec`, as today, or the name of a loss: `'squares'`, `'log'` (the
  squares of the logarithms, for concentrations and counts), or `'likelihood'`, the negative log-likelihood of the
  noise of the observation model. Poisson and negative binomial likelihoods use their continuous extensions in the
  predicted mean, so gradients exist.
- `validation` holds out runs with `split_runs`, records `<interaction path>/validation_loss` every epoch, and with
  `patience` stops when it has not fallen for that many epochs, keeping the best parameters.
- The Diffrax solver, tolerances, and adjoint stay those of the solver, lifted to module constants that `identify`
  and `calibrate(method='variational')` read, so a fit and a simulation agree.

### Identification

- `identify(system, data, *, trained, method='gradient', loss=None, initial=(), noise=False, horizon=None,
  windows=None, continuity=1.0, paths=16, optimizer='adam', learning_rate=1e-2, epochs=500, batch_size=None,
  validation=0.0, patience=None, random_state=None)` in `core/experiments/_identification.py`. `trained` names
  adaptive parameters, learned weights included. `loss=None` takes `'likelihood'` when the data has a noisy
  observation model and `'squares'` otherwise.
- **Initial states and noise.** `initial` names states whose initial values are fitted, one value per run, started
  from the first observed value where the state is observed and from the value of the system otherwise. `noise=True`
  fits the scale of the observation model with the likelihood loss, one value per path. Fitting the initial states
  through the run is the strong constraint 4D-Var of data assimilation, and both are what Carpenter (2018) estimates
  beside the four rates, so the probe compares like for like. The result reports them as `initial_states`, a mapping
  from run label to state path to value, and `noise_scales`, a mapping from path to scale.
- **Prediction error and multiple shooting.** Output error through a long run of a chaotic or stiff system has many
  local minima (Kidger 2202.02435, Schoukens and Ljung 1902.00683). `horizon=k` starts a fresh rollout from every
  observation time at which all states are observed and scores the next `k` observations, the k step prediction
  error, which `k=1` makes the one step prediction error of system identification. It raises `ValidationError` naming
  the unobserved states otherwise. `windows=m` cuts every run into `m` windows whose start states are trained, as
  `initial` trains the first one, and adds `continuity` times the squared gap between the end of each window and the
  start of the next, so unobserved states are allowed. `horizon=None` and `windows=None` keep the error of the whole
  run.
- **SDE fits.** `method='gradient'` accepts systems whose only stochastic laws are SDE terms. Every epoch draws
  `paths` Brownian paths per run with `diffrax.VirtualBrownianTree`, seeded from `random_state`, and the loss compares
  the mean and the variance of the paths at every observation time with those of the observed runs that share their
  conditions, so drift and diffusion are both fitted. Gradients are pathwise through Diffrax, after Kidger
  (2202.02435, ch. 4). Jump laws are still refused, and latent SDEs and SDE GANs are left for later.
- `method='gradient'` builds the training system, a `TrainingInteraction` bound to the system, simulates it with
  `GradientDescentSolver`, and reads the last or best frame. This is the one path of training, so a user who wants the
  training as a system writes the same interaction.
- `method='derivatives'` smooths the observed states with pysindy's `SmoothedFiniteDifference`, subtracts the known
  terms of every rate, and fits the scikit-learn regressors of the learned functions to the residuals, as SINDy fits a
  library. It needs every state of a learned rate observed, and raises `ValidationError` naming the unobserved state.
- `IdentificationResult` holds `system`, `estimates` (path to value, weights excluded), `initial_states`,
  `noise_scales`, `losses` and `validation_losses` (one per epoch), `training_error`, `validation_error`, and
  `trajectory`, the run of the training system. These field names are the convention every function that learns a system follows.
- `identify` is not `calibrate`. `calibrate` fits few named parameters of a known system within bounds and priors, and
  gives uncertainty. `identify` fits many unbounded adaptive parameters, learned weights included, over many runs by
  stochastic gradients, and gives a point estimate.

### Reinforcement learning

- `EnvironmentSpec(observation, actions, reward, terminal=None, interval=1.0, discount=0.99)`. `actions` maps the path
  of a bounded adaptive parameter to its bounds for a continuous action, or to a tuple of values for a choice.
  `reward` and `terminal` are observables. `interval` is the time between decisions, during which the actions are held,
  as a schedule holds a parameter.
- `PolicyInteraction(label, target, environment, model, weights=None, *, distribution='normal', scale=0.1,
  form='symbolic')`. The role `target` binds the element, group, or nested system the policy observes and controls.
  Every `interval` the policy reads the observation of its environment, evaluates `learned_function(model, ...)` on it,
  and sets the actions: logits of a numpyro `Categorical` for choices, the mean of a numpyro `Normal` of standard
  deviation `scale`, clipped to the bounds, for continuous actions, or the mean itself with `'deterministic'`. Every
  stochastic action is a named draw `<label>.action`, so `record_draws` keeps it and replay forces it. Bound to a group,
  the policy acts for every member with shared weights. `form='agent'` acts inside the step of every agent on the Mesa
  backend, through the system view.
- The runtime handles the interaction by type: it splits the run at the decision times and writes the actions as the
  schedule writes parameters, so the plant laws read them unchanged on every backend. The deterministic policy on the
  JAX backend is one more term of the traced function, so a closed loop is differentiable.
- `train_policy(system, *, method='ppo', trained=None, critic=None, episodes=16, epochs=200, horizon=None,
  gae=0.95, clip=0.2, optimizer='adam', learning_rate=3e-4, executor='serial', random_state=None)` in
  `core/experiments/_policy_training.py`. It finds every `PolicyInteraction` by type and trains their weights, or the
  paths named by `trained`.
  - `'reinforce'` runs `episodes` rollouts with recorded draws through the runner of `ensemble`, cuts each at the first
    decision where the terminal observable is nonzero or at `horizon`, and takes the gradient of the log densities of
    the recorded actions times the discounted returns minus their mean, Williams (1992).
  - `'ppo'` adds a critic, an Equinox module of the observation passed as `critic`, generalised advantage estimates
    (Schulman et al. 2015, arXiv 1506.02438), and the clipped surrogate objective of Schulman et al. (2017, arXiv
    1707.06347), with several passes over each batch of rollouts.
  - `'pathwise'` maximises the expected discounted reward through the differentiable run of a system whose tags are
    differentiable, with the deterministic policy, by `GradientDescentSolver`.
- `PolicyTrainingResult` holds `system`, `returns` (mean return per epoch), `losses`, `episodes` (the number of
  rollouts per epoch), and `trajectory` for every method, a trajectory whose times are the epochs and whose series are
  the trained weights, the mean return, and the loss, as the run of `TrainingInteraction` records an identification.
  So "the record of a training is a trajectory" holds for every method, and `lyapunov`, `rolling`, and plots read the
  training of a policy as they read any run.
- **Off-policy, designed for `add-off-policy-learning`.** A transition is read from recorded runs: the observation,
  the recorded action draw, the reward, and the next observation. The store of transitions is an `ObservedRunsSpec` of
  recorded rollouts, so data logged elsewhere trains the same way. A Q-function is a second learned function passed to
  `train_policy`, and the temporal-difference loss is one more gradient for `_OptaxLoop`. No new concept is needed.
- **Multi-agent, designed for `add-multi-agent-learning`.** Several `PolicyInteraction` components, each with its own
  environment and reward, are independent learners trained in one call. A policy bound to a group already shares its
  weights across members. A centralised critic reads the joint observation. PettingZoo exposes the agents through the
  bridges.

### Forecast skill

- `forecast_skill(forecast, truth, *, variables=None, metric='smape', threshold=30.0, baseline='persistence')` in
  `core/analysis/_forecast_skill.py` reads trajectories or arrays. It returns `errors` by lead time and variable, the
  `valid_time`, the first lead at which the error exceeds `threshold`, as in Gilpin's benchmarks of chaotic systems, and
  `skill`, one minus the ratio of the error to that of the baseline. `metric` is also `'nrmse'`. It declares
  `AnalysisTags(DataKind.SERIES)`, so `applicable_analyses` lists it.

### Probes

- **`UniversalLotkaVolterraSystem`** in `biology/ecology/_universal_lotka_volterra.py`, with `prey`, `predators`, `r`,
  `d`, `form='mechanistic'` or `'hybrid'`, `hidden=16`, and `random_state`. The hybrid form keeps the growth of hares
  and the death of lynx and replaces both predation terms by one Equinox network of the two populations with two
  outputs, after the universal differential equations of Rackauckas et al. (2020) that 2306.10335 analyses. The data
  are the hare and lynx pelts of 1900 to 1920 of the build tutorial, as distributed with Carpenter (2018), whose
  posterior means 0.55, 0.028, 0.80, and 0.024 are the reference values of the mechanistic fit. That model also
  estimates the two initial populations and the log-normal noise scale of each series, so the probe fits them with
  `initial` and `noise`, and their posterior means are read from the case study as further reference values. The
  held-out years are split with `split_runs(kind='time')`. Primitives: `rates`,
  `learned-functions`, `observed-data`, `learning`.
- **`LearnedContactSIRSystem`** in `biology/epidemiology/_learned_contact.py`, with `population=763`, `beta`, `gamma`,
  `form='constant'` or `'learned'`, and `random_state`. The learned form multiplies the contact rate by a positive
  network of time. The data are the boys confined to bed in the influenza outbreak of a boarding school in the north of
  England in 1978, reported in the British Medical Journal of that year, fetched from that note and checked against the
  copy of the R package `outbreaks`. Only the confined boys are seen, a readout of the infected compartment with
  negative binomial noise, so the probe exercises partial observation and count likelihoods. Its reference values are
  the posterior estimates of Grinsztajn et al. (2021, Statistics in Medicine) for this outbreak. Primitives: `rates`,
  `learned-functions`, `observed-data`, `learning`.
- **`CartPoleSystem`** in `engineering/control/_cart_pole.py`, the cart and pole of Barto, Sutton, and Anderson (1983)
  with the corrected equations of Florian (2007): gravity 9.8, cart mass 1.0, pole mass 0.1, half length 0.5, force 10
  as a choice of push left or right, decisions every 0.02 seconds, reward 1 per decision, and failure past 12 degrees
  or 2.4 metres, the constants of Gymnasium's `CartPole-v1`. It runs with explicit Euler steps of 0.02, as the
  reference does, so the bridges follow-up compares the two step by step. Its environment is a class attribute. A
  return of 475 averaged over 100 episodes of 500 decisions is the threshold at which Gymnasium counts the task solved.
  Primitives: `rates`, `policies`, `learned-functions`, `draws`.

### Follow-up order

1. `add-learning-bridges`: scikit-learn splitters, scorers, and parameter grids inside `identify`, `calibrate`, and
   `train_policy`, a system as a Gymnasium environment and a Gymnasium environment as a system, and the same for
   PettingZoo.
2. `add-off-policy-learning`: Q-learning with a target network, soft actor-critic, and fitted Q iteration from logged
   runs.
3. `add-multi-agent-learning`: independent learners, centralised critics, and self-play, with a probe game.

The drafts that build on this change may follow in any order after it. None needs the follow-ups.

### The data package and its layer

Decided on 2026-11-10. `skcomplex.core.data` is a new package of the core that holds `read_runs`, `observe`,
`split_runs`, and `make_arrays`, as `sklearn.datasets` holds what makes data. The specs of observed runs stay in
`skcomplex.core.specs`, since the fits read them by type.

The layers contract of `pyproject.toml` runs `experiments`, `analysis`, `engine`, `components`, `specs`, `base`, then
`types` and `errors`. `data` enters between `analysis` and `engine`, so experiments and analyses may import it and it
imports neither. It imports only `skcomplex.core.specs`, the trajectories of `skcomplex.core.engine.runtime`, and the
root modules of types and errors. `observe` and `make_arrays` read an `ExperimentResult` through its `runs`, run result
specs of `skcomplex.core.specs`, so the package never imports `skcomplex.core.experiments`. A scenario of the
observed data checks that `observe` and `make_arrays` read an `ExperimentResult` while no module of the package
imports `skcomplex.core.experiments`. A forbidden contract keeps
it from `skcomplex.core.components` and the other engine packages, so data stays a reader of results and specs.

### Decisions of 2026-11-10

- Systems are not scikit-learn estimators, and no adapter class is added, as recorded above.
- `ObservedRunsSpec` replaces `DatasetSpec`, a breaking change with no alias.
- The package `skcomplex.core.data` is approved and holds `read_runs`, `observe`, `split_runs`, and `make_arrays`.
- The subpackage `skcomplex.systems.general.environments` is approved for the bridges of `add-learning-bridges`.
- From the gap report of the learning drafts, approved the same day: `identify` fits initial states, noise scales, and
  SDE terms, and offers prediction error and multiple shooting, `split_runs` splits by time within a run,
  `PolicyTrainingResult.trajectory` exists for every method, and `ObservationSpec` gains binomial noise. Filtering,
  smoothing, and data assimilation are the follow-up `add-state-estimation`, and optimal designs and design
  optimisation the follow-up `add-experimental-design`.

## Risks / Trade-offs

- [Removing `DatasetSpec` breaks existing training code] → The Elman network, its example, and the user guide move in
  this change, and the changelog shows the two-line migration.
- [The registry of static models makes an expression depend on an object outside it] → The key comes from the model
  structure, `clone` and `set_params` keep the model, and a test pickles a system with a learned law and runs it again.
- [Python calls of scikit-learn regressors and NumPy callbacks of Equinox are slow on large groups] → One call per
  evaluation over all members, and the user guide states that JAX is the fast path.
- [Policy gradients have high variance] → PPO with a critic is the default, tests use cart-pole budgets that pass in
  seconds with a loose threshold, and the full threshold is a slow test and the gallery example.
- [Poisson and negative binomial likelihoods of continuous predictions] → The predicted mean is clipped at a small
  positive value, which the docstring states.
- [The real data of the probes are small] → The scenarios check agreement with published estimates and held-out years,
  not the accuracy of a large benchmark.
- [The mean and variance of SDE paths are noisy estimates of the loss] → `paths` sets the number of paths, the seed of
  every epoch comes from `random_state`, and the scenario fits a process whose stationary moments are known in closed
  form.
- [Multiple shooting adds trained start states that a short run may not determine] → `continuity` weights the gaps,
  and the scenario compares windows with the error of the whole run on a chaotic system where the latter fails.

## Migration Plan

`DatasetSpec({'inputs.x': X, 'outputs.target': Y}, weights=W, times=T)` becomes `ObservedRunsSpec` with one
`ObservedRunSpec` per sequence whose `inputs` hold the slices of `X` and `Y` at the times `T`, built by
`read_runs` from an xarray dataset of dimensions `run` and `time`. Loss observables keep reading the target
parameters. `DatasetSpec` keeps no alias. No other public name changes.

## Open Questions

- Torch models train outside the one Optax loop, by their own libraries. Whether that stays is the open question of
  `add-torch-surrogates`.
