## Context

`add-learning-foundation` adds `ObservedRunsSpec`, `identify` and its `IdentificationResult`, `EnvironmentSpec`,
`PolicyInteraction`, `train_policy`, and `make_arrays` in `skcomplex.core.data`. It records the decision that systems
are not scikit-learn estimators. Every function that learns a system returns a result with the fields `system`,
`training_error`, and `validation_error`. `BaseConfig` derives from `sklearn.base.BaseEstimator`, so `get_params` and
`set_params` of systems already reach nested parameters with double underscores. The runtime runs a system step by
step through `simulations` and its clocks. `CartPoleSystem` uses the constants and Euler steps of Gymnasium's
`CartPole-v1`.

## Goals / Non-Goals

**Goals:**
- Model selection with scikit-learn splitters, scorers, and parameter grids inside `identify`, `calibrate`, and
  `train_policy`, with the scores of every fold and the selected settings in their frozen results.
- Any system with an environment as a Gymnasium or PettingZoo environment, and any such environment as a system.
- Selection on one run by splits of its times, information criteria and predictive scores of fits, residual tests,
  and the spread of learned estimates over seeds.

**Non-Goals:**
- Adapter classes with `fit` and `predict` around systems, experiments, or analyses. They are rejected below.
- Selection inside the surrogate analyses. `extend-law-discovery` selects their dictionaries on held-out runs, and a
  pipeline trains on the arrays of `make_arrays`.
- Wrapping agent libraries such as stable-baselines3 or skrl. Their agents read the Gymnasium environment directly.
- Vectorised Gymnasium environments and rendering. A later change can add them.
- JAX environments of gymnax or brax.
- Bayes factors and the evidence of nested sampling. LOO and WAIC compare Bayesian fits without them.
- Laplace approximations and Bayesian neural networks for learned weights. Ensembles over seeds come first.

## Decisions

### scikit-learn inside, not around

Decided on 2026-11-10. A system is the model being simulated, not an estimator, so scikit-learn enters inside the
library in three places, and `prepare` and `run` stay the only verbs of its classes:

- scikit-learn regressors and pipelines are learned terms of laws through `learned_function`, in the foundation.
- `make_arrays` of `skcomplex.core.data` turns runs into `X`, `y`, and `groups` for any pipeline, in the foundation.
- `identify`, `calibrate`, and `train_policy` accept splitters, scorers, and parameter grids, in this change.

Alternatives considered:

- Adapter classes `IdentificationEstimator` and `SurrogateEstimator`, with `fit`, `predict`, and `score` around
  `identify` and the surrogate analyses, so `cross_validate`, `GridSearchCV`, and `Pipeline` take them. Rejected: they
  are a second way of doing what `identify` and the analyses do, and a system is the model being simulated, not an
  estimator.
- `fit` and `predict` on systems. Rejected for the same reasons, and because every class would carry a second verb
  pair.

The accepted cost: a fitted system cannot sit inside a scikit-learn `Pipeline` or `GridSearchCV`. The search runs
inside the experiment instead, and its result holds the selected system.

The selection is specified here and not in the foundation. It adds arguments and result fields to three experiments,
and `identify` works without it, so the foundation stays smaller.

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `cv`, `groups`, `scoring`, `param_grid`, `split`, and the selection fields of the results | 2, arguments of experiments over any system | `core/experiments/_identification.py`, `_calibration.py`, `_policy_training.py`, and the private `_selection.py` they share | every field with data or a policy |
| `aic`, `bic`, `loo`, `waic`, `residuals`, `replicates` | 2, fields and options of the results of experiments over any system | `core/experiments/_identification.py`, `_calibration.py` | ecology and epidemiology (rival mechanistic and hybrid models), systems biology (model selection of Raman 6.4), engineering (validation of identified plants), climate and finance (forecast models) |
| `whiteness`, `WhitenessResult` | 2, an analysis that reads series by their tags | `skcomplex.core.analysis` | engineering (residual analysis of identified plants), econometrics (residual autocorrelation), epidemiology and ecology (checks of fitted models), climate (forecast errors) |
| `make_environment`, `make_parallel_environment` | 2, the runtime steps a system through its `EnvironmentSpec` | `skcomplex.core.engine.runtime` | engineering, epidemiology, ecology, economics, social systems, computer science |
| `GymnasiumSystem`, `PettingZooSystem` | 3, recipes of an element and an update law with a `CallSpec`, used by several fields | `skcomplex.systems.general.environments`, approved on 2026-11-10 | as above |

### Dependencies

Checked on PyPI on 2026-11-10. Gymnasium 1.4.0, released 2026-10-05, needs numpy, cloudpickle, typing-extensions, and
farama-notifications. PettingZoo 1.27.0, released 2026-08-13, needs numpy, gymnasium, and typing-extensions, and
supports Python up to 3.14. Both are runtime dependencies, since the runtime of the core imports them.

ArviZ 1.3.0, released 2026-08-11 with its parts `arviz-base`, `arviz-stats`, and `arviz-plots` of the same day, needs
numpy, scipy, xarray (a runtime dependency since the foundation), and lazy_loader, and requires Python 3.12, the lower
bound of the project. `arviz_stats.loo` gives the Pareto smoothed importance sampling leave-one-out score with its
Pareto shape diagnostics. ArviZ 1.x dropped WAIC in favour of LOO, so WAIC is computed from the same pointwise
log-likelihoods by its formula, the log pointwise predictive density minus the variance of the log-likelihoods
(Watanabe 2010), a few lines with no library. statsmodels 0.15.0, released 2026-08-27, BSD-3-Clause, needs numpy,
scipy, pandas, patsy, packaging, and formulaic, and gives `acorr_ljungbox`. `extend-network-and-emergence-measures`
uses it for Granger tests, and whichever change lands first adds it. All are runtime dependencies, since experiments
and analyses of the core import them. Their extras for
Atari, Box2D, and MuJoCo are not installed. scikit-learn 1.9.1, released 2026-09-10, is already a runtime dependency
and gives the splitters, `ParameterGrid`, `ParameterSampler`, and the scorers.

### Model selection

- `identify` and `calibrate` gain `cv=None`, `groups=None`, `scoring=None`, and `param_grid=None`.
  - `cv` is a number of folds, read as `KFold`, or any scikit-learn splitter, such as `KFold`, `ShuffleSplit`,
    `GroupKFold`, or `LeaveOneGroupOut`. It splits the runs of the `ObservedRunsSpec`, so no run is in both parts of a
    fold.
  - `groups` holds one label per run, or names a condition of the runs whose values are the labels, such as the
    patient of a clinical series. It is passed to the `split` of the splitter.
  - `scoring` is the name of a scikit-learn regression scorer, such as `'r2'` or `'neg_mean_squared_error'`, or a
    callable `score(y_true, y_pred)` such as a metric of `sklearn.metrics`, greater being better. The fitted system is
    simulated with the conditions and inputs of every test run, and the observed values are compared with the
    simulated ones at the observation times, missing values left out. Names are the regression names of
    `sklearn.metrics.get_scorer_names`, each read as its metric with the sign scikit-learn gives it. `None` scores
    minus the loss of the fit.
  - `param_grid` is a mapping of lists or a list of them, expanded by `ParameterGrid`, or a `ParameterSampler` for a
    random search. Keys are keyword arguments of the experiment, such as `learning_rate`, `epochs`, or `method`, or
    parameters of the system as `get_params` names them, set through `set_params` as starting values.
  - With `param_grid` and no `cv`, five folds are used, as `GridSearchCV` does. With `cv` and no `param_grid`, the one
    setting given is scored, as `cross_validate` does.
  - After the search the experiment fits all the runs with the best setting, so the result is the one a plain call
    with those settings returns, plus `cv_results`, `best_params`, and `best_score`.
- `cv_results` is a mapping of arrays with the keys of `GridSearchCV.cv_results_` that apply: `params`,
  `split<k>_test_score`, `mean_test_score`, `std_test_score`, `rank_test_score`, and `mean_fit_time`. Without a
  selection the three fields are `None`.
- `calibrate` scores the point estimate of every fold, the optimum for SciPy and Optuna and the posterior mean for
  pyabc and numpyro.
- `train_policy` gains `conditions=None`, a sequence of mappings of parameters or initial states, and its episodes
  start from them in turn. `cv` splits the conditions, `groups` labels them, and `scoring` is `'return'`, the mean
  return of evaluation episodes started from the test conditions, or a callable of those returns. `evaluation=20` sets
  the number of evaluation episodes per test condition. A selection without `conditions` raises `ValidationError`,
  since there is nothing to hold out.
- Folds run on the executor of the experiment, so the seeds of every fold come from its `random_state` and a search is
  reproducible.

### Splits by time

- `identify` and `calibrate` gain `split='runs'`, the default, or `'time'`. With `'time'`, `cv` must keep the order of
  times, as `TimeSeriesSplit` does, or be a number of folds read as `TimeSeriesSplit`, and it splits the observation
  times of every run. Every test part is scored by running the fitted system from the start of its run, as
  `split_runs(kind='time')` of the foundation scores its validation runs. A splitter that shuffles, such as `KFold`
  with `shuffle=True`, raises `ValidationError` with `'time'`, since later times would train a forecast of earlier
  ones.

### Information criteria and predictive scores

- `IdentificationResult` and `CalibrationResult` gain `log_likelihood`, the maximised log-likelihood of the observation
  model at the estimates, `aic = 2 k - 2 log L` and `bic = k log n - 2 log L`, with `k` the number of fitted values,
  learned weights, initial states, and noise scales included, and `n` the number of observed values that are not
  missing. A fit whose observation model has no noise reports them for Gaussian noise of the scale of the residuals,
  as least squares implies, and says so in the docstring. Raman (6.4.1) and Brunton and Kutz (4.7) rank rival models
  this way, and the SINDy literature selects sparse models by AIC.
- `calibrate(method='mcmc')` records the pointwise log-likelihood of every observed value for every posterior draw,
  and its result gains `loo` and `waic`, each with the expected log pointwise predictive density, its standard error,
  and the effective number of parameters, and `loo` with the largest Pareto shape. Other methods report `None`.

### Residual tests

- Both results gain `residuals`, an `ObservedRunsSpec` of the observed minus the fitted values at the observation
  times, the one step prediction errors when the fit used `horizon=1` and the output errors otherwise.
- `whiteness(source, *, inputs=None, lags=20, level=0.05)` in `core/analysis/_whiteness.py` reads residuals as
  observed runs, a trajectory, or an array. It returns a frozen `WhitenessResult` with the Ljung-Box statistic and
  p-value of every path from `statsmodels.stats.diagnostic.acorr_ljungbox`, pooled over runs by summing the
  statistics and their degrees of freedom, the cross-correlations of every residual with every input up to `lags` with
  the bounds `1.96 / sqrt(n)`, and the flags `white` and `independent`. It declares `AnalysisTags(DataKind.SERIES)`.
  Schoukens and Ljung (1902.00683) validate an identified model by exactly these two tests.

### Ensembles of fits

- `identify` gains `replicates=1`. With more, it fits that many times from seeds drawn from `random_state`, on the
  executor of the experiment, and its result gains `replicate_systems`, the fitted systems, and `spread`, the standard
  deviation of every estimate over them. `system` stays the fit of the first seed, so one call with `replicates=1`
  returns what it did. The runs of the replicate systems form an ensemble forecast, scored by the probabilistic scores
  of `forecast_skill` that `add-state-estimation` adds.

### Gymnasium

- `make_environment(system, *, environment=None, until=None, solver=None, backend=None, random_state=None)` takes the
  environment of the system, its class attribute or the argument. The observation space is a `Box` of the observation
  paths, bounded by the domains of their states. The action space is a `Box` of the continuous actions or a `Discrete`
  of the choices, and several choices give a `MultiDiscrete`.
- `reset(seed=...)` prepares a fresh run from the initial states. `step(action)` writes the actions as the policy
  would, advances one interval through the stepwise runtime, and returns the observation with its noise, the reward,
  `terminated` from the terminal observable, and `truncated` at `until`.
- A system that holds a `PolicyInteraction` for those actions is refused, since an external agent replaces it.

### PettingZoo

- `make_parallel_environment(system, *, environments=None, ...)` makes one agent per `PolicyInteraction` of the system,
  or one agent per member when a policy is bound to a group, named by the policy label and the member index. Each agent
  has the spaces of its environment and its own reward. The external agents replace the policies of the system.

### Environments as systems

- `GymnasiumSystem(environment, label='system')` takes an environment or its identifier. It holds one element with the
  observation as a vector state, the last reward, and the terminated flag, and one discrete-time update law with a
  `CallSpec` that steps the environment with the current action. The action is a bounded adaptive parameter, or a
  choice of the indices of a `Discrete` space, so a `PolicyInteraction` sets it. Its class attribute `environment`
  reads the observation, the reward, and the flag.
- `PettingZooSystem(environment, label='system')` holds one element per agent with the same states and one update law
  that steps every agent at once.
- They are recipes of core parts used by several fields, in the new concept subpackage
  `systems.general.environments`, approved on 2026-11-10 and named by the concept of reinforcement learning it
  bridges.

## Risks / Trade-offs

- [AIC and BIC of learned weights count every weight, which penalises networks heavily] → The docstring says the
  criteria compare models of few parameters, and the probe compares mechanistic and hybrid forms by a time split as
  well.
- [Pointwise log-likelihoods of long series take memory] → They are kept as float32 arrays of draws by values, and
  `loo` and `waic` are computed once and the arrays dropped.

- [A fitted system cannot enter a scikit-learn `Pipeline` or `GridSearchCV`] → The search runs inside the experiment,
  `make_arrays` serves every pipeline, and the user guide shows both.
- [A search multiplies the cost of a fit by the folds and the settings] → Folds run on the executor of the experiment,
  and the user guide shows a small grid.
- [Gymnasium seeds and the run's random stream differ] → `reset(seed=...)` seeds the run's stream, and the test checks
  that two resets with one seed give equal episodes.
- [External environments are not differentiable] → `GymnasiumSystem` and `PettingZooSystem` declare stochastic,
  non-differentiable tags, so `train_policy(method='pathwise')` refuses them before running.
