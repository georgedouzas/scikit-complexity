## Why

`add-learning-foundation` makes identification, learned laws, and policy training work inside the library. The methods
and agents most users already have live in three ecosystems outside it. scikit-learn holds cross-validation, scorers,
and hyperparameter searches. Gymnasium is the interface of single-agent reinforcement learning, which agents of
stable-baselines3, skrl, and many papers read. PettingZoo is the same interface for many agents. The experiments of the
library should select their settings with the tools of scikit-learn, and compare and validate fitted models as the
literature of system identification does, by information criteria, predictive scores, and tests of the residuals
(Schoukens and Ljung 1902.00683, Raman 6.4.1, Brunton and Kutz 4.6 and 4.7), a system of the library should be usable as a
Gymnasium or PettingZoo environment, and an environment of those ecosystems should be usable as a system of the
library, so its policies train against it with `train_policy`.

Systems are not scikit-learn estimators. A system is the model being simulated, and `prepare` and `run` stay the only
verbs of the classes of the library, so scikit-learn enters inside the experiments, not around the systems.

## What Changes

- **Model selection inside the experiments.** `identify`, `calibrate`, and `train_policy` accept a scikit-learn
  splitter or a number of folds as `cv`, with `groups` for group splitters such as `GroupKFold`, a scorer as
  `scoring`, and a parameter grid as `param_grid`. `identify` and `calibrate` split observed runs, and `train_policy`
  splits the conditions its episodes start from, given by its new argument `conditions`. Every setting of the grid is
  fitted on the training part of every fold and scored on its test part, the result is fitted again on all the data
  with the best setting, and the frozen result reports `cv_results` with the score of every fold, `best_params`, and
  `best_score`.
- **Splits by time.** `identify` and `calibrate` gain `split='runs'` or `'time'`. With `'time'` the splitter, such as
  `TimeSeriesSplit`, splits the observation times of every run, so a selection works on one run, such as the pelt
  records and the boarding school outbreak of the foundation's probes.
- **Information criteria and predictive scores.** `IdentificationResult` and `CalibrationResult` gain `aic` and `bic`,
  from the maximised log-likelihood of the observation model and the number of fitted values. `calibrate(method='mcmc')`
  gains `loo`, the Pareto smoothed importance sampling leave-one-out score of ArviZ, and `waic`, from the pointwise
  log-likelihoods that numpyro returns.
- **Residual tests.** Both results gain `residuals`, observed minus fitted values as observed runs. New analysis
  `whiteness(source, *, inputs=None, lags=20)` tests that residuals are white by the Ljung-Box test and independent of
  the inputs by their cross-correlations, the validation of Schoukens and Ljung (1902.00683).
- **Ensembles of fits.** `identify` gains `replicates`, the number of fits from different seeds, and its result holds
  the fitted systems and the spread of their estimates, the deep ensembles that Cheng et al. (2303.10462, sect. IV)
  review as the uncertainty of learned models.
- New function `make_environment(system, ...)` in `skcomplex.core.engine.runtime`, which returns a Gymnasium
  environment that steps the system through its `EnvironmentSpec`, so external agents run against any system.
- New function `make_parallel_environment(system, ...)` beside it, which returns a PettingZoo parallel environment
  with one agent per policy or per member of a group.
- New systems `GymnasiumSystem` and `PettingZooSystem`, which wrap an external environment as a system whose actions
  are adaptive parameters and whose observations are states, so `PolicyInteraction` and `train_policy` act on it. They
  are recipes of core components in the new concept subpackage `skcomplex.systems.general.environments`, approved on
  2026-11-10.
- Dependencies: Gymnasium 1.4.0, PettingZoo 1.27.0, ArviZ 1.3.0, and statsmodels 0.15.0 become runtime dependencies.

## Capabilities

### New Capabilities

- `learning-bridges`: systems as Gymnasium and PettingZoo environments, and those environments as systems.

### Modified Capabilities

- `experiments`: cross-validation over runs and over times, scorers, and parameter grids in `identify` and
  `calibrate`, information criteria, predictive scores, residuals, and ensembles of fits.
- `trajectory-analysis`: the whiteness and independence tests of residuals.
- `reinforcement-learning`: cross-validation over conditions, scorers of returns, and parameter grids in
  `train_policy`.
- `model-library`: the cart-pole system agrees step by step with Gymnasium's `CartPole-v1`.

## Impact

- Core: `core/experiments/_identification.py`, `_calibration.py`, and `_policy_training.py` gain the selection, with
  a private module `core/experiments/_selection.py` that the three share, the two fits gain criteria, residuals, and
  replicates, and new modules `core/analysis/_whiteness.py` and `core/engine/runtime/_environments.py`.
- Systems: the new concept subpackage `skcomplex.systems.general.environments` and a test of
  `engineering/control/_cart_pole.py`.
- Dependencies: `gymnasium>=1.4.0` (1.4.0, released 2026-10-05), `pettingzoo>=1.27.0` (1.27.0, released
  2026-08-13), `arviz>=1.3.0` (1.3.0, released 2026-08-11), and `statsmodels>=0.15.0` (0.15.0, released 2026-08-27).
  scikit-learn is already a runtime dependency.
- Docs: sections of the calibrate and experiments pages of the user guide and a gallery example that trains a policy
  of the library on a Gymnasium environment and runs a Gymnasium agent on a system.
- Depends on `add-learning-foundation`. Before `add-off-policy-learning` and `add-multi-agent-learning`.
