# Run experiments

Sweep parameters, repeat runs, and collect their results in one table.

## Experiments

Run a system many times, over a design of parameters, under scenarios, and with replicates, and collect the results in
one table. Complex systems are studied through many runs, and the experiment functions organise them, as the model
selection functions of scikit-learn run a model over folds and parameters.

```python
from skcomplex.core.components.specs import ScenarioSpec
from skcomplex.core.experiments import compare_scenarios, ensemble, sweep
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.economics.resources import HarvestingSystem

replicated = ensemble(SIRSystem(stochastic=True), replicates=10, until=60, random_state=0)
round(float(replicated.get_summary()[0]['I.x.mean']), 1)

swept = sweep(SIRSystem(), {'gamma': [0.1, 0.2]}, until=60, outputs=['I.x'])
[row['gamma'] for row in swept.rows]

policies = [ScenarioSpec('baseline'), ScenarioSpec('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
stocks = compare_scenarios(HarvestingSystem(), policies, until=80, outputs=['land/stock.n'])
[round(float(row['land/stock.n'])) for row in stocks.rows]
```

Every experiment takes the system as its first argument, a system instance whose copies take model parameters through
`set_params`, or a factory of systems, such as a system class. The experiments are these:

- `sweep` runs a design of model parameters and dotted paths, such as `gamma` above,
- `ensemble` repeats the runs of a stochastic system, and raises a `TagError` for a deterministic one, which has
  nothing to repeat,
- `compare_scenarios` runs scenarios declared as `ScenarioSpec`, such as a harvest quota that starts at time 50,
- `basin_stability` and `orbit_diagram` run a system over initial conditions or the values of one parameter, as the
  page [Analyse](analyse.md) shows.

A name with a dot is an initial state or a parameter of a part, which keeps its value for the whole run. Every run goes
through `simulate`, so the tuned run settings of the system fill what a call leaves out.

Every design row and replicate gets its own random stream, spawned from `random_state` in a fixed order. So results are
the same serially, on threads, or on processes, which `executor` chooses. Every scenario of a row and replicate shares
its stream. Runs that change only dotted paths, initial states, or seeds reuse one compiled problem, so sweeps of
thousands of runs compile once per model parameter.

The settings of an experiment are these:

- **Designs.** `sweep` takes a grid of values per name by default, or `design='random'`, `'latin_hypercube'`, or
  `'sobol'` with `n` rows, from a pair of bounds or a SciPy distribution per name. A sequence of mappings is a design
  of given rows.
- **Outputs.** `outputs` keeps states and observables as series, after `warmup`, and functions of a trajectory as
  scalars.
- **Analyses.** `analyses` are analysis functions, or functions with fixed settings from `bind_analysis`. Every run
  applies each one to its trajectory, and keeps the fields of its result as columns, such as `entropy.entropy`.
- **Results.** An `ExperimentResult` is a frozen value. It gives one row per run in `rows`, `column` and `series` by
  name, `ensemble` with one row per run on common times and the seeds of the runs, `summary` with means, standard
  deviations, and quantiles across replicates, and `to_frame` for pandas.

### Calibration and sensitivity

Fit the parameters of a system to observed data, and rank which inputs matter most.

```python
import numpy as np

from skcomplex.core.experiments import calibrate, sensitivity
from skcomplex.systems.biology.ecology import LogisticGrowthSystem

times = np.arange(0.0, 21.0)
observed = 500 / (1 + 49 * np.exp(-0.8 * times))
bounds = {'r': (0.1, 2.0), 'K': (100, 1000)}
fit = calibrate(LogisticGrowthSystem(), {'population.x': (times, observed)}, bounds)
round(fit.estimates['r'], 2), round(fit.r2, 4), round(fit.system.r, 2)

ranked = sensitivity(LogisticGrowthSystem(), bounds, output='population.x', method='oat', until=20)
ranked.ranking
```

- `calibrate` fits model parameters to observed series, by least squares or a Gaussian likelihood, in the role of
  `GridSearchCV`. It returns a frozen `CalibrationResult` whose `system` is the system with the fitted parameters,
  with the estimates and their covariance, and R², RMSE, MAE, and MAPE. Above, it recovers the growth rate 0.8 of the
  data.
- `sensitivity` runs a design of its own and ranks the inputs one at a time with `method='oat'`, by Morris screening
  with `'morris'`, or by Sobol indices with SALib with `'sobol'`.

### Bayesian calibration

Get posterior uncertainty for the parameters of any system, deterministic or stochastic, with checks that it can be
trusted. `method` says how `calibrate` searches or samples, and `loss` what scores a candidate.

```python
import numpy as np
from scipy import stats

from skcomplex.core.experiments import calibrate
from skcomplex.systems.biology.ecology import LogisticGrowthSystem

rng = np.random.default_rng(0)
times = np.arange(0.0, 21.0)
observed = 500 / (1 + 49 * np.exp(-0.8 * times)) * (1 + 0.02 * rng.standard_normal(times.size))
bounds = {'r': stats.norm(0.8, 0.2), 'K': (100, 1000)}
posterior = calibrate(
    LogisticGrowthSystem(n=10.0),
    {'population.x': (times, observed)},
    bounds,
    method='mcmc',
    loss='likelihood',
    n_samples=300,
    random_state=0,
)
round(posterior.estimates['r'], 2), posterior.diagnostics['sampler'], posterior.predictive['coverage'] > 0.75
```

- Methods: `'local'`, the default, is a local SciPy fit, `'global'` searches the bounds with Optuna's seeded
  tree-structured Parzen estimator for `budget` evaluations, `'abc_smc'` samples an approximate posterior by ABC-SMC
  with pyabc, `'mcmc'` samples the posterior of a likelihood with numpyro, and `'variational'` fits a multivariate
  normal posterior with numpyro SVI through differentiable runs on JAX.
- Losses: `'squares'`, `'likelihood'`, the Gaussian likelihood with an unknown variance, and `'events'`, the
  likelihood of recorded draws. `'local'` and `'global'` take all three, `'abc_smc'` compares summaries, `'mcmc'` takes
  `'likelihood'` and `'events'`, and `'variational'` takes `'likelihood'` and `'squares'`.
- Priors: a bound is a pair `(low, high)`, read as a uniform prior, or a SciPy distribution, read as the prior. Point
  fits search the pair or the central 99.9% of the distribution. MCMC and variational calibration take the uniform,
  normal, log-normal, truncated normal, beta, gamma, and exponential families.
- Summaries: `summary=` maps the series by path to summary statistics, such as final counts or peak times, which
  ABC-SMC compares by an adaptive distance. Without it ABC-SMC compares the series at the data times.
- Events: `loss='events'` takes as data a trajectory recorded with `simulate(..., record_draws=True)` and scores
  every candidate by the log-probability of the recorded draws replayed under it, the exact likelihood of a
  stochastic model with no model-specific code, as the page [Simulate](simulate.md) shows. Pass the seed of the
  recorded run as `random_state`.
- Diagnostics: a posterior result holds `samples`, `weights`, and the posterior medians as `estimates`.
  `diagnostics` holds the tolerance and the acceptance rate of every generation of ABC-SMC, the split R-hat and the
  effective sample size of every parameter of MCMC, and the loss history of variational calibration. `predictive`
  holds the 5%, 50%, and 95% quantiles of the series of posterior draws at the data times and the share of observed
  points inside the band.
- MCMC runs NUTS through the JAX run of a system with the tag `differentiable` and the Gaussian likelihood, and the
  gradient-free affine invariant ensemble sampler otherwise, which is slower and mixes poorly in many dimensions.

Check that a posterior method gives credible intervals that can be trusted with `posterior_coverage`. It draws true
parameters from the priors, simulates data at each with Gaussian noise, calibrates every data set, and reports how
often the central interval of every level holds the truth, with the rank of every truth among its samples, the check
of simulation-based calibration. A calibrated method covers the truth at about the rate of the level, and its ranks
are uniform.

```python
import numpy as np

from skcomplex.core.experiments import posterior_coverage
from skcomplex.systems.biology.ecology import LogisticGrowthSystem

coverage = posterior_coverage(
    LogisticGrowthSystem(n=10.0),
    {'r': (0.5, 1.0)},
    np.arange(0.0, 11.0),
    method='variational',
    n=4,
    levels=(0.5, 0.9),
    noise=5.0,
    n_samples=100,
    budget=300,
    random_state=0,
)
coverage.coverage[0.9]['r'], coverage.ranks['r'].shape
```
