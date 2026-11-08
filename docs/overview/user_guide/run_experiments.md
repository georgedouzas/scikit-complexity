# Run experiments

Sweep parameters, repeat runs, and collect their results in one table.

## Experiments

Run a system many times, over a design of parameters, under scenarios, and with replicates, and collect the results in
one table. Complex systems are studied through many runs, and the experiment functions organise them, as the model
selection functions of scikit-learn run a model over folds and parameters.

```python
from skcomplex.core.components.specs import ScenarioSpec
from skcomplex.core.experiments import compare_scenarios, ensemble, sweep
from skcomplex.systems import all_systems

SIRSystem = dict(all_systems(field='biology/epidemiology'))['SIRSystem']
replicated = ensemble(SIRSystem(stochastic=True), replicates=10, until=60, random_state=0)
round(float(replicated.summary()[0]['I.n.mean']), 1)

swept = sweep(SIRSystem(), {'gamma': [0.1, 0.2]}, until=60, outputs=['I.n'])
[row['gamma'] for row in swept.rows]

HarvestingSystem = dict(all_systems(field='economics/resources'))['HarvestingSystem']
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

from skcomplex.core.experiments import Calibration, sensitivity
from skcomplex.systems import all_systems

LogisticGrowthSystem = dict(all_systems(field='biology/ecology'))['LogisticGrowthSystem']
times = np.arange(0.0, 21.0)
observed = 500 / (1 + 49 * np.exp(-0.8 * times))
bounds = {'r': (0.1, 2.0), 'K': (100, 1000)}
fit = Calibration(LogisticGrowthSystem(), {'population.n': (times, observed)}, bounds).prepare().run()
round(fit.estimates_['r'], 2), round(fit.r2_, 4), round(fit.system_.r, 2)

ranked = sensitivity(LogisticGrowthSystem(), bounds, output='population.n', method='oat', until=20)
ranked.ranking
```

- `Calibration` is the one experiment that stays an estimator, in the role of `GridSearchCV`. It fits model
  parameters to observed series, by least squares or a Gaussian likelihood, and `run` sets `system_`, the system with
  the fitted parameters. It keeps the estimates and their covariance, with R², RMSE, MAE, and MAPE. Above, it recovers
  the growth rate 0.8 of the data.
- `sensitivity` runs a design of its own and ranks the inputs one at a time with `method='oat'`, by Morris screening
  with `'morris'`, or by Sobol indices with SALib with `'sobol'`.
