# Run experiments

Sweep parameters, repeat runs, and collect their results in one table.

## Experiments

Run a system many times, over a design of parameters, under scenarios, and with replicates, and collect the results in
one table. Complex systems are studied through many runs, and `Experiment` organises them.

```python
from skcomplex.core.components.specs import ScenarioSpec
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.systems import describe

bounds = {'stochastic': [True], 'gamma': [0.1, 0.2]}
sir = describe('sir').factory
ensemble = Experiment(sir, GridDesign(), bounds, replicates=10, until=60, random_state=0).prepare().run()
round(float(ensemble.result_.summary()[0]['I.n.mean']), 1)

policies = [ScenarioSpec('baseline'), ScenarioSpec('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
harvesting = describe('harvesting').factory
stocks = Experiment(harvesting, scenarios=policies, until=80, outputs=['land/stock.n']).prepare().run()
[round(float(row['land/stock.n'])) for row in stocks.result_.rows]
```

`Experiment` runs a system, or a factory of systems, over three things:

- a design of factory arguments and dotted paths, such as `gamma` above,
- scenarios declared as `ScenarioSpec`, such as a harvest quota that starts at time 50,
- replicates.

Every design row and replicate gets its own random stream, spawned from `random_state` in a fixed order. So results are
the same serially, on threads, or on processes. Every scenario of a row and replicate shares its stream. Overrides of
parameters and initial states reuse one compiled problem, so sweeps of thousands of runs compile once.

An experiment speaks the two verbs of every object. `prepare` validates its settings and resolves the system and the
runs. `run` executes them, and keeps the results in `result_`.

The parts of an experiment are these:

- **Designs.** They are `GridDesign`, `RandomDesign`, `LatinHypercubeDesign`, `SobolDesign`, `RowsDesign`, and the
  designs of sensitivity analysis `OneAtATimeDesign`, `MorrisDesign`, and `SaltelliDesign`. A design stores its
  settings, such as `SobolDesign(n=64)`, and prepares on the `bounds` of the experiment. The bounds are a pair or a
  SciPy distribution per name for a sampled design, a sequence of values per name for a grid, and nothing for a rows
  design. `prepare` sets the rows of a design in `rows_`.
- **Outputs.** `outputs` keeps states and observables as series, after `warmup`, and functions of a trajectory as
  scalars.
- **Analyses.** `analyses` are unprepared analyses. Every run clones each one, prepares it on its trajectory, runs it,
  and keeps what its `run` sets as columns, such as `entropyanalysis.entropy`.
- **Results.** They give one row per run, `series`, `ensemble` with one row per run on common times and the seeds of the
  runs, `summary` with means, standard deviations, and quantiles across replicates, and `to_frame` for pandas.

### Calibration and sensitivity

Fit the parameters of a system to observed data, and rank which inputs matter most.

```python
import numpy as np

from skcomplex.core.experiments import CalibrationExperiment, Experiment, OneAtATimeDesign, SensitivityAnalysis
from skcomplex.systems import describe

logistic_growth = describe('logistic_growth').factory
times = np.arange(0.0, 21.0)
observed = 500 / (1 + 49 * np.exp(-0.8 * times))
bounds = {'growth.r': (0.1, 2.0), 'growth.K': (100, 1000)}
fit = CalibrationExperiment(logistic_growth, {'population.n': (times, observed)}, bounds).prepare().run()
round(fit.estimates_['growth.r'], 2), round(fit.r2_, 4)

sweep = Experiment(logistic_growth, OneAtATimeDesign(), bounds, until=20, outputs=['population.n'])
sweep = sweep.prepare().run()
SensitivityAnalysis('oat', 'population.n').prepare(sweep.result_).run().ranking_
```

- `CalibrationExperiment` fits dotted paths to observed series, by least squares or a Gaussian likelihood. It keeps the
  estimates and their covariance, with R², RMSE, MAE, and MAPE. Above, it recovers the growth rate 0.8 of the data.
- `SensitivityAnalysis` prepares on the result of an experiment and ranks its inputs one at a time, by Morris
  screening, or by Sobol indices with SALib.
