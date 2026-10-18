# Design

## Context

`Simulation(system, random_state=...)` runs one model, and trajectories hold states, agents, and, after
`add-general-model-interface`, observables. Library models are factories with default run settings. See proposal.md
for the references that motivate experiments.

## Goals / Non-Goals

**Goals:**

- Any model, of any form, in ensembles, sweeps, and scenarios through one API.
- Reproducible results: the same seed gives the same experiment whatever the number of worker processes.

**Non-Goals:**

- Distributed execution beyond one machine. The runner takes a `concurrent.futures` executor, so a cluster executor
  can be passed later.
- Optimisation of policies (Bala et al. 2017 §10.6). Calibration covers fitting, and policy optimisation can reuse it.

## Decisions

### Model sources

An experiment takes a model source: a registered model name, a factory, or a system. Factories receive design values
for their parameters, and systems receive them as overrides of parameter and initial-state paths, such as
`hunt.a` or `rabbits.n`, through `set_params` on a clone.

### Seeds

The experiment's `random_state` spawns one `SeedSequence` child per run in a fixed order over designs, scenarios, and
replicates. Each run receives its own generator whatever process executes it, so results do not depend on scheduling.

### Designs

`Grid(**values)`, `RandomDesign(n, **distributions)`, `LatinHypercube(n, **bounds)`, and `SobolDesign(n, **bounds)`
produce rows of parameter values. Bounds accept a pair or a SciPy distribution, and a design concatenates with `+` and
crosses with `*`.

### Scenarios

`Scenario(name, overrides=None, schedule=None)` applies overrides before the run and `schedule`, a list of
`(time, path, value)` changes, during it, through the live parameter mechanism. A system can be swapped too, which
covers structural policy changes.

### Runner and results

`Experiment(source, design=None, scenarios=None, replicates=1, until=None, dt=None, outputs=None, warmup=0,
random_state=None, executor=None).run()` returns an `ExperimentResult`. `outputs` names the states and observables to
keep, defaulting to every observable, and a callable output reduces a trajectory to scalars. Results hold one tidy
table of scalar outputs with columns for the design values, scenario, replicate, and seed, plus the kept time series as
arrays indexed by run. `summary()` gives the mean, standard deviation, and quantiles across replicates, and
`to_frame()` returns a pandas frame when pandas is installed.

### Sensitivity and calibration

`sensitivity(experiment_like, method='morris' | 'sobol' | 'oat', bounds, output)` builds the design with SALib, runs
it, and returns indices with confidence intervals. `calibrate(source, data, parameters, bounds, loss='squares' |
'likelihood')` matches observed series to output paths and fits with `scipy.optimize.least_squares` or `minimize`,
running ensembles when the model is stochastic, and returns the estimates, their covariance where defined, and R²,
RMSE, MAE, and MAPE as in Bala et al. (2017 ch. 6).

## Risks / Trade-offs

- [Memory of kept time series] → Only declared outputs are kept, and thinning by frame reduces them further.
- [Process start-up cost] → Runs are batched per worker, and a thread or serial executor is the default for small
  experiments.
- [Stochastic calibration is noisy] → The ensemble size per evaluation is a parameter, and the loss averages it.
