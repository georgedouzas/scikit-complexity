# Proposal

## Why

Every reference studies complex systems through many runs, not one. Thurner et al. (2018 §1.5.2) call for "ensembles
of histories", Caiani et al. (2016 §2.1.5-2.2.5) wrap simulations in Monte Carlo loops with independent seeds,
sensitivity analysis, and policy experiments, Bala et al. (2017 ch. 5-7) calibrate system-dynamics models to data and
compare policy scenarios, Chesney (2021 ch. 9) runs full-factorial designs with statistical tests, and Datseris and
Parlitz (2022 ch. 4, 12) build bifurcation diagrams and basin stability from sweeps over parameters and initial
conditions. The framework runs one simulation at a time, so all of this is hand-written loops today.

## What Changes

- Add ensembles: replicates of one model with independent random streams spawned from one seed, run in parallel
  processes.
- Add parameter designs: grids, random, Latin hypercube, and Sobol samples over parameters and initial states.
- Add scenarios: named overrides of parameters and initial states, and scheduled parameter changes at given times, for
  baseline and policy comparisons.
- Add an experiment runner that crosses a model factory with a design, scenarios, and replicates, discards a warm-up,
  and returns tidy results with summary statistics across replicates.
- Add sensitivity analysis: one-at-a-time, Morris elementary effects, and Sobol indices of chosen outputs.
- Add calibration: fit parameters to observed time series by least squares or likelihood, with fit statistics.

## Capabilities

### New Capabilities

- `experiments`: ensembles, designs, scenarios, the experiment runner, sensitivity analysis, and calibration.

### Modified Capabilities

None.

## Impact

- New package `skcomplex.experiments` between `runtime` and `analysis` in the layer order, which the constitution's
  Project Profile records.
- New dependency `SALib` for Morris and Sobol designs and indices.
- Depends on `add-general-model-interface` for factories and observables as experiment outputs.
