# Spec Delta

## Purpose

Lets a user describe what was observed of a system, which states or observables and with what noise, hold observed
runs with irregular times, missing values, inputs, and conditions, read them from pandas and xarray, simulate them from
trajectories, and split them into training and validation runs, so every fit of the library reads one kind of data.

## ADDED Requirements

### Requirement: Observation models say what is seen and with what noise
`ObservationSpec(paths, noise, scale)` SHALL name the dotted paths of the states or observables that are seen, or
observable readouts of them, and SHALL take `noise` among `'none'`, `'gaussian'`, `'log-normal'`, `'poisson'`, and
`'negative-binomial'`, with `scale` as one value or one value per path. It SHALL refuse an unknown noise, a negative
scale, and a scale given for noise that has none.

#### Scenario: Unknown noise
- **WHEN** an observation is declared with `noise='laplace'`
- **THEN** `prepare` raises `ValidationError` that names the noise and lists the five allowed ones

#### Scenario: Readout of a compartment
- **WHEN** an observation names the observable `0.8 * infected.x` of an SIR system with Poisson noise
- **THEN** `observe` returns counts whose mean over 10000 draws at every time is within 2% of 0.8 times the infected
  states of the run

### Requirement: Observed runs hold irregular, partial, and conditioned data
`ObservedRunSpec(series, inputs, conditions)` SHALL hold, for one run, the times and values of every observed path with
its own times and NaN for missing values, the times and values of parameter inputs held between their times, and the
conditions set before the run. `ObservedRunsSpec(runs, observation)` SHALL hold several runs and the observation model,
and SHALL refuse no runs, paths that differ between runs, and times that do not increase.

#### Scenario: Paths with their own times
- **WHEN** a run observes `prey.x` at 21 yearly times and `predators.x` at 7 of them
- **THEN** the spec keeps both sets of times, and a fit compares each path only at its own times

#### Scenario: Times that decrease
- **WHEN** a run gives the times `0, 2, 1` for a path
- **THEN** `prepare` raises `ValidationError` that names the run, the path, and the first time out of order

### Requirement: Runs are read from pandas and xarray
`read_runs(frame, ...)` SHALL read observed runs from a pandas data frame in long or wide form, or from an xarray
dataset with the dimension `time` and an optional dimension `run`, with columns or coordinates named as paths, inputs,
and conditions.

#### Scenario: Long data frame
- **WHEN** a data frame has the columns `run`, `time`, `prey.x`, and `predators.x` for three runs of different lengths
- **THEN** `read_runs(frame, run='run')` returns three runs whose times and values equal the rows of each run

#### Scenario: Dataset with conditions
- **WHEN** an xarray dataset has the dimensions `run` and `time` and the coordinate `r` along `run`
- **THEN** `read_runs(dataset, conditions=('r',))` sets `r` as the condition of every run

#### Scenario: Missing column
- **WHEN** `paths` names a column the frame does not have
- **THEN** it raises `ValidationError` that names the column and lists the columns of the frame

### Requirement: Simulated observations come from trajectories and experiments
`observe(source, observation, ...)` SHALL sample trajectories, or the runs of an `ExperimentResult`, at given times or
at their frames, apply the noise of the observation with a seeded generator, and record the swept values of every
design point as conditions, keeping every replicate or their mean.

#### Scenario: Same seed
- **WHEN** `observe` runs twice on one trajectory with Gaussian noise and `random_state=0`
- **THEN** both results are equal

#### Scenario: Sweep as runs
- **WHEN** `observe` reads a sweep of 5 values of `r` with 3 replicates each and `replicates='mean'`
- **THEN** it returns 5 runs, each with the condition `r` of its design point and the mean of its replicates

### Requirement: Runs split into training and validation
`split_runs(data, validation, random_state)` SHALL split whole runs into a training and a validation `ObservedRunsSpec`,
with the share of runs given, at least one run in each, and the same split for the same seed.

#### Scenario: No run in both sets
- **WHEN** 10 runs are split with `validation=0.3`
- **THEN** the training set holds 7 runs, the validation set 3, and no label appears in both

#### Scenario: Too few runs
- **WHEN** one run is split
- **THEN** it raises `ValidationError` that says a split needs at least two runs
