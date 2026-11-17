# Spec Delta

## MODIFIED Requirements

### Requirement: Dead worker processes raise a named error
An experiment run on processes SHALL raise a named error when its worker processes die, and the message SHALL name the
main guard that a script needs and the executors that avoid processes. An experiment run on a Dask or Ray executor
SHALL raise the same named error when a remote worker or node dies.

#### Scenario: Script without a main guard
- **WHEN** a script without `if __name__ == '__main__':` builds an orbit diagram with `executor='process'`
- **THEN** it raises `ExperimentError`, whose message names the main guard and `executor='thread'`

#### Scenario: Dead Ray worker
- **WHEN** a Ray worker that runs a chunk of an ensemble is killed
- **THEN** the experiment raises `ExperimentError`, whose message names `RayExecutor` and `max_retries`

## ADDED Requirements

### Requirement: Results convert to xarray datasets
`ExperimentResult.convert_to_dataset()` SHALL return the result as an xarray dataset with the dimensions `run` and
`time`, one variable per kept series and scalar output, and the design values, scenario, replicate, and seed as
coordinates of `run`, lazily for a result held in a store.

#### Scenario: Ensemble as a dataset
- **WHEN** a user converts the result of an ensemble of 20 SIR runs that keeps `I`
- **THEN** the dataset holds the variable `I` with the dimensions `run` and `time`, and 20 values of `seed`

#### Scenario: Stored result
- **WHEN** a user converts the result of a sweep run with a store
- **THEN** the variables of the dataset are lazy arrays read from the store
