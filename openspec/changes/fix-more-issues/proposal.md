# Proposal

## Why

Twelve more GitHub issues, #33 to #44, came from writing tutorials on scaling, branching, and urns. They report wrong
or missing docstrings, analysis functions that fail silently or print to the console, a stable law fit that is
unstable near an index of 1, slow systems, and per-agent values of groups whose size changes that `get_values` cannot
read.

## What Changes

- **Docstrings (#33, #34, #43, #44):**
  - `SimonYuleSystem` states the rank-frequency slope `1 - rho` and the count exponent `1 + 1 / (1 - rho)`.
  - `PolyaUrnSystem` says that draws happen at the times of a Poisson process of rate 1.
  - `clusters(kind='percolation')` says the strength is a share of all sites and how to convert it to a share of the
    occupied ones.
  - `ContactProcessSystem` says that its critical rate depends on the time slice `tau` and how to locate it.
- **Pólya urns (#34):** `PolyaUrnSystem(time='discrete')` makes one draw per step.
- **Galton Watson (#35):** the offspring law is Poisson, geometric, or any finite distribution given by its
  probabilities, and `individuals=False` tracks only the size of every generation, so supercritical runs stay cheap.
  The system draws nothing when prepared, so it takes no seed of its own, which its docstring says.
- **Stable fits (#36):** `fit_law(law='stable')` reports the location in the S0 parametrisation, continuous in the
  index, and a skewness of 0 at an index of 2, where it is not identified. **BREAKING:** the location changes meaning.
- **Branching ratios (#37):** `events(kind='branching')` silences the console output of mrestimator and passes its
  messages on as Python warnings.
- **West Brown Enquist (#38):** the run time grows with the number of vessels, not much faster.
- **Box counting (#39, #40):** a grid thinner than a box raises an `AnalysisError`, and an array of points of shape
  `(N, d)` is counted at every box size without a fixed binning.
- **Detrended fluctuation (#41):** the default window sizes are unique.
- **Variable groups (#42):** `get_values` reads a per-agent state of a group whose size changes, as one array per
  frame.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-catalog`: Pólya urns in discrete time and Galton Watson offspring laws and generation sizes.
- `trajectory-analysis`: stable fits, branching ratios, box counting, and detrended fluctuation.
- `simulation`: per-agent states of groups whose size changes through `get_values`.

## Impact

- Code: `systems/social/spreading/_growth.py`, `systems/mathematics/stochastic_processes/_urns.py`, `_branching.py`,
  `systems/biology/physiology/_distribution_networks.py`, `systems/physics/statistical_mechanics/_processes.py`,
  `core/analysis/_fit.py`, `_events.py`, `_scaling.py`, `_clusters.py`, `core/engine/runtime/_trajectories.py`.
- Breaking: the stable location is now S0. Default box-counting and fluctuation results can change slightly.
