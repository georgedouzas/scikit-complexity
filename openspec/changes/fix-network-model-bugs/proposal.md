## Why

The network and criticality tutorials filed bugs against the network models (issues #59 to #64 and #75). Checked
against the current code on 2026-11-28:

- #61: the newest node of a Holme-Kim network has degree 1 in the last frame, because its triad edges come one step
  later. The `CapacityError` at `until=297` no longer occurs.
- #63: the Watts-Strogatz construction rewires an edge more than once and can move either end, so with `steps=10` and
  `rewiring=1.0` the degree variance is 114.6 and the minimum degree 0. NetworkX gives about 5 and a minimum of
  `mean_degree / 2`. With `steps=1` the minimum degree is 3, below 5.
- #64: the Erdős-Rényi construction caps the linking probability of a node at 1 per step, so `mean_degree=10` with
  `steps=1` gives a mean degree of 2 and no warning.
- #59: Barabási-Albert growth with the default `until=300` has 303 nodes. The docstring states `seed_nodes + t` nodes
  after `t` steps but not the default run length.
- #75: the degrees of `ConfigurationModelSystem(random_state=2)` correlate with `np.random.default_rng(2)`. This is
  by design: build-time draws use `np.random.default_rng(random_state)` as scikit-learn does, and `simulate` spawns its
  own stream from `SeedSequence(random_state)`. The docstring does not say so.
- #60 and #62 no longer reproduce. `network_measures` reads a growing trajectory, and the transitivity of Holme-Kim
  growth matches `nx.powerlaw_cluster_graph` at 404 nodes for `triad` 0 and 1.

## What Changes

- Holme-Kim growth gives every new node all its `edges` edges within the step it joins.
- The Watts-Strogatz construction considers every lattice edge once over the run, rewires it with total probability
  `rewiring`, and keeps its lattice end.
- The Erdős-Rényi construction raises a `ValidationError` when `mean_degree` cannot be reached in `steps` steps, naming
  the smallest number of steps that reaches it.
- The docstrings of the growing networks state the default run length and the resulting node count, and the
  docstring of `ConfigurationModelSystem` states how `random_state` seeds its draws.
- Tests pin the reference behaviour of each model, and of `network_measures` on a growing trajectory and of the
  Holme-Kim transitivity at 1000 nodes against NetworkX.

## Capabilities

### New Capabilities

### Modified Capabilities
- `model-library`: network models match the reference graphs they construct.

## Impact

- `src/skcomplex/systems/mathematics/networks/_growth.py` and `_ensembles.py`, and possibly the `rewire` and `add-edge`
  topology requests in `core/components/laws` or the engine that applies them.
- Tests in `tests/systems/mathematics/networks/test_growth.py` and `test_ensembles.py`, and
  `tests/core/analysis/test_networks.py`.
- Issues #59 to #64 and #75 are closed with the change.
