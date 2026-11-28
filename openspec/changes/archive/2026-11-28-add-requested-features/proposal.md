# Proposal

## Why

Ten GitHub issues, #49 to #58, came from writing the Complexity Science tutorials. Each asks for something a chapter
now rebuilds with NumPy: the energy and the interaction graph of a lattice, ensembles and variants of known systems,
a discrete-time walk with boundaries, the attractors of a finite system, and faster runs of Langton's ant and of large
Boolean networks. All of them are general, so they belong in the library and not in one book.

## What Changes

- **Spin energies and interaction graphs (#49):**
  - `IsingSystem`, `PottsSystem` and `XYSystem` record the observable `energy` at every frame.
  - `Trajectory.get_network(frame)` returns the interaction graph of a system whose space is fixed: the
    neighbourhood a lattice interaction reads, such as von Neumann for Ising and Moore for Life, or the pairs within
    the interaction radius of agents in a continuous space at that frame.
- **Majority rule with replacement (#50):** `GalamMajoritySystem(groups='sample')` lets every agent draw its own
  group with replacement each round. The default `groups='partition'` keeps the disjoint groups.
- **Logistic map ensembles (#51):** `LogisticMapSystem(x=array)` iterates one map per initial value, with the state
  `map.x` of shape `(n,)`.
- **Pairs within a radius (#52):** `neighbors(kind='radius', r_max=)` returns, per frame, the pairs of agents within
  the radius, across periodic edges, and the degree of every agent.
- **Discrete-time walks (#53):** `RandomWalkSystem(time='discrete')` steps every walker by `+1` or `-1` once per unit
  of time, with `absorbing=(a, b)` to stop walkers at either end and `floor=` to reflect them at a floor.
- **Pólya urns of any number of colours (#54):** `PolyaUrnSystem(balls=(1, 1, 2))` holds one count per colour, with
  the state `urns.balls` of shape `(n_urns, n_colours)`. **BREAKING:** `red=` and `blue=` are removed, with no alias.
- **Discrete growth factors (#55):** `GibratProcessSystem(factors=(1.5, 0.6), weights=(0.5, 0.5))` draws the factor
  from a discrete law. The default stays the normal law of `mu` and `sigma`.
- **Attractors of finite systems (#56):** `automaton(system, kind='attractors')` maps every configuration of a small
  finite automaton or Boolean network in one vectorised pass, and returns the transient length and cycle of every
  configuration, the attractors, and their basin sizes.
- **Langton's ant (#57):** the ant starts on given colours, at a given site and heading. It records its position and
  heading at every step, and each step updates only its own cell. `shape=None` puts it on the unbounded plane.
- **Random Boolean networks (#58):** the nodes are one group with the state `nodes.x` of shape `(n_nodes,)`.
  `update='asynchronous'` updates one random node at a time, and `annealed=True` redraws the inputs and tables every
  step. `automaton(kind='difference')` measures damage spreading between two copies of a network.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-catalog`: the system changes above.
- `trajectory-analysis`: `get_network` on fixed spaces, `neighbors(kind='radius')`, and the `attractors` and Boolean
  network `difference` kinds of `automaton`.

## Impact

- Systems: `_spins.py`, `_persuasion.py`, `_maps.py`, `_walks.py`, `_urns.py`, `_growth_processes.py`,
  `_automata.py` and `_boolean.py`, their tests, and examples or guide pages that pass `red=` or `blue=`.
- Analysis: `_neighbors.py` and `_automata.py`.
- Runtime: `Trajectory.get_network` in `_trajectories.py`.
