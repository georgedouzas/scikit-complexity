# Design

## Context

The systems live in self-contained field modules. Analyses are functions of `core.analysis` that take a trajectory
or a system and a `kind`. The engine compiles fixed-shape arrays, and lattice spaces already build their neighbour
tables privately (`LatticeSpace._get_index_table`, `_get_neighbor_pairs`).

## Goals / Non-Goals

**Goals:** every feature of #49 to #58 through the existing parts: options on existing systems, new kinds of existing
analyses, and the existing `get_network` accessor. No new package and no new public class unless an element or an
interaction needs one.

**Non-Goals:** a general lattice that grows during a run for every system. Only Langton's ant gets an unbounded plane.

## Decisions

- **Energy (#49):** an `Observable('energy')` on each lattice spin system, evaluated on the neighbour pairs of its
  lattice: `-J sum s_i s_j - h sum s_i` for Ising, `-J sum delta(s_i, s_j)` for Potts and `-J sum cos(theta_i -
  theta_j)` for XY. The energy is total, not per site, and the docstrings say so.
- **Interaction graph (#49, #52):** `get_network(frame)` already returns a `NetworkSnapshotSpec` for changing
  networks. For a system with no changing network, it returns the graph of the neighbourhood its interactions read:
  the edges of a lattice, or the pairs within the radius of a continuous space at that frame. This keeps one accessor
  for the interaction graph instead of a second public method on spaces. A system with neither still raises the
  `KeyError`, with a message that names both cases.
- **Pairs within a radius (#52):** `neighbors(kind='radius', r_max=)` computes the pairs with `scipy.spatial.cKDTree`
  and its `boxsize` for periodic boxes, and returns the pairs and the degrees of every frame. It works without any
  neighbourhood in the system, for any radius.
- **Galam (#50):** `groups='sample'` draws, for every agent, `group_size` agents uniformly with replacement, possibly
  itself, and sets its opinion to their majority. Ties follow the rule of the partition variant.
- **Logistic ensembles (#51):** `x` takes a scalar or an array. The state has the shape of `x`, so a scalar keeps
  today's behaviour.
- **Discrete walks (#53):** `RandomWalkSystem(time='discrete')` follows the precedent of `PolyaUrnSystem(time=)`.
  The step is `+1` with probability `bias`. `absorbing=(a, b)` freezes a walker that reaches `a` or `b`. `floor=f`
  maps `x` to `max(f, x + step)`. Both only apply in discrete time and raise otherwise. `start=` sets the initial
  position. The positions are one vector state, so 10^3 walkers over 10^4 steps run in seconds.
- **Pólya (#54):** `balls=` is a sequence of initial counts, at least two. The draw picks a colour with probability
  proportional to its count and adds `added` balls of it. The red share of the docstring becomes the share of each
  colour, with its Dirichlet limit.
- **Gibrat (#55):** `factors=` and `weights=` give a discrete law of the factor. When `factors` is given, `mu` and
  `sigma` are unused and the docstring says so. Weights default to uniform and must sum to 1.
- **Attractors (#56):** `kind='attractors'` reads the global map of a finite deterministic system from the same
  sources `automaton` already reads, plus the tables and inputs of a Boolean network. It encodes every configuration
  as an integer, applies the map to all of them at once with NumPy, and finds the cycles and transients of the
  resulting functional graph. It refuses more than `2^24` configurations with an `AnalysisError` that names the
  count. The result holds the transient and the attractor of every configuration, and the cycles with their lengths
  and basin sizes.
- **Langton's ant (#57):**
  - `colors=` gives the initial colours (an array of the shape), `start=` the initial site and `heading=` the
    initial heading.
  - The ant's position and heading are recorded as states at every step.
  - A step updates only the ant's cell instead of the whole lattice.
  - `shape=None` is the unbounded plane. The black cells are a sparse set held by the ant, through the mechanism of
    rewritten elements whose structure `get_structure` reads, so memory grows with the cells visited and not with a
    lattice. If the engine cannot express this, the implementer stops and reports before choosing another design.
- **Boolean networks (#58):**
  - One group element holds all nodes, with the state `nodes.x`, the inputs and the tables as parameters.
  - `update='synchronous'` stays the default. `'asynchronous'` updates one node drawn uniformly per step, where one
    unit of time is `n_nodes` single updates, as the docstring says.
  - `annealed=True` redraws the inputs and tables at every step from the same `k` and `bias`.
  - `automaton(kind='difference')` accepts a Boolean network and returns the Hamming distance between two copies over
    time.

## Risks / Trade-offs

- The unbounded ant needs a structure that is not an array. Relying on rewritten elements reuses an existing path but
  may not fit. The stop-and-report rule above bounds that risk.
- Removing `red=` and `blue=` breaks callers. The tutorials are the main ones and asked for the change.
- Moving Boolean networks to one group changes the element names of the old layout. The docstring names the new
  state, and tests read the new names.
