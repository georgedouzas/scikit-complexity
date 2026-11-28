## Context

Issues #59 to #64 and #75 came from the network tutorials. Each was reproduced against the current code on
2026-11-28 (see the proposal). Four need code: #61, #63, #64, and the docstrings of #59 and #75.

## Decisions

- **Holme-Kim (#61).** Today the node joins with one edge and the `closure` interaction adds `edges - 1` triad edges
  the next step, when the node's age is below `JOINING_AGE`. The fix makes the triad edges part of the joining step,
  so the last frame never holds a half-joined node. If the engine applies topology requests of one step in order, the
  closure runs after `growth` in the same step. Otherwise the run takes one more closure step after the last arrival.
  The implementer picks whichever needs no change to the engine, and states which in the commit.
- **Watts-Strogatz (#63).** Each lattice edge gets one chance over the whole run: at every step an edge not yet
  considered is rewired with probability `1 - (1 - rewiring) ** (1 / steps)` and is then never rewired again, and
  the rewired edge keeps its lattice source and moves its target, as `nx.watts_strogatz_graph` does. If the `rewire`
  request reorders edge ends, it is fixed to keep the first end, since that matches its meaning everywhere.
- **Erdős-Rényi (#64).** A node adds at most one edge per step with probability `mean_degree / 2 / steps`, so it
  cannot exceed 1. Raising a `ValidationError` in `_prepare` that names `ceil(mean_degree / 2)` steps is simpler than
  drawing several edges per node per step, and keeps the construction a sequence of single-edge steps.
- **Seeding (#75).** No code change. Build-time draws use `np.random.default_rng(random_state)`, the scikit-learn
  convention that about 200 systems follow, and `simulate` already spawns an independent stream. The
  `random_state` docstring of `ConfigurationModelSystem` says so.
- **Default run length (#59).** No code change. The docstrings of the growing networks state that `simulate` runs
  `until` steps by default, so the network ends with `seed_nodes + until` nodes.
- **#60 and #62.** No code change. Tests pin the behaviour.

## Risks / Trade-offs

- Fixing `rewire` may change other systems that rewire edges, such as adaptive SIS. Their tests show it.
