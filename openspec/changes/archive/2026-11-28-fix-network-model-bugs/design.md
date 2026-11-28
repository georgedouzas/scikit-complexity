## Context

Issues #59 to #64 and #75 came from the network tutorials. Each was reproduced against the current code on
2026-11-28 (see the proposal). Four need code: #61, #63, #64, and the docstrings of #59 and #75.

## Decisions

- **Holme-Kim (#61).** Decided 2026-11-28. The engine decides every topology request of a step from the state at
  the start of the step, and a system does not know `until`, so neither a same-step closure nor an extra closure step
  works inside the model. An `'add-node'` request of `TopologyRequestSpec` gains further edges: after its first
  `edges` edges are drawn, it adds more edges in the same step whose weights may read the new node's first contacts,
  such as their neighbours. This is a general primitive for triadic growth. The step meaning of every other request
  and system stays the same. Holme-Kim growth uses it and drops its `near` and `closure` interactions.
- **Holme-Kim reference (#62).** Decided 2026-11-28. The model follows Holme and Kim 2002, where triad edges go to
  neighbours of the first preferential contact. `nx.powerlaw_cluster_graph` closes triangles on the most recent target
  instead, so at `triad=1` and 1000 nodes it gives 0.144 against 0.170 for the paper's rule. The test compares with
  NetworkX at `triad=0` and with a hand-written version of the paper's rule at `triad=1`, and the docstring states
  the difference.
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
