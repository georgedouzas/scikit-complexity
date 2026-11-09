## Context

`network_measures` in `src/skcomplex/core/analysis/_networks.py` dispatches on `kind` over `KINDS`, checks `method`
against `METHODS = ('louvain', 'label_propagation')` for every kind, and reads the settings of each kind from
`SETTINGS` with `_check_settings`. `_communities` wraps NetworkX.

`emergence` in `src/skcomplex/core/analysis/_emergence.py` computes the gain of `'causal'` in `_causal` from a square
matrix that `_square` normalises by row, with `_effective_information` and `_coarse_matrix`. A partition is given or
found by `_greedy` or `_exhaustive`. `scaling(kind='allometric')` in `_scaling.py` already runs `scipy.stats.bootstrap`
with the settings `resamples`, `confidence`, and `random_state`, and stores the interval as `interval`.

Agent systems with a shared pool exist, such as `PolymerAssemblyPoolSystem` in `systems/biology/evolution/_assembly.py`,
a variable group with an interaction that draws from it every step.

## Goals / Non-Goals

**Goals:**
- Core-periphery detection with the same frame handling as the other kinds of `network_measures`.
- A bootstrap of the gain that reuses SciPy and the settings names `scaling` already uses.

**Non-Goals:**
- Continuous coreness models (Rombach et al.) and multi-core models (Kojaku and Masuda). One discrete model and the
  k-core cover the papers in hand.
- Significance of the core against null graphs. `motifs` already holds null ensembles, and a later change can reuse
  them.
- Re-searching the partition on every resample. The bootstrap measures the uncertainty of the gain of one partition.
- Bootstraps of `'criteria'` and `'closure'`.

## Decisions

- **Borgatti and Everett, discrete, in the core.** The model finds the core set `C` that maximises the Pearson
  correlation between the upper triangle of the weighted adjacency and the ideal pattern `delta_ij = 1` when `i` or
  `j` is in `C`. The search starts from the nodes ranked by strength, takes the best prefix of that ranking, then
  swaps single nodes while the correlation grows, with ties broken by a seeded generator. The coreness of a node is its
  position in the final ranking scaled to `[0, 1]`. The k-core method takes the nodes of the largest `core_number` of
  NetworkX as the core and that number as the coreness. Both report the same correlation as `quality`, so the two
  methods compare.
- **Result fields.** `NetworkMeasuresResult` gains `core` (a boolean array per frame in node order), `coreness` (per
  frame), and `quality` (an array over frames). The kind reads `method` and `random_state`.
- **`method` per kind.** `METHODS` becomes a mapping from kind to its methods, and `method=None` takes the first.
  This keeps one setting for the algorithm of a kind, rather than a second setting for the core.
- **Bootstrap by SciPy.** The counts matrix is expanded into paired arrays of source and target states, one row per
  observed transition. `scipy.stats.bootstrap` resamples the pairs with `paired=True`, `method='percentile'`, and the
  statistic rebuilds the counts with `np.add.at` and returns the gain of the fixed partition. The p-value is the
  fraction of `bootstrap_distribution` at or below zero. A matrix whose entries are not whole numbers raises
  `AnalysisError`, since probabilities carry no sample size. `EmergenceResult` gains `gains`, `interval`, and
  `p_value`. `'causal'` reads `resamples` (0, the default, for none), `confidence`, and `random_state`.
- **Surviving by serving as an agent system.** The pool is a variable group of unit vectors in five dimensions, with
  the generating agent and the raw type of every vector as states. Agents are a group with requirement vectors, the
  coefficients `a` and `b`, the modification vector `R`, credit, and the uncredited count. Phases of the discrete time
  run in the order of the paper: source, ageing, transformation, evaluation. The transformation interaction keeps the
  cumulative transfer counts between agents as a shaped state, the network the paper reads. Defaults follow Table 1
  of the paper: `T_ada = 30`, `n = 4`, `sigma_ada = 0.05`, two raw types, two evaluators, five dimensions, and the
  threshold 0.48 of its first run.
- **Libraries surveyed.** `cpnet` 0.0.21 (2022-07-11) implements Borgatti and Everett and several other models but is
  past the 24-month limit of the constitution. NetworkX 3.7 (2026-09-21) has `core_number` and `rich_club_coefficient`
  but no Borgatti and Everett model. `graph-tool` is not on PyPI. So the discrete model is a short private function
  over NumPy, and the k-core comes from NetworkX. SciPy 1.18.1 (2026-08-21) gives the bootstrap.

## Risks / Trade-offs

- [The swap search finds a local optimum] → It starts from the strength ranking, which is optimal for many graphs,
  and the planted-core test checks it. The docstring states that the search is local.
- [Memory of expanded transitions] → One pair per transition. Counts in the millions use a few tens of megabytes. The
  guide states this.
- [Breaking default of `method`] → Calls that passed a method keep working. Calls without one get the same Louvain
  result.

## Migration Plan

None beyond the changelog line for the `method` default.
