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

`partial_information` in `_partial_information.py` decomposes the information that sources hold about a target with
dit, with the minimum and the minimum mutual information redundancies. `surrogate_test` in `_surrogates.py` shuffles
every variable independently and compares a scalar field of an analysis on the data and on the surrogates.

## Goals / Non-Goals

**Goals:**
- Core-periphery detection with the same frame handling as the other kinds of `network_measures`.
- A bootstrap of the gain that reuses SciPy and the settings names `scaling` already uses.
- Causal Emergence 2.0 over a path of scales, with the emergent complexity of 2503.13395.
- The group measures of emergent coordination of 2510.05174 and the null models it tests them against.

**Non-Goals:**
- Continuous coreness models (Rombach et al.) and multi-core models (Kojaku and Masuda). One discrete model and the
  k-core cover the papers in hand.
- Significance of the core against null graphs. `motifs` already holds null ensembles, and a later change can reuse
  them.
- Re-searching the partition on every resample. The bootstrap measures the uncertainty of the gain of one partition.
- Bootstraps of `'criteria'` and `'closure'`.
- The singular-value heuristics of Causal Emergence 2.0 (its SI S3) and higher-order macrostates, which are not
  partitions.
- Synergy of orders above two in `'synergy-capacity'`. The paper computes order two, and the coalition gain covers
  order three.

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

- **Placement of the new kinds.** `'scales'`, `'synergy-capacity'`, `'coalition'`, and the surrogate methods are kinds
  and settings of analyses of the core that read trajectories and matrices by their tags, step 2 of the placement
  rule. They are cross-domain: Markov models of neural, genetic, and ecological dynamics and cellular automata for
  `'scales'`, and groups of agents, neurons, firms, and animals for the synergy, coalition, and null models.
- **Causal Emergence 2.0 as a kind of `emergence`.** `kind='scales'` reads a transition matrix and the settings
  `path`, `primitives`, `weighting`, `search`, and `horizon`. `path` is a sequence of partitions from the micro scale,
  each a coarsening of the one before, or `None` to search. The causal primitives at a scale are computed with a
  uniform distribution of causes, as the paper does. `primitives='determinism-specificity'` (the default) gives
  `CP = determinism + specificity - 1`, the effectiveness of CE 1.0, and `'sufficiency-necessity'` gives
  `CP = sufficiency + necessity - 1`, with `suff(e, c) = P(e | c)` and `nec(e, c) = 1 - P(e | C, not c)` averaged over
  transitions weighted by `P(e | c)`.
- **Consistency.** A partition is consistent when random walkers dropped on every micro state and coarse-grained
  follow the macro chain for `horizon` steps (5 by default), measured as the summed Kullback-Leibler divergence, which
  must be zero up to 1e-12. Inconsistent partitions are skipped by the search and rejected in a given path with
  `AnalysisError` naming the step.
- **Search of the path.** The endpoint is the consistent partition of largest CP, and among ties the one with most
  blocks. Up to 10 states the search is exhaustive over partitions. Above, it merges greedily the pair of blocks whose
  merge keeps consistency and gives the largest CP, which `_greedy` already does for effective information. The path
  is the longest chain of consistent coarsenings from the micro scale to the endpoint that the search found.
- **Results.** `EmergenceResult` gains `scale_values` (CP per scale), `contributions` (the micro CP followed by the
  gains `dCP_i` of every step), `path`, and `complexity`, the entropy `-sum p_i log2 p_i` of the positive gains of the
  macro steps normalised to sum to one. The total causal emergence is the existing `gain`, the CP of the endpoint minus
  that of the micro scale.
- **Group synergy and coalitions.** `kind='synergy-capacity'` reads the series of `N` agents, discretised by the
  existing estimators, and for every pair computes the synergy of `{X_i(t), X_j(t)}` about `(X_i(t + lag), X_j(t +
  lag))` with `partial_information`, and returns the median over pairs as `value` and the matrix of pairs.
  `kind='coalition'` reads the agents and a macro signal given by `partition` as in `'criteria'`, computes `I3` for
  every triplet and `G3 = I3 - max` of its pairs, and returns the median of `G3` and of `I3`. Both read `redundancy`,
  `'mmi'` by default as the paper's robustness check, or `'imin'`.
- **Surrogate methods.** `surrogate_test` takes `method`, `'shuffle'` by default so existing calls keep their result.
  `'shift'` rolls every variable by an independent uniform lag, which keeps its autocorrelation and breaks alignment.
  `'swap'` permutes the variables at every time independently, which keeps the values of each time and breaks the
  identities of the agents. `'block'` shuffles blocks of `block` samples of every variable. These are the
  column-shift, row-shuffle, and block-shuffle nulls of 2510.05174.

## Risks / Trade-offs

- [The swap search finds a local optimum] → It starts from the strength ranking, which is optimal for many graphs,
  and the planted-core test checks it. The docstring states that the search is local.
- [Memory of expanded transitions] → One pair per transition. Counts in the millions use a few tens of megabytes. The
  guide states this.
- [Breaking default of `method`] → Calls that passed a method keep working. Calls without one get the same Louvain
  result.

- [Exhaustive search over partitions grows as the Bell numbers] → The exhaustive search stops at 10 states, 115975
  partitions, as `'causal'` does, and the greedy search runs above.
- [Plug-in synergy of many pairs is biased] → The surrogate tests give the null for the same estimator, which is how the
  paper uses the measure.

## Migration Plan

None beyond the changelog line for the `method` default.
