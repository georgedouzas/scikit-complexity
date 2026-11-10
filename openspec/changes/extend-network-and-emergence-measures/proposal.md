## Why

Two small measures are missing for papers already surveyed. Metzner et al. (2026, arXiv 2606.26733) find a stable core
of agents surrounded by a periphery in the transformation network of their surviving-by-serving model, and pick the
core by hand from activity ranks. `network_measures` detects communities but not a core and a periphery. Russo (2026,
arXiv 2604.19827) states causal emergence as confirmed only when the macro effective information exceeds the micro one
with bootstrap significance. `emergence` gives the gain of a partition but no uncertainty on it.

Two papers of the survey of complex adaptive systems ask for more measures of emergence. Causal Emergence 2.0 (Hoel
2025, arXiv 2503.13395) apportions causation over every scale of a path from the micro to the macro scale and measures
emergent complexity as the spread of these contributions, while `emergence` compares one partition with the micro
scale. Riedl (2025, arXiv 2510.05174) tests a group of agents for emergent coordination with the synergy that pairs of
agents hold about their joint future, the gain of triplets over pairs in predicting a macro signal, and null models that
break the identities or the alignment of the agents. `partial_information` and the practical criterion exist, but these
group-level measures and their nulls do not.

## What Changes

- `network_measures` gains the kind `'core-periphery'`: the core of every frame by the discrete model of Borgatti and
  Everett (1999) or by the innermost k-core, the coreness of every node, and the quality of the split, the correlation
  of the adjacency with the ideal core-periphery pattern. Directed and weighted graphs are symmetrised by summing
  reciprocal weights, as the paper does with transfer counts.
- **BREAKING:** `method` of `network_measures` defaults to `None`, which picks the default method of the kind,
  `'louvain'` for communities and `'borgatti-everett'` for the core. Code that passes `method` is unaffected.
- `emergence(kind='causal')` gains `resamples`, `confidence`, and `random_state`. With `resamples` above 0 and
  transition counts as input, it resamples the observed transitions, recomputes the gain of the same partition on each
  resample, and returns the resampled gains, a percentile confidence interval, and the bootstrap p-value of a positive
  gain.
- `emergence` gains the kind `'scales'`, Causal Emergence 2.0: the causal primitives (determinism and specificity, or
  sufficiency and necessity) at every scale of a path of nested partitions that are dynamically consistent with the
  micro chain, given or searched, the contribution of every scale, the total causal emergence, and the emergent
  complexity, the entropy of the contributions of the macro steps.
- `emergence` gains the kinds `'synergy-capacity'`, the median over pairs of agents of the synergy that their present
  states hold about their joint future, and `'coalition'`, the median over triplets of the information that a triplet
  holds about a macro signal beyond its best pair, after Riedl (2025).
- `surrogate_test` gains `method`: `'shuffle'`, the present independent shuffles and the default, `'shift'`, circular
  shifts of every variable by its own random lag, `'swap'`, a permutation of the variables at every time, and
  `'block'`, shuffles of blocks of a given length.
- New system `SurvivingByServingSystem` in `biology/evolution`: agents that transform pairs of feature vectors from a
  shared, aging pool, gain credit when their outputs are used by agents or evaluators, and mutate after long
  uncredited spells by a Hill-type probability. It records the transfers between agents, the network whose core and
  periphery the gallery example detects.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `network-analysis`: core-periphery structure is detected.
- `complexity-measures`: the gain of causal emergence has a bootstrap interval and p-value, causal emergence is
  apportioned over a hierarchy of scales, group synergy and coalition gains are measured, and surrogate tests keep
  chosen structure.
- `model-library`: adds the surviving-by-serving system.

## Impact

- Code: `src/skcomplex/core/analysis/_networks.py` (new kind, `NetworkMeasuresResult` fields, `method` default),
  `src/skcomplex/core/analysis/_emergence.py` (bootstrap of `'causal'`, the kinds `'scales'`, `'synergy-capacity'`,
  and `'coalition'`, `EmergenceResult` fields), and `src/skcomplex/core/analysis/_surrogates.py` (`method`). A new
  module in `src/skcomplex/systems/biology/evolution/`.
- No new dependencies. NetworkX gives `core_number`, SciPy gives `stats.bootstrap`, and dit, already a runtime
  dependency, gives the decomposition of the synergy. `cpnet`, the core-periphery package, is not used because its
  last release is 0.0.21 of 2022-07-11.
- Docs: the network and emergence sections of the analysis guide, a gallery example, and an inventory entry.
- `add-language-model-agents` adds the group guessing game of 2510.05174, whose gallery example reads the new synergy
  measures.
