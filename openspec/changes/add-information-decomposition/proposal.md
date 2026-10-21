# Proposal

## Why

Varley (2023, arXiv 2304.12482) and Scagliarini et al. (2022, arXiv 2207.03581) make higher-order information the
standard toolkit of complex systems: total and dual total correlation, O-information, which tells redundancy from
synergy, its gradients, which attribute synergy to single variables, and the partial information decomposition. Yuan
et al. (2023, arXiv 2312.16815) survey causal emergence and give the integrated-information-decomposition criteria Ψ,
Δ, and Γ, and Rosas et al. (2024, arXiv 2402.09090) define emergence through closure: a macro level is closed when it
predicts itself as well as the micro level does, tested with computational mechanics. The analysis toolkit has entropies,
mutual and transfer information, and `add-emergence-analysis` adds effective information, but none of these measures.

## What Changes

- Add total correlation, dual total correlation, O-information, and the O-information gradients of each variable.
- Add the partial information decomposition of two sources about a target, with the redundancy measure as a keyword,
  wrapping `dit`.
- Add local, pointwise versions of entropy, mutual information, and transfer entropy.
- Add the ΦID emergence criteria Ψ, Δ, and Γ of a macro feature, after Rosas et al. (2020) and Yuan et al.
- Add closure tests of a coarse-graining and ε-machine reconstruction of discrete series.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `complexity-measures`: higher-order information, decomposition, local measures, and emergence criteria.

## Impact

- `skcomplex.core.analysis` gains `total_correlation`, `dual_total_correlation`, `o_information`,
  `o_information_gradients`, `partial_information`, `local_information`, `emergence_criteria`, `closure`, and
  `epsilon_machine`.
- `dit` becomes a runtime dependency if the design's survey confirms it.
