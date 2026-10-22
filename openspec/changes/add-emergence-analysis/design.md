# Design

## Context

The analysis toolkit already estimates transition matrices (`markov_state_model`), coarse-grains agent positions onto
grids (`coarse_grain`), measures avalanches and branching ratios (`criticality-analysis`), and computes entropies and
transfer entropy (`complexity-measures`). It has no measure of emergence across scales and no diagnostic of ergodicity
breaking, and the papers added to `references/papers` define both precisely.

This design is rebased on `redesign-system-model`. The measures are analysis functions over trajectories and transition
matrices and add no primitive, solver, or backend. The probe model uses the redesigned vocabulary, and its learning
rule is a realization in the model package, since learning heuristics such as logit choice moved out of the core.

## Goals / Non-Goals

**Goals:**
- Measure causal emergence of any system the library can turn into a Markov chain, at any coarse-graining.
- Diagnose whether an ensemble of trajectories, or a population of agents, is ergodic.
- Add EvoSK as a probe model that combines covered primitives in a new way.

**Non-Goals:**
- Integrated information theory (Φ), which needs exhaustive bipartitions and has its own library, PyPhi.
- The categorical interaction-space formalism of Giordano (2024), which defines but does not compute.

## Decisions

### Effective information on transition matrices

For a transition matrix `T` over `n` states, the effective information is the mutual information between a uniform
intervention on the current state and the next state: `EI = H(mean row of T) - mean over rows of H(row)`, in bits. It
splits into determinism, `log2 n - mean row entropy`, and degeneracy, `log2 n - H(mean row)`, so `EI` equals
determinism minus degeneracy. Rows with no observed transition are dropped with a named warning rather than filled.

Library survey: `einet` (Klein and Hoel) implements effective information for networks, but is not maintained or
released on PyPI with current Python support, and PyPhi computes integrated information, not effective information. The
formula is a few lines over NumPy and SciPy's `entropy`, so the core implements it, as it does Shannon entropy today. The
design records the survey so a maintained library can replace it.

### Partitions and causal emergence

A coarse-graining is a partition of the micro states, given as an integer label per state or a function of the state.
The macro transition matrix aggregates micro transitions within each block, weighted by the stationary or the uniform
distribution over the block, the two conventions of Hoel et al. (2013) and Klein and Hoel (2020), selected by a keyword.
Causal emergence is `EI(macro) - EI(micro)`. The partition search merges the pair of blocks with the largest gain until
no merge increases it, which is the greedy method of Klein and Hoel, and returns the partition and its gain.

### Ergodicity breaking

For an ensemble of trajectories of an observable, the ergodicity-breaking parameter is the normalised variance of the
time averages across trajectories, which tends to zero for ergodic dynamics. The Thirumalai-Mountain metric is the
mean square deviation of the time averages from their ensemble mean as a function of time, which decays as `1/t` for
ergodic dynamics. For agent populations, the total-variation distance between the time-averaged distribution of a trait
and its ensemble distribution follows Lesmana et al. All three read trajectories through the existing adapters.

### EvoSK as a probe

`evolutionary_sk_game` is a fixed `Group` of `N` elements under `DiscreteTime()` in no space, with a quenched
symmetric Gaussian coupling matrix, a per-agent incentive updated by memory-weighted reinforcement, softmax actions at a
per-agent temperature, and extremal replacement of the least fit with a temperature redrawn uniformly. Every part is an
existing core primitive: group reductions with weights, draws, `Update` laws, and `Extremal`. The softmax choice is a
learning heuristic, so the model package defines it over draws, instead of importing the learning helpers of
`skcomplex.models.economics` or `skcomplex.models.social`, which are other fields' realizations. Its inventory entry
is tagged with discrete time, no space, a group, the members binding, and discrete laws. Its priority is `core` because
no implemented model combines learning and extremal selection on a frustrated landscape.

## Risks / Trade-offs

- [Effective information depends on the convention for unobserved states] → The function takes the convention as a
  keyword and the warning names the dropped states.
- [Greedy partition search can miss the best coarse-graining] → It is documented as greedy, and an exhaustive search is
  available below a state count where it stays cheap.
