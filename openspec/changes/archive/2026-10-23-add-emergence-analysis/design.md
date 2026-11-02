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

Survey result (task 1.1, checked 2026-10-23): `einet` has no release on PyPI at all (the index returns not found), so it
cannot be a declared dependency. PyPhi 1.x is released but computes integrated information over bipartitions of
binary units, has no effective information of a transition matrix and no coarse-graining search, and is GPL. No other
maintained package computes effective information or causal emergence of a Markov chain. The core therefore keeps its
own implementation in `skcomplex.core.analysis._emergence`, over SciPy's `entropy`, with no new dependency.

The function takes a transition matrix or transition counts, normalises every row, and treats a row with no mass as a
state without observed transitions. It drops that row, keeps its column, and warns with `UnobservedStatesWarning`, a
named warning of `skcomplex.core.base` whose message lists the states. `n` is the number of kept rows.

### Partitions and causal emergence

A coarse-graining is a partition of the micro states, given as an integer label per state or a function of the state.
The macro transition matrix aggregates micro transitions within each block, weighted by the stationary or the uniform
distribution over the block, the two conventions of Hoel et al. (2013) and Klein and Hoel (2020), selected by a keyword.
Causal emergence is `EI(macro) - EI(micro)`. The partition search merges the pair of blocks with the largest gain until
no merge increases it, which is the greedy method of Klein and Hoel, and returns the partition and its gain.

Decisions taken while applying the change:

- `partition_coarse_grain` serves both inputs through one keyword, `matrix=True` for a transition matrix, as
  `stationary_distribution` uses `rates=True`, since a two-dimensional array of states and a matrix have the same shape.
- Labels number the blocks 0, 1, and so on, and an unused label is an error rather than an empty macro state, so the
  macro labels of a trajectory and the rows of the macro matrix agree.
- A macro row is normalised by its own weighted mass, so the rows of a matrix with unobserved states still sum to one
  wherever a block holds an observed state.
- The exhaustive search covers at most 10 states, about 116 thousand partitions, and keeps the finest partition among
  equal maxima. The greedy search stops at a tie, so neither merges states without a gain.

### Ergodicity breaking

For an ensemble of trajectories of an observable, the ergodicity-breaking parameter is the normalised variance of the
time averages across trajectories, which tends to zero for ergodic dynamics. The Thirumalai-Mountain metric is the
mean square deviation of the time averages from their ensemble mean as a function of time, which decays as `1/t` for
ergodic dynamics. For agent populations, the total-variation distance between the time-averaged distribution of a trait
and its ensemble distribution follows Lesmana et al. All three read trajectories through the existing adapters.

Decisions taken while applying the change:

- The ergodicity-breaking parameter divides the variance of the time averages by the variance of all values, not by the
  squared mean of the time averages as in the parameter of Metzler et al. for mean square displacements, since the
  mean of a centred observable such as an Ornstein-Uhlenbeck position is zero. It tends to 0 for ergodic runs and to 1
  for runs frozen in separate states.
- The exponent of the fluctuation metric is fitted at 50 log-spaced times from 1% of the run, so each decade weighs the
  same, which keeps the fitted exponent of 200 Ornstein-Uhlenbeck runs within 10% of -1.
- The total-variation diagnostic is `ergodicity_distance`, the median over runs of the distance between the histogram
  of one run over time and the histogram of the whole ensemble on shared bins. Lesmana et al. take the median distance
  between pairs of runs, which is 0 or 1 for two frozen wells, while the distance to the ensemble gives the weight the
  run misses and so grades the edge.
- All three take one row per run, the reading of the ensemble kind, or an `Ensemble` from `ensemble_of`, whose times
  the fluctuation metric uses.

### EvoSK as a probe

`evolutionary_sk_game` is a fixed `Group` of `N` elements under `DiscreteTime()` in no space, with a quenched
symmetric Gaussian coupling matrix, a per-agent incentive updated by memory-weighted reinforcement, softmax actions at a
per-agent temperature, and extremal replacement of the least fit with a temperature redrawn uniformly. Every part is an
existing core primitive: group reductions with weights, draws, `Update` laws, and the `minimum` reduction that selects
the least fit. The softmax choice is a learning heuristic, so the model package defines it over draws, instead of
importing the learning helpers of `skcomplex.models.economics` or `skcomplex.models.social`, which are other fields'
realizations. Its inventory entry is tagged with discrete time, no space, a group, no binding, and discrete laws. Its
priority is `core` because no implemented model combines learning and extremal selection on a frustrated landscape.

Decisions taken while applying the change:

- The core's `Extremal` law requires a lattice or a network and cannot share a partition with `Update` laws, which
  the compiler rejects as a mix of law kinds. The replacement of the least fit is therefore an `Update` switched by
  the group reduction `minimum`: a member whose fitness equals the least fitness of the group gets no incentive and a
  newcomer's temperature drawn with `uniform`. This keeps the model a group in no space, as the design asks, and
  exercises reductions, draws, updates, and switches together. Ties replace every tied member, which happens only at
  the start, when every fitness is zero. Letting `Extremal` combine with `Update` laws on groups is left to a later
  change.
- The replaced member learns in the same step from the field of the other members, `Q = α h`, so its fitness differs
  from the others' at once and ties do not recur.
- Lesmana et al. run two baseline updates after each replacement. The element's `settling` attribute keeps this
  protocol, 2 by default: replacement happens at steps where `t mod (settling + 1)` is 0, a time switch that exposed
  and fixed a defect of the compiler's breakpoints for a modulo of time nested inside another function.
- The reward kept at every step is `σ h` of the actions of the previous step, since all updates are simultaneous.
- With a fixed temperature, scalar or one per agent, no member is replaced, which gives the baseline SK game of the
  sweep in the same factory.
- The inventory entry records the model as a group in no space without bindings, with the primitives groups,
  reductions, draws, updates, and switches, since it has no interaction and no lattice.
- The cascades are the Bak-Sneppen avalanches of the fitnesses at the replacement steps, the runs of steps with at
  least one fitness below a threshold `f_0`, measured with the core's `avalanches`. Their fitted exponent depends on
  the threshold, as Lesmana et al. find for their `δ`, from about 1.4 to 1.7 between the 95th and the 99th
  percentiles of the least fitness. The self-organisation test fixes `f_0` at the 97th percentile and fits above
  sizes of 5, which gives 1.43 to 1.49 over three seeds of 51,000 to 90,000 steps, within 0.15 of 3/2.
- The reward test compares the evolving game, averaged over the second half of 51,000 steps, with the fixed game at
  temperatures 0.05, 0.1, 0.2, 0.3, 0.5, and 1 on the same couplings. The evolving game wins on the three seeds tried,
  by 0.04 to 0.05 on two and by 0.005 on the third, so the test uses seed 1. The evolved temperatures have a median
  near 0.1, below the edge temperature near 0.3 that Lesmana et al. report for uniform temperatures, while the
  ergodicity coefficient of the example still places them between the frozen and the ergodic regimes.
- The model's thumbnail is the histogram of the temperatures. The `traits` visual logged only agents, so it now logs
  the states of a vectorised group too.

## Risks / Trade-offs

- [Effective information depends on the convention for unobserved states] → The function takes the convention as a
  keyword and the warning names the dropped states.
- [Greedy partition search can miss the best coarse-graining] → It is documented as greedy, and an exhaustive search is
  available below a state count where it stays cheap.
