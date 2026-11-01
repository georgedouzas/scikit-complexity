# Design

## Context

The gap list of `general-elements-and-interactions` names, for every class left on its own laws, the capability it
lacks. The core classes and conventions of that change are the starting point: operation names, variants as specs,
`form='symbolic'|'agent'`, the member schedule, reference states, and field cases as thin subclasses in models.

## Goals / Non-Goals

**Goals:**
- Every model class of the gap list is rebuilt on general classes, or recorded as staying in its model for a stated
  reason.
- Each new capability extends an existing general class with an argument whose default keeps today's behaviour,
  unless it is a new operation.

**Non-Goals:**
- Couplings by kernels, matrix-valued weights, and laws per hyperedge, which other pending changes add.
- Non-numeric states.

## Decisions

### Extend before adding
Coupling of separate elements and of a hub are new binding modes of `CouplingInteraction`. Parametric coefficients
are values of the existing stoichiometry. Edge transfers are a pairing of `TransferInteraction`. Visits are a schedule
of the agent forms. Agent coupling over radii is the agent form of `CouplingInteraction`.

### Equality is the acceptance test
Every rebuilt model keeps its factory's name and parameters, and its trajectories stay within its tests' tolerance.
A model whose draws change is compared over 20 seeds against its previous trajectory and reverted if it differs.

## Risks / Trade-offs

- **[Risk] Some rebuilds fail again.** The commons herders already failed once. → Each task investigates the cause
  first and records the model as staying if the difference is a property of its order of draws.
- **[Trade-off] Splitting modules renames private modules.** Public names do not change, since packages re-export them.

### Decisions of 2026-11-01

- **`PersonElement` takes the timing of `TransitionInteraction`.** Exposed people become infectious when their age in
  the exposed state reaches `incubation`, counted by the general class, which shifts the onset by half a step from the
  hand-kept clock. The model docstring states the convention and its tests take the new trajectory; the agent model
  still matches its compartment model within the existing tolerance.

### Experiments replicate interactions (1.7)

`BaseExperiment._replicate` accepts plain single elements with their interactions: the replicated system holds shallow
copies of the interactions, which bind members by index. Per-member values apply to element states and parameters
only, so sweeping a parameter that lives on an interaction keeps the non-vectorized path. `Trajectory.member` copies
entries whose size is the same in the replicated and the original problem.

## Migration Plan

Work on `development`, with parallel agents on disjoint packages merging back as they report.

## Open Questions

- None.
