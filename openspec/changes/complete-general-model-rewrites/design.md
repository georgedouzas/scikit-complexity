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

### Parametric coefficients and agent reactions (1.2)

A stoichiometric coefficient may be a function of the reaction's namespace returning an expression; under mass action
it is also the exponent, so a reaction whose order differs from its scaled change passes a `propensity` (the
Oregonator). `ReactionInteraction(form='agent', phase=...)` lowers to a `RuleLaw` in which every member of the target
group reacts on its own states, all computed from the start-of-step states: mode `'rate'` adds `change * propensity`
per step, mode `'jump'` fires at most once per member per step with probability `1 - exp(-propensity)`, the per-step
convention of the agent form of `TransitionInteraction`, and mode `'langevin'` is refused. No current model uses the
agent form yet. The rats' `NervousnessInteraction` became `AlarmInteraction` plus a plain calming reaction.

### Large biology ODE models (2.4)

43 single-element models of oncology, neuroscience, physiology, and systems biology keep their states and parameters
and build one interaction per mechanism in their factories, within 1e-5 relative of their previous trajectories.
`biology/common/kinetics` holds `MichaelisMentenInteraction` and `biology/common/pharmacokinetics` holds
`DosingInteraction`; `TumourElement` merged into `PopulationElement`. Kept: elements that tests simulate alone
(`TumourImmuneChemotherapyElement`, `ResistanceSwitchingElement`, `OptimalChemotherapyElement`), models whose terms are
not separate mechanisms (vascularized spheroid geometry, HER2, the piecewise-linear prostate fit), growth laws the
populations family lacks, single equations and normal forms (theta, rate unit, leaky integrate-and-fire, Izhikevich,
Mackey-Glass, Aliev-Panfilov), and the stochastic `ExcitableElement`.

### Ecology agents and herders (2.1, 2.3)

Agent predation, trails, ants, angiogenesis, and the stigmergic plants and animals run on member interactions, equal
over 20 seeds or identical. The herders differed earlier for two reasons: placement `'here'` put new cows at the
herder's cell instead of a random one, and `reset` applies only after a birth; a placement function and member maps
fix both. The meadow stays: a thin subclass cannot rename the roles of its general class (`cows` and `pasture` for
`source` and `target`), and its nested-farm test reads `interaction.pasture`.

### Social classes still on their own laws (2.5)

Rebuilt: majority rule, coevolving voter, join a club, social force walls, evacuation crowding and guidance,
criminality recruitment, the spatial prisoner's dilemma and public goods game, the system traps, employment takeover,
and the sabotage supervisors. Kept, each lacking a capability:
- `StandardizationInteraction`: choice proposals over all options including the current one, N members drawn with
  replacement per step, and utilities reading global sums updated between proposals.
- `BlockMovesInteraction`: a movement schedule of N members drawn with replacement, updating a block state on accept.
- `BelieverElement`: the agent form of `MapInteraction` on a vector state.
- `ContenderElement`: a move into an occupied cell settled by a contest that transfers between the two agents.
- `EmployeeElement`: a conditional discrete relaxation with a reset by life cycle.
- `EaterElement`: moves to anchor locations with transfers at each, and a periodic schedule.
- `ExecutionInteraction`: selection of one member per step by a service policy.
- `DailyMovementsInteraction`: counts grouped by cell and periodic schedules.

### Visits, compositions, and histories (1.4, 1.6)

`PhaseSpec(visits=...)` with the activation `'visits'` runs `int(visits * size)` visits per phase, each drawing an
agent with replacement and running its phase rules, with `ModelView.time` at `k + i/n` during visit `i`; the earlier
perishables rewrite failed because it aged products at whole-step time. `ModelView.composition(name, state)` keeps
counts by value, and `ModelView.history(name)` returns the values of a run observable at earlier steps. Observables and
`agents_` receive agent views. `ChoiceInteraction` and `ReinforcementInteraction` gain `schedule='member'`,
`ReinforcementInteraction` gains `chosen`, and `LifeCycleInteraction` gains `pick` for culls. The reporter-only
interactions of El Farol, Sugarscape, and the labour market stay until their tests read observables instead.

### Experiments replicate interactions (1.7)

`BaseExperiment._replicate` accepts plain single elements with their interactions: the replicated system holds shallow
copies of the interactions, which bind members by index. Per-member values apply to element states and parameters
only, so sweeping a parameter that lives on an interaction keeps the non-vectorized path. `Trajectory.member` copies
entries whose size is the same in the replicated and the original problem.

## Migration Plan

Work on `development`, with parallel agents on disjoint packages merging back as they report.

## Open Questions

- None.
