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

### Couplings of separate elements and over radii (1.1, 1.5)

In mode `'group'`, `CouplingInteraction` binds a list of separate elements coupled to each other, a single hub element
reduced over a group, or a group reading separate elements; the reduction over separate elements is written term by
term. Its agent form (`form='agent'`) runs per agent with `radii` of `RadiusSpec(radius, function, reduce, priority)`,
counting only the radii of the highest priority that hold a neighbour, then a `ClampSpec` bounds the turn angle and the
norm; `combine(agent, model, reduction, priority)` defaults to `v + reduction`. Winners take all, LEGION, the power
grid lines, the contour cell, boids, and the Couzin model run on it.

### Roles resolved through the subclass (2.10)

`BaseInteraction._get_role(role)` returns the role unchanged by default; every agent-form lookup of a role by a fixed
name (`model.agents`, `model.groups`, `model.create`, `model.deposit`, `RuleLaw(members=...)`) and the default
`TransferInteraction._get_holders` go through it, so a subclass that only renames its roles overrides `_get_role` in
both forms. The meadow's `GrazingInteraction` (a cell-paired transfer from `pasture` to a sink) and
`HerdingInteraction` (a movement to the nearest free grown patch, removing a cow with none) run on it, equal over 20
seeds with unchanged tests. Parameters are still read by fixed names, so a movement subclass declares `probability`.

### Core extensions and social rebuilds (2.9)

`ChoiceInteraction(proposal='all')` draws proposals over all options including the current one (`'fermi'` with
`'all'` is the heat-bath rule); the agent form of `MapInteraction` maps vector states; `PhaseSpec(period, offset)`
runs a phase every `period` steps; `ModelView.occupancy(name)` counts agents per location. Beliefs, standardization,
segregation blocks, the sabotage employee, and the trafficking bills run on them, identical or equal over 20 seeds.
The obesity `EaterElement` waits: its meal needs a partner chosen within reach or else the nearest, and legs that
credit both ends from one shared draw.

### Observables and economics transfers (1.6, 2.8)

El Farol, Sugarscape, and the labour market read their catalog observables (`attendance`, `population` and `gini`,
`searching`); El Farol's predictors read `model.history('attendance')`. The economics transfer interactions keep only
their legs, their parameters live on the elements, and the factories build the other mechanisms as rate and map
interactions. The stock-flow consistent economy is bit-identical; the beer game matches within 1.8e-12, since each
interaction builds its own hidden delay stages, so the split duplicates the mail, shipping, and production delay
chains. The meadow still waits: the agent forms of transfer, movement, and life cycle read their roles by the fixed
names `source` and `target`, and `_get_holders` serves the symbolic form only.

### Transfers along edges (1.3)

`TransferInteraction` gains `offset` (every member transfers to the member `offset` places along the group order),
`boundary` (`'periodic'` or `'fixed'`), and `reverse_rate` (jumps back, reported as `<flow>_reverse`); a symbolic leg
end `'<role>.<state>'` names a state of any role, and subclasses with other role names give their pairs through
`_get_holders`. A draw inside `shift_member` reads the draw of the member at the offset, so both ends of an edge share
one draw. Flow reporters of drawn amounts record the expected amount. The zero-range, exclusion, reservoir, and TASEP
processes, the beer game, and the stock-flow consistent economy run on it; the economics subclasses still add their
own rate and update laws next to their transfer legs.

### Gallery budget (2.7)

Pair reductions that factor into member and other-member parts lower to one linear reduction per part, and class
declarations are parsed once per class; the reflecting floor and the bank borrowers are computed once per step. The
unbounded-alphabet example draws the symmetric Pascal triangle, since the band check failed before the rebuild too.

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

### Decisions of 2026-11-01, second round

- **No field mechanism enters the core.** A model interaction does one operation: the economics transfer subclasses
  keep only their transfer legs, their parameters move to their elements, and each other mechanism is a general
  interaction in the factory.
- **Reporter-only interactions go.** The El Farol, Sugarscape, and labour market tests read the catalog observables,
  as the perishables test does, a test change beyond imports that the maintainer accepted.
- **The meadow uses `_get_holders`.** Its grazing is a `TransferInteraction` subclass with the roles `pasture` and
  `cows`, so its test stays unchanged.
- **Four core extensions pass the entry rule** (a fundamental paradigm usable across domains): the heat-bath choice
  variant whose proposals include the current option, the agent form of `MapInteraction` on vector states, periodic
  schedules, and counts grouped by cell beside `composition`. The hierarchy contest and the queue service stay in their
  models; the employee is a composition of existing classes.

### Decisions of 2026-11-01, third round

- **Thin subclasses name their own roles in both forms.** The agent forms of the general interactions resolve their
  roles through the subclass, as `_get_holders` does for the symbolic transfer, so the meadow's grazing is a transfer
  subclass with the roles `pasture` and `cows` and its test stays unchanged.
- **The obesity `EaterElement` stays in its model.** Its meal pairs a partner within reach or else the nearest
  restaurant and credits both ends from one draw, a rule specific to that model, like the hierarchy contest and the
  queue service.

### Independent random streams (2026-11-01)

Models seeded their initial states with `default_rng(s)` and ran `Simulation(random_state=s)`, so the first draws of
the dynamics repeated the numbers that built the initial state. `Simulation` derives its stream from
`SeedSequence(random_state).spawn(...)` instead, which fixes it for user code as well as for the library. Every seeded
trajectory changes once, so tests with fixed expected values take new numbers. It runs after the rebuilds that
compare against the current trajectories.

### Library audit (2026-11-01)

An audit of the code and of every open change for reimplemented library features found little. Adopted: scipy's
connected components for periodic cluster merging (fixing missed diagonal joins across the boundary), igraph's
RAND-ESU for motif counting (GPL, like passagemath), and library generators already among the dependencies for five
hand-built ones (XGI hypergraph and simplicial complex, ReservoirPy reservoir weights, SciPy Cauchy quantiles, NetworkX
random graph), done with the random-stream change since they shift seeded trajectories. Rejected: CasADi for optimal
control (a second symbolic lowering beside Sage) and pymittagleffler (too little use). Kept: the Numba solvers,
recurrence analysis, the L-system engine, automata on groups, early warnings, and the Markov and power-law helpers,
for which no mature library fits.

## Migration Plan

Work on `development`, with parallel agents on disjoint packages merging back as they report.

## Open Questions

- None.
