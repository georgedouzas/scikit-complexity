# Proposal

## Why

The core gives models containers (`Element`, `GroupElement`, `FieldElement`, `LatticeElement`, `ParticleElement`) and
binding patterns (`Interaction`, `PairInteraction`, `NeighborInteraction`, `HyperedgeInteraction`), but no mechanism a
model can use as it is. So the models define about 500 element classes and 154 interaction classes, and the same
mechanisms come back across fields:
- random walks of agents in 27 classes across 5 fields, with one private `_wander` helper copied four times;
- mass-action reactions and compartment flows in 18 classes across 6 fields;
- coupling of states (Kuramoto, consensus, alignment, power lines) in 18 classes across 5 fields.

Some are copied outright: `CowElement`, `PastureElement`, and `GrazingInteraction` exist in both biology and
economics, and `StuartLandauElement` twice in physics. Two specs cause much of this:
- "Specific realizations live in the model library" makes a field that needs another field's shared code define its
  own copy.
- "The core is application agnostic" allows a core primitive only one neutral default.

So the logit choice of economics, the potentials of physics, and the market matching protocols are rewritten
wherever another field needs them.

## What Changes

- **General operations in the core.** The core gains one class per mathematical operation, configured by the states it
  binds and by functions the user passes, and named by the operation, never by a quantity of a field. design.md lists
  them with their signatures:
  - continuous time: `RateInteraction`, `RelaxationInteraction`, `GradientInteraction`, `EulerLagrangeInteraction`,
    `ConservationInteraction`, `DiffusionInteraction`;
  - discrete time: `MapInteraction`, `LocalRuleInteraction`, `CascadeInteraction`;
  - coupling: `CouplingInteraction` over edges, groups, and ordered neighbours, and `PairCouplingInteraction` over pairs
    within a cutoff, with a function or a pair potential;
  - events and discrete states: `JumpInteraction`, `ReactionInteraction`, `TransitionInteraction`,
    `SpreadingInteraction`, `EncounterInteraction`, `CrossingInteraction`, `ImpulseInteraction`, `SamplingInteraction`;
  - choice and learning: `ChoiceInteraction`, `ImitationInteraction`, `ReinforcementInteraction`,
    `EvolutionInteraction`;
  - agents, groups, and exchange: `MovementInteraction`, `TransferInteraction`, `LifeCycleInteraction`,
    `GroupingInteraction`, `MatchingInteraction` with eight protocols, `FeedbackInteraction`, `TopologyInteraction`;
  - elements: `CompartmentElement`, `DensityElement`, and `PointElement`, which replaces `ParticleElement`.
- **No quantities in the core.** A force is a rate on a state of order 2, and its mass is a divisor. **BREAKING:**
  `PotentialInteraction`, `PairPotentialInteraction`, `ParticleElement`, and `StateSpec.velocity_of` leave the core, and
  the mechanisms first added by this change that named field quantities move to the fields as subclasses of the
  general classes: forces, damping, the Ornstein-Uhlenbeck process, growth forms, foraging, regrowth, alignment, and
  named oscillators.
- **Laws and utilities.** `RelaxationLaw` becomes `CascadeLaw`, and the laws with field wording lose it. The law
  namespace gains `hill`, `logistic`, `pulse_train`, `clip`, `minimum`, `maximum`, `sign`, the stock-and-flow delays,
  keyed reductions, and a circular mean.
- **BREAKING: matching, feedback, and evolution move to the core** with neutral names: `MarketInteraction` becomes
  `MatchingInteraction`, `InstitutionInteraction` becomes `FeedbackInteraction`, and `evolve` becomes
  `EvolutionInteraction`. The protocols are named by their algorithm.
- **Every model is rebuilt on the general classes.** Every model class uses the general classes and keeps only what is
  its own, so the models become examples of the core. A realization that several packages of one field share moves to
  the field's `common` package. The six audits in `inventory/` list every class.
- **Group mechanisms run as agent groupings** until `add-higher-order-interactions` compiles laws per hyperedge.

## Capabilities

### New Capabilities

- `general-mechanisms`: the general elements and interactions the core provides, one per mathematical operation,
  each configured by states, functions, and named variants, which together cover every model.

### Modified Capabilities

- `system-modeling`:
  - "The core is application agnostic" now allows a core mechanism standard variants named by their mathematics.
  - Matching protocols and choice rules count as general mechanisms, not as specific realizations.
- `model-library`:
  - "Specific realizations live in the model library": a mechanism that models of more than one field need lives in
    the core instead of being copied. A field's `common` package holds only realizations named by the field.
  - "Pair potentials for particle groups" and "Langevin thermostat" are realizations built on the core interactions.
  - "Economic and social agent models" use the core's matching, feedback, choice, and evolution.

## Impact

- **Code:** `skcomplex.core.elements`, `skcomplex.core.interactions`, and `skcomplex.core.laws` are rebuilt as above.
  The engine reads the divisor, the order-2 states, and the agent form. LAMMPS accepts elements by their states.
  Every model package is rewritten, and each field gains `common` families for its shared realizations.
- **API:** breaking for the moved and removed names above. Model factories keep their names and parameters, and their
  trajectories stay within the tolerance of their tests.
- **Overlap with pending changes:** `add-higher-order-interactions` compiles laws per hyperedge, after which the group
  mechanisms can move from agent groupings to hyperedges. `add-learning-dynamics` adds gradient learning, which is
  distinct from `EvolutionInteraction` and `ReinforcementInteraction`.
