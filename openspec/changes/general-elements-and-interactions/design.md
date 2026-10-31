# Design

## Context

See proposal.md for the motivation. The current state that shapes the approach:
- `skcomplex.core.elements` holds `Element` and the containers `GroupElement`, `FieldElement`, `LatticeElement`, and
  `ParticleElement`. `skcomplex.core.interactions` holds the bindings `Interaction`, `PairInteraction`,
  `NeighborInteraction`, and `HyperedgeInteraction`, and the mechanics bases `PotentialInteraction` and
  `PairPotentialInteraction`, which hard-code the states `x`, `v`, and `m`.
- A part declares its states, parameters, and roles with `Annotated[..., StateSpec() | ParameterSpec() | RoleSpec()]`
  in `__init__`, which only stores its arguments, and returns its laws from `_get_laws(namespace)`. Every part
  implements `_check`, `_prepare`, and `_run`, itself or through its family base, and `run` sets `laws_`.
- The shared code to move is:
  - `models/economics/common/markets`: `MarketInteraction`, `BaseMarketProtocol`, seven protocols, `ClearingSpec`,
    `Order` (a mutable dataclass), and `Trade`.
  - `models/economics/common/institutions`: `InstitutionInteraction`.
  - `models/economics/common/learning`: `logit_choice`, `logit_probabilities`, `evolve`, and the forecasting
    heuristics.
  - `models/physics/common/particles`: the force and potential classes and their LAMMPS translations.
- The import contracts forbid a model package from importing another package, except its field's `common`, and
  forbid the core from importing models. `tests/test_types.py` and the field-term test guard the core's names.

## Goals / Non-Goals

**Goals:**
- Each shared mechanism exists once, in the core, under a mathematical name. Every model that implements it uses it.
- Every model factory keeps its name, parameters, and trajectories within the tolerance of its tests.

**Non-Goals:**
- New models, or changes to what a model computes.
- Mechanisms that only one field uses. They stay in that field, in its `common` package when several of its packages
  share them.
- Gradient learning, which `add-learning-dynamics` covers, and k-body tensors, which `add-higher-order-interactions`
  covers.

## Decisions

### Mechanisms are interactions; elements only hold states
A mechanism changes states, so it is an interaction that binds the groups it acts on by roles and takes the names of
the states it reads and writes as constructor arguments. Each is a subclass of the binding it needs:
- `Interaction`: reaction, growth, Ornstein-Uhlenbeck, damping, constant force, choice, evolution, feedback,
  matching.
- `NeighborInteraction`: spreading, imitation, network coupling, threshold, on the extended edge kernel below.
- `PairInteraction`: pair force, alignment within a radius.
- `HyperedgeInteraction`: group mechanisms.

Elements are added only where several fields declare the same states: `DensityElement`
and `CompartmentElement`. A model that needs other states keeps its own element and binds the core interactions to it.

Alternative considered: mechanisms as elements with built-in laws, such as a `WalkerElement`. Rejected, because a
model then cannot combine two mechanisms on one element without multiple inheritance.

### Neighbour interactions take what the mechanisms need
Spreading, imitation, network coupling, and thresholds stay `NeighborInteraction` subclasses, so the engine extends the
binding instead of the mechanisms falling back to plain interactions:
- Per-member laws: an edge kernel takes laws of the members at either end that do not depend on the edge, such as
  recovery and loss of immunity, compiled once per member rather than once per edge.
- Degree: the edge namespace offers the degree of the source and the target, so a rate can be proportional to the
  fraction of neighbours.
- Delays: `delayed` reads the past state of the source or the target of an edge.
- Relaxation: an edge kernel takes `RelaxationLaw`, `DriveLaw`, and `LocalRuleLaw`, so cascades run over the edges of a
  network and are recorded as events, as on lattices.

### Reductions name the reading member, and reduce to an argmax
- A reduction over a group can read the member that reads it, apart from the member being reduced, so
  `mean(f(other.x, x), over=...)` evaluates any coupling function `f` against the reading member. The mean field of
  `CouplingInteraction` rests on it for any function, not only separable ones.
- A reduction returns the state of the neighbour at which an expression is largest, which `'best'` imitation reads.

### Mechanisms serve agent groups and symbolic groups
A mechanism that agent models use lowers to a `RuleLaw` when its role binds an agent group, and to its symbolic laws
otherwise. In the agent form, the functions it takes, such as a utility, are callables of the agent and the model view,
so they can read the model's history and the agent's neighbours. One class serves both, and the bound group decides
the lowering.

### Variants are named arguments, checked in `_check`
A variant that changes the form of the laws is a string argument checked against a module constant of the allowed
names, such as `step='levy'`, `form='gompertz'`, `rule='fermi'`, or `mode='jump'`. A variant that carries its own
settings and algorithm is a configurable class, such as the matching protocols, which keep their `BaseConfig` contract
and `trades_` result. Every name is mathematical.

Alternative considered: one subclass per variant, such as `LogisticGrowthInteraction`. Rejected, because it
multiplies classes and brings field-shaped names back.

### Functions are user arguments
Where a mechanism is defined by a function, the user passes it as a callable of the namespace, the same way
`CouplingInteraction(function=...)`, `PairForceInteraction(energy=...)`, and `ChoiceInteraction(utility=...)` work.
The neutral default is the identity difference for coupling, and no default for energies and utilities.

### Matching keeps its algorithm and changes its words
`MarketInteraction` becomes `MatchingInteraction(label, *, requesters, providers, protocol=None, limit='limit',
quantity='quantity', budget='budget', transfers=(), phase=None, on_match=None)`. The bid and ask become one `limit`
state per side, read by name. `money` and `goods` become `transfers`, a sequence of `(state, sign)` per side that a
match moves by price times quantity or by quantity.

The protocols keep their algorithms under the names in the proposal. `Order` becomes `_Offer`, a private record of
the matching module: a clearing lowers its quantity and budget, so it cannot be a frozen spec. `Trade` becomes
`MatchSpec`, and `ClearingSpec` stays. The economics models pass
`limit='bid'` and `limit='ask'`, and `transfers=(('money', ...), ('goods', ...))`.

### Learning splits by generality
- `logit_choice` and `logit_probabilities` become the `'logit'` rule of `ChoiceInteraction`.
- `evolve` becomes `EvolutionInteraction`.
- The forecasting heuristics (adaptive, trend, anchoring, fundamental) are a field realization and stay in
  `economics/common/learning`, renamed `forecasting`.

Module functions become private methods of the interaction that uses them, by the project's helpers-as-methods rule.

### Physics particles are built on the core
- `ConstantForceInteraction` moves to the core.
- `LinearDragInteraction` becomes the core `DampingInteraction`.
- `HarmonicPotentialInteraction`, `SpringInteraction`, `GravityInteraction`, `LennardJonesInteraction`, and
  `SoftSphereInteraction` stay in `physics/common/particles` as subclasses of `PotentialInteraction` or
  `PairForceInteraction` with their energy, and keep their LAMMPS translations.
- `LangevinThermostatInteraction` becomes a subclass of `OrnsteinUhlenbeckInteraction` on the velocity. Its rate is
  the friction and its noise is fixed by the temperature.
- Biology's `MorsePotentialInteraction` and social's repulsion classes become `PairForceInteraction` with their
  energies.

### Potentials take state names
`PotentialInteraction(label, *, position='x', velocity='v', mass='m')` and the same arguments on
`PairPotentialInteraction` and `PairForceInteraction`. The symbolic gradient is taken with respect to the named
position, and its effect goes to the named velocity divided by the named mass.

### Group mechanisms on hyperedges
The Galam groups, public-goods groups, Kolkata restaurants, and minority-game groups are built as hyperedges of a
`NetworkSpace(hyperedges=...)`, redrawn each round where the model reshuffles groups. Each becomes a
`HyperedgeInteraction` subclass whose laws read the members of one hyperedge. Where reshuffling needs a new hyperedge
set every round, the model updates the space's hyperedges through the model view, and the change verifies that
compilation supports it. If it does not, that model keeps its `RuleLaw` and the gap is reported, rather than adding a
mechanism to the space in this change.

### Duplicates resolve by reach
- A named equation, such as Stuart-Landau, Van der Pol, or the logistic map, is a realization, not a core primitive.
  Each field keeps its own class, since a field never imports another field's package.
- An equation that one field uses twice goes to that field's `common` package, or stays next to its one user. The
  physics Stuart-Landau copies become one in `physics/common`. FitzHugh-Nagumo moves to
  `biology/common/excitable`.
- The meadow and commons models keep their elements and use the core foraging and regrowth.
- Both `ExponentialGrowthInteraction` classes become `GrowthInteraction(form='exponential')`.

### Rewriting models
Each model class in `inventory.md`, the appendix of this design, is rewritten to bind the core class. Its own element keeps only the
states, parameters, and laws the core class does not give. A class that becomes empty is deleted, and its factory
builds the core class directly. Classes listed as "composite" keep their element and bind the core interaction for
the part of their laws it covers, and classes listed as "excluded" stay as they are. Trajectory tests and doctests are the acceptance check: every factory's existing tests
pass unchanged, except imports of moved names.

## Risks / Trade-offs

- **[Risk] Rewritten models change their numbers.** Stochastic models draw in a different order once the core class
  draws. → Seeds stay, and tests that compare exact draws are rerun before and after. A model whose test tolerance
  fails is compared statistically against its previous trajectory over 20 seeds, and only a statistically equal
  model is accepted.
- **[Risk] Generalised classes grow many arguments.** → Each class takes at most one variant argument and the state
  names. A model that needs more subclasses the core class instead of the core class gaining a flag.
- **[Risk] Field-term creep.** → The field-term test runs on every new name, and review rejects names like
  `Infection`, `Market`, `Price`, `Predator`, or `Neuron` in the core.
- **[Risk] Overlap with pending changes.** → `add-higher-order-interactions` and this change touch hyperedges. The one
  that lands second rebases on the other.
- **[Trade-off] Matching speaks of limits and quantities, not bids and asks.** Economics models gain a few binding
  arguments. In exchange, engineering's dispatch and the social models use the same protocols.

## Migration Plan

Work happens on `development`. Parallel agents each work on a worktree branch cut from `development`, which merges back as
soon as the agent reports and its tests pass. Moved public names are not kept as aliases: imports change in the same commit as the move. Rollback is reverting the merges.

## Open Questions

- Whether compilation supports changing a network's hyperedges during a run, which the reshuffled group mechanisms
  need. Answered by task 5.1: the engine validates that a hyperedge interaction binds members of a network with
  hyperedges, but compiles its laws as those of a plain interaction, once per member, and ignores the hyperedges. Laws
  run once per hyperedge are the scope of the pending `add-higher-order-interactions` change, so the group mechanisms
  of section H are blocked by it, whether or not their groups are reshuffled.
