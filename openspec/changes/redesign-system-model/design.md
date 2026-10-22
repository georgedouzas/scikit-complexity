# Design

## Context

A system today is `System(label, elements, interactions, systems, space, observables, history, noise, phases)`.
Elements and interactions are `Component` dataclasses that declare `state`, `parameter`, and `role` fields and return
laws from `laws(self, s)`.

The compiler infers a `Form` from the set of law kinds through one table, `FORMS` in `compile/_compiler.py`, and
partitions systems whose children have different forms. Every other incompatibility is a local raise somewhere in
compilation or a solver.

`skcomplex.core.model` exports 158 names. About sixty of them name a field rather than a mathematical or modelling
concept.

## Goals / Non-Goals

**Goals:**
- A system is its components, interactions, space, and time. Nothing about how it runs is part of its definition.
- A small vocabulary of general primitives, each named by its mathematics or modelling role, which composes to cover
  every implemented and pending model.
- Extension without editing central tables: a new primitive, solver, or backend declares its requirements and support,
  and validation and documentation follow from the declarations.
- One validation pass that reports every problem with its path, reason, and fix before compilation.

**Non-Goals:**
- New numerical methods. Solvers and kernels keep their algorithms; only their declarations and their selection change.
- Deprecation shims. The package is on 0.1.x, and the migration is done in one change.
- A graphical or acausal modelling language. Acausal components with ports are pending in `add-acausal-components`
  and must map onto this design (see the inventory gate), but they are not built here.

## Decisions

### A system is components, interactions, space, and time

```text
System(label, *, components=(), interactions=(), space=None, time=None)
```

| Part | Holds | Primitives |
|---|---|---|
| Components | everything that carries state | `Element`, `Group`, `System` (nested) |
| Interactions | stateless couplings, bound through roles | `Interaction` |
| Space | the substrate, its boundary conditions, and its sites, edges, and hyperedges | `Space` kinds |
| Time | continuous, discrete with a schedule, or absent | `Time` kinds |

A `System` is itself a component, so composition is recursive. An interaction in a parent refers to a component of a
child by slash path, as today.

Dynamics are not a fifth part. They are the laws of components (intrinsic dynamics) and of interactions (coupling).
Removed from `System`:

| Was | Becomes |
|---|---|
| `elements=`, `systems=` | `components=` |
| `observables=` | `Simulation(observables=...)` and catalog entries; the model factory returns them beside the system |
| `history=` | `state(history=...)` on the delayed state |
| `noise=` | `Noise(convention=...)` |
| `phases=` | `DiscreteTime(schedule=Phases(...))` |
| `MesaSystem` | removed; Mesa is a backend |

Rejected alternative: keeping observables on the system. An observable is a measurement chosen by the user of a model,
not part of the system. Catalog entries still ship a model's standard observables, so `create`/`run` users lose
nothing.

### Components

| Primitive | Meaning |
|---|---|
| `Element` | one entity with declared states, parameters, and intrinsic laws |
| `Group(label, element=, size=, variable=False, on=None)` | `size` identical members held as arrays |
| `System` | a nested system |

Group options:
- `variable=True` lets laws add and remove members. This covers agent birth and death, particle creation, and
  network rewiring.
- `on` places one member per site, edge, or hyperedge of the space.

Field-specific component types dissolve into these:

| Today | Becomes |
|---|---|
| `AgentType` | `Element` whose laws are `Rule`s, in a `variable` group |
| `Particle` | `Element` with position and velocity states in a continuous space |
| `Field` | `Group(on='sites')` over a lattice or mesh; its differential operators stay expressions |
| `Aggregate` | a reduction expression; co-simulation reads it across partitions |
| `QuantumElement`, `Spin`, `Qubit`, `Mode`, `Site` | `Element` with a Hilbert-space state domain; the species move to the library |
| `Word`, `Patch`, `HostGraph` | `Element` with a sequence, tiling, or graph state domain |
| link weights, synapses, hyperedge states | `Group(on='edges')` or `Group(on='hyperedges')` on a network space |

State domains therefore grow to structured values:
- scalar real, integer, Boolean, categorical, angle, complex (existing);
- vector of the space's dimension (existing);
- Hilbert space of a dimension;
- sequence over an alphabet;
- graph;
- tiling;
- tensor of a declared rank and symmetry over the space's dimension (added by the inventory gate).

The domain decides which laws and solvers are valid, through validation.

### Interactions

`Interaction` declares roles and parameters, and returns laws. Every role has an arity and a binding:
- `role()` binds one component or one member;
- `role(many=True)` binds a sequence;
- the binding `over` is one of `'members'`, `'pairs'`, `'neighbours'` (pairs adjacent in the space), or
  `'hyperedges'`.

`PairInteraction` becomes `over='pairs'` or `over='neighbours'`, and higher-order interactions are `over='hyperedges'`.
This change declares the bindings and validates them. The binding `'pairs'` keeps the cutoff search of the former pair
interactions, and `'neighbours'` runs the laws of quantum groups once per bond, as `Exchange` did. Classical laws that
reduce over neighbourhoods, the dynamics of edge states, rewiring, and hyperedge laws are specified by
`add-adaptive-networks` and `add-higher-order-interactions`, which build on these declarations.

Interactions hold no state. Rejected alternative: stateful interactions. These would duplicate state storage, group
vectorisation, and variable size for couplings. Putting coupling state on edge groups reuses all three.

### Space

| Kind | Sites | Notes |
|---|---|---|
| none (well-mixed) | — | the default |
| `Lattice` | regular cells with ordered neighbourhoods | square, hexagonal, triangular, any dimension |
| `Network` | nodes, edges, optional hyperedges | static, or dynamic through variable edge groups |
| `Continuous` | positions in `d` dimensions | boundary conditions `periodic`, `reflecting`, `open` |
| `Mesh` | cells and faces | for finite volumes |

Boundary conditions belong to the space: `PeriodicBox` and `ReflectingBox` become `Continuous(boundary=...)`. Agent
spaces (grid, network, continuous) are these same kinds; an agent group lives on a space like any group.

### Time

| Kind | Meaning |
|---|---|
| `ContinuousTime()` | laws are rates, noise, events, and jumps in continuous time |
| `DiscreteTime(schedule=...)` | laws are updates applied each step; schedules: `'synchronous'`, `'random-sequential'`, `Phases(...)` |
| `None` | static: constraints and objectives only, solved once |

When `time` is omitted, it is inferred from the law classes, and validation reports an ambiguous inference. Declaring
time explicitly is always allowed and always checked. Static time covers the constraint-based, optimal-control, and
static models of the pending changes. This change defines static time and validates it; the static form and its root
and optimisation solvers are specified by `add-static-models`, which builds on it.

### Laws by time semantics

| Class | Laws | Valid under |
|---|---|---|
| Continuous | `Rate`, `Noise(convention=)`, `Equation` (implicit), `Generator` (unitary) and `Dissipator` (Lindblad) on Hilbert domains | continuous time |
| Event | `Jump(clock=exponential or distribution)` (absorbs `Renewal`), `Event` (guard and reset) | continuous time |
| Discrete | `Update`, `Transition` (Markov chain), `Rule` (imperative), `LocalRule` (neighbourhood table), `Rewrite` (productions on structured domains), `Sample` (energy, move, and acceptance; absorbs `Energy`, `Move`, `Sampler`) | discrete time |
| Constraint | `Equation` (algebraic), `Boundary`, `Invariant`, `Objective` (minimised under static or continuous time) | any time |
| Cross-cutting | `Transfer`, `Match` | lowered per time class |

`Hamiltonian` keeps its mathematical meaning under the name `Generator`. Rejected alternative: keeping `Hamiltonian`
as an alias. The application-agnostic test allows physics only where the term is the mathematics itself; the Lindblad
dissipator is, and an energy operator generating unitary flow is better named for its role.

### General primitives that replace field vocabulary

| Primitive | Meaning | Replaces |
|---|---|---|
| `Transfer(amount, source, sink)` | moves a quantity between two states; a pair of opposite rates in continuous time, an atomic update in discrete time; `source` or `sink` may be outside the system | `Flow`, `Source`, `Sink`, `transfer`, the trades of markets |
| `Invariant(label, expression, tolerance=)` | a conserved expression, recorded and checked at every frame; enforcing constraints by projection needs the constraint solvers of `add-static-models` | `Identity`, ledger checks, conserved-quantity checks |
| `Match(left, right, pairing=..., capacity=...)` | pairs the members of one or two groups each step; `pairing` is a function from the two groups to pairs, and the core ships only uniform random pairing | `Market` protocols, matching in mating, labour, and assignment models |
| `parameter(adaptive=True)` | a parameter that laws may change; compiled as a state and recorded | `Institution` |
| `Equation` solved for adaptive parameters | clears an aggregate balance each step | `Walrasian`, `CallAuction` clearing |

### Core packages mirror the system definition

The packages of `skcomplex.core` follow the definition of a system, so a reader finds a primitive where the concept
puts it. The modelling packages hold what a system is made of, and the engineering packages hold how it runs:

| Package | Holds |
|---|---|
| `core.system` | `System`, its composition and nesting, ensembles, and the observables that measure systems |
| `core.components` | `Element`, `Group`, states and parameters, and state domains |
| `core.interactions` | `Interaction`, its bindings, and matching |
| `core.space` | the continuous, lattice, network, and mesh spaces, and the continuous, discrete, and static times |
| `core.laws` | every law, from `Rate` and `Jump` to `Rule`, `Transfer`, `Invariant`, and `Generator` |
| `core.base` | symbolic support, errors, the registry, and the declaration vocabulary |
| `core.compile` | lowering of a validated system to a problem, with no backend |
| `core.solvers` | steppers, with no backend |
| `core.backends` | every backend: NumPy, Numba, LAMMPS, and Mesa, each with its private printers and adapters |
| `core.runtime`, `core.analysis`, `core.viz`, `core.experiments` | running, analysing, drawing, and experimenting |

Laws get their own package because both components and interactions use them, so neither owns them. Space and time
share one package because a system lives in space and time together, and each is a substrate the laws are validated
against. A helper that serves one backend, such as the printer of LAMMPS's Lepton expressions, is private to that
backend's module, not a sibling module that reads as a backend. `skcomplex.core.model` is removed without a shim, like
the other renamed names of this change.

`validate` stays in `core.compile`, because it reads the compiler's census of laws and the registry of solvers, so
placing it in `core.system` would make the bottom of the modelling packages import the engineering packages above
them. Within the modelling packages, every package imports only the ones before it in the order base, space, laws,
components, interactions, system, which a layout test checks.

Rejected alternatives:
- Keeping the flat `core.model`: it hides the definition of a system and mixed general primitives with realizations.
- Splitting the laws between `components` and `interactions`: `Rate`, `Rule`, and most other laws serve both.

### Realizations live in the models

The core holds each primitive in its general form, with at most one neutral default:
- `Match` defaults to random pairing;
- `Sample` defaults to Metropolis acceptance, and also offers the general Glauber and heat-bath rules, which are
  Monte Carlo algorithms rather than realizations of a field;
- `Jump` defaults to an exponential clock.

Every specific realization lives in the `skcomplex.models.<field>.<subfield>` module of the models that use it. This
includes:
- a named potential, force law, or thermostat;
- a pairing algorithm, such as stable matching, an order book, or an auction;
- a rule code or neighbourhood rule family;
- a quantum species or operator set;
- a learning heuristic;
- a tiling, turtle, or production family;
- a waiting-time distribution beyond the default.

A realization that several model packages of one field share lives in `skcomplex.models.<field>.realizations`, which
only that field's packages import, so model packages stay otherwise self-contained. A package of another field that
needs the same realization defines its own.

The core depends on no models module. Its tests define their realizations inline, and an import-graph test checks that
the core never imports `skcomplex.models`.

These are library compositions, not core names:

| Moved to `skcomplex.models` | Built on |
|---|---|
| markets, order books, auctions, institutions | `Match`, `Transfer`, `Invariant`, adaptive parameters, `Equation` |
| `logit_choice`, `imitate`, `evolve`, `heuristics` | draws, lags, reductions |
| `Gravity`, `Spring`, `LennardJones`, `SoftSphere`, `LinearDrag`, `ConstantForce`, `HarmonicPotential`, `LangevinThermostat` | `Potential`, pair interactions, noise |
| `Spin`, `Qubit`, `Mode`, `Site` | Hilbert domains, `Generator`, local operators |
| `Wolfram` codes and life-like rules | `LocalRule` with a `Table` or an `OuterTotalistic` rule; a rule code is any object whose `table()` returns a table |
| `smooth`, `smooth3`, `delay1`, `delay3` | the distributed-delay expression |

Reviewing the moves against the code kept some names in the core, because they are general forms rather than
realizations:
- `Exchange` is removed, not moved: an interaction with `over = 'neighbours'` and one role does what it did, and
  `neighbour_pairs` lists the bonds of a lattice or network.
- `Table`, `Totalistic`, and `OuterTotalistic` are general lookup rules that the compiler fuses into kernels.
- `Prototile`, `Affine`, and `Substitution` are the rewrite law of the tiling domain, as `Production` is of sequences
  and `GraphProduction` of graphs; named tilings live in the models.
- `Turtle` interprets words as geometry, so it moves to `core.viz` as a visual in the package restructuring.
- `Drive`, `Relaxation`, and `Extremal` are general laws of driven, threshold, and extremal dynamics, which the lattice
  stepper runs; lowering them to `Update` with member selection is left to a later change, since it would rewrite the
  lattice compiler without changing what a model can say.
- `Particle`, `Potential`, and `PairPotential` are the general forms the named forces build on.
- `QuantumSystem` is removed, because a system holds only components, interactions, space, and time. Its whole-system
  initial state is a simulation's `initial_state`, by dotted state name, which classical systems use too, and its
  `max_dimension` is an argument of the quantum solvers.
- Backends that run a realization natively read it from registries the realization fills, such as
  `register_lammps_pair`, so no backend names a realization.

### The application-agnostic rule

A core name is a mathematical term (rate, noise, generator, invariant), a modelling term (element, group, interaction,
agent rule, matching), or an engineering term (solver, backend). A test holds a reviewed list of forbidden field terms
(market, bank, firm, predator, gravity, qubit, turtle, and others) and fails if any core public name or core module
contains one. New field terms are added to the list when a review finds them.

### Validation is declarative

Every primitive, solver, and backend declares a frozen record:

```text
Requires(time={...}, space={...}, domains={...}, arity={...}, laws={...})
Supports(time={...}, space={...}, domains={...}, laws={...}, groups={'fixed', 'variable'})
```

`validate(system, *, solver=None, backend=None) -> ValidationReport` walks the system once and checks:

1. **Structure:** unique labels, resolvable paths, every role bound, binding arity met.
2. **Components with space:** for example, a vector state with no space, or `on='edges'` without a network.
3. **Laws with time:** for example, a `Rate` under discrete time, or an `Update` under continuous time.
4. **Laws with domains:** for example, a `Rate` on a categorical state, or a `Generator` on a real state.
5. **Laws with laws:** combinations within one partition that no solver supports, with the partition boundaries that
   would make them valid.
6. **The system with the solver, and the solver with the backend:** the chosen or `'auto'` solver supports every
   requirement, and the backend supports the solver.
7. **Availability:** optional backends are importable.

Each problem is an `Issue(code, path, message, fix, severity)`:
- `code` is stable and documented;
- `path` is the slash path of the component or law;
- `fix` names the change that makes the system valid.

`compile_system` and `Simulation` call `validate` first and raise one `ValidationError` that lists every error. Warnings
are emitted as named warnings.

Adding a solver or backend extends what is valid without editing the validator. The compiler keeps its table from the
law types it lowers to forms: forms are a lowering choice below the system's definition, and they turn on distinctions
the vocabulary deliberately does not carry, such as an extremal update that needs the lattice stepper against a plain
update, or a jump on an exponential clock against one on a clock of waiting times. Deriving forms from the coarser
declarations would pick wrong forms or force lowering detail into every declaration, so validation reads the
declarations and the compiler reads the table.

The documentation page of valid combinations is generated from the declarations, and a test fails if any registered
primitive, solver, or backend has no declaration.

Rejected alternative: a hand-written compatibility table. It is the source of the current drift, and it cannot be
extended by a plug-in backend.

### Mesa is a backend

Agent systems are ordinary systems: elements with `Rule` laws in variable groups, discrete time, and a space. Mesa is
the backend that executes discrete-time systems with imperative rules, and it declares that support like any other
backend.

`MesaSystem` and the wrapping of arbitrary Mesa model classes are removed. Its only users were tests and one gallery
example, and that example is ported.

### Inventory gate

Before any code changes, every inventory entry is re-tagged with the new vocabulary: the 438 implemented models and the
entries of the 26 pending changes. A script reports entries that do not map, and each one either maps after a review or
adds a general primitive to this design. The models most likely to strain the design:
- acausal ports;
- optimal-control objectives;
- multiscale coupling across time classes;
- dynamic hypergraphs.

**Result of the gate.** All 537 entries are tagged: the 438 implemented models from their compiled systems, and the 99
others from their curated primitives. Every implemented model built and compiled during the probe. Four terms did not
map, and each was resolved by a general primitive:

| Unmapped need | Pending change | Resolution |
|---|---|---|
| tensor-valued states | `add-tensor-fields` | the `tensor` domain, with rank and symmetry |
| quantities to minimise | `add-optimal-control`, `add-constraint-based-models`, `add-static-models` | the `objective` law in the constraint class, valid under static time (optimisation) and continuous time (control over a horizon) |
| Caputo derivatives | `add-fractional-dynamics` | a non-integer `order` of a state, which rate laws target unchanged; solvers declare support for fractional orders |
| parameters fitted by an inner loss | `add-learning-dynamics` | `parameter(adaptive=True)` changed by update laws, the inner model a nested system |

Four further needs mapped without a new primitive:
- acausal ports are interactions over members with algebraic equations;
- distribution states are site groups on a lattice over the product space;
- link and plaquette states are edge and hyperedge groups;
- variational mechanics derives equation laws.

The vocabulary and the tags live in `docs/overview/inventory.toml`, and the inventory check fails on any term outside
it.

The baseline of the migration holds the trajectory of every catalog model run with random state 0 at commit
`73093d9`: times, states, numeric observables, and reporters. It is stored outside the repository, in
`~/.cache/skcomplex-work/baseline-73093d9`, with the script that records it, and every model recorded without error.

## Risks / Trade-offs

- **The migration is large: 438 models, gallery, and tests.** Mitigation: migrate field by field behind the
  conformance suite, which already runs every model, and compare trajectories before and after on fixed seeds.
- **Generic names are less discoverable for domain users.** Mitigation: the library keeps the domain building blocks
  under their domain names, so `skcomplex.models.economics` still offers an order book.
- **Explicit time adds a choice for simple models.** Mitigation: time is inferred when the law classes are unambiguous,
  and validation explains an ambiguous inference.
- **The pending changes become stale.** Mitigation: rebase their artifacts onto the new vocabulary in this change's
  final task, before any of them is applied.

## Migration Plan

1. Inventory gate and declarations, with no behaviour change.
2. Validation pass behind the existing compiler; existing raises are retired as the validator covers them.
3. New system signature, space and time, and law classes in the core, migrating the models field by field.
4. Move field primitives to the library.
5. Remove `MesaSystem` and the old names.
6. Restructure the core packages to mirror the system definition, and gather every backend in `core.backends`.
7. Rebase the pending changes.
