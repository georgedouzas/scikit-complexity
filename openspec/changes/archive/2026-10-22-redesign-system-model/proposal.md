# Proposal

## Why

The core has grown one primitive at a time, each added for the model that needed it. The result is 158 names in
`skcomplex.core.model`, and they mix three kinds of thing:

- the definition of a system;
- how a system runs;
- the vocabulary of particular fields.

**Definition mixed with execution.** `System` takes a noise convention, a delay history, agent phases, and observables
next to its elements and interactions. `MesaSystem` is a system that only wraps an execution engine.

**Field vocabulary in the core.** `Market`, `OrderBook`, `Walrasian`, `Institution`, `Gravity`, `LennardJones`,
`Qubit`, `Spin`, `Wolfram`, `Turtle`, `logit_choice`, `imitate`, `transfer`, `Source`, `Sink` and `smooth3` name
economics, physics, quantum computing, cellular automata, L-systems and system dynamics. Each new field adds more of
them. These names hide the general primitive underneath: a market is matching plus a conserved transfer, and an
institution is a parameter that changes under a law.

**Validation scattered through the compiler.** Not every combination of components, interactions, spaces, solvers,
and backends is valid. Today the invalid ones are found by about 200 `ValidationError` raises spread through the
compiler and solvers, and the user sees them one at a time, often deep inside compilation. The only declared
compatibility table is the map from law kinds to forms.

The core exists to describe any complex system, and the models exist to reveal what the core needs. That needs a
small, application-agnostic vocabulary in which a system is defined by what it is, not by how it is run, and a
validator that knows which combinations are valid.

## What Changes

- **BREAKING** A system SHALL be defined by four parts and nothing else:
  - **components**: elements, groups of elements, and nested systems;
  - **interactions**: couplings between components bound through roles;
  - **space**: the substrate the components live on;
  - **time**: continuous, discrete with a schedule, or absent.

  Dynamics are typed **laws** attached to components and interactions. Laws are classified by how they treat time:
  continuous, discrete, event, and constraint.
- **BREAKING** Everything else moves out of `System`:
  - noise conventions become an argument of noise laws;
  - delay histories become an argument of the states that need them;
  - agent phases become the schedule of discrete time;
  - observables move to simulations and catalog entries;
  - forms, solvers, and backends become execution concerns.
- **BREAKING** `MesaSystem` is removed. Agent systems are ordinary systems, and Mesa is an execution backend chosen like
  any other.
- **BREAKING** Every core name is application agnostic: a mathematical or modelling term, never a field's term. The
  field-specific primitives move to `skcomplex.models` as library building blocks over general primitives:
  - markets and institutions;
  - learning heuristics;
  - named potentials and thermostats;
  - quantum species;
  - turtles and tilings;
  - Wolfram and totalistic rule codes;
  - system-dynamics names.
- New general primitives replace the field-specific ones:
  - **transfer laws** move a quantity between states under any time semantics;
  - **invariants** declare conserved sums;
  - **matching laws** pair the members of groups;
  - **adaptive parameters** may be changed by laws.
- Interactions stay stateless. State on couplings, such as link weights, synapses, and hyperedges, is held by groups of
  components that live on the edges or hyperedges of a network space. Groups may have a variable size, which covers
  agent birth and death and network rewiring.
- New capability `system-validation`. Every primitive, solver, and backend declares what it requires and supports:
  - time kind;
  - space kind;
  - state domains;
  - role arity;
  - law kinds.

  One validation pass computes compatibility from these declarations before compilation. It reports every problem at
  once, each with the component path, the reason, and a fix, and the same declarations generate the documentation
  table of valid combinations.
- **BREAKING** The core packages mirror the system definition: `core.system`, `core.components`,
  `core.interactions`, `core.space` (space and time), and `core.laws` replace `core.model`, and every backend, NumPy,
  Numba, LAMMPS, and Mesa, lives in `core.backends`.
- Acceptance gate: every inventory entry, the implemented models and those of the pending changes, maps to the new
  vocabulary before any code changes. Any entry that does not map exposes a missing general primitive, which is
  specified here first.

## Capabilities

### New Capabilities

- `system-validation`: declared requirements and support, the validation pass, the report, and the generated
  compatibility table.

### Modified Capabilities

- `system-modeling`:
  - the four-part system;
  - components, including variable groups and edge groups;
  - space and time;
  - law classes;
  - transfers, invariants, matching, and adaptive parameters;
  - the application-agnostic rule.
- `model-compilation`: the form is derived from validated laws and time, and is internal.
- `agent-based-modeling`: Mesa is a backend, phases are a time schedule, and markets, institutions, and learning leave
  the core.
- `stocks-and-flows`: flows and ledgers become transfers and invariants.
- `simulation`: observables are given to simulations, not to systems.
- `model-library`: the field-specific building blocks live in the library.
- `model-inventory`: entries list primitives from the new vocabulary, and coverage is reported by its categories.

## Impact

- `skcomplex.core.model` is replaced by `core.system`, `core.components`, `core.interactions`, `core.space`, and
  `core.laws`; backends move from `core.compile` and `core.solvers` to `core.backends`; `skcomplex.core.base`,
  `skcomplex.core.compile`, `skcomplex.core.solvers`, and `skcomplex.core.runtime` change their public names. This is
  a breaking change on the 0.x line, without shims.
- All 438 library models and their gallery examples are migrated to the new vocabulary.
- The 26 pending OpenSpec changes are rebased onto the new vocabulary after this change, before any of them is applied.
- `docs/overview/inventory.toml` replaces its flat primitive vocabulary with the categories of the new design.
