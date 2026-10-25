# Design

## Context

- **Where the base classes are today:**

  | Base | Module | Derived classes |
  |---|---|---|
  | `Component` | `core/components/_component.py` | `Element`, `Group`, `Interaction`, `System` |
  | `Law` | `core/laws/_law.py`, with `register_laws`, `law_classes`, `is_registered`, `has_trait` | every law |
  | `Space` | `core/space/_spaces.py` | `SiteSpace`, `ContinuousSpace`, the lattices, the networks, `Mesh` |
  | `Time` | `core/space/_time.py` | `ContinuousTime`, `DiscreteTime`, `StaticTime` |
  | `Stepper` | `engine/solvers/_stepper.py` | every solver, and `ExternalBackend` |
  | `Backend` | `engine/compile/_backends.py` | `NumpyBackend`, `NumbaBackend` |
  | `Problem` | `engine/compile/_problem.py` | `KineticProblem`, `QuantumProblem`, `RewritingProblem` |
  | `Pass`, `SystemPass` | `engine/compile` | the compile passes, the system passes of quantum and rewriting |

- **What `Component` gives each class:**
  - `Element`, `Group`, and `Interaction` use all of it: the label, the declared states, parameters, and roles, the
    declaration-based constructor, `get_params`, `define`, and `laws`.
  - `System` uses only the label.
- **How `System` holds its parts:** it takes `components=` (elements, groups, and systems) and `interactions=` as
  separate lists.
- **`Problem`:** a dataclass of about 50 fields.
  - Every solver reads `form`, `states`, `initial_state`, and `initial_parameters`.
  - The rest are read by the solvers of one form or one formalism.
  - `Layout` imports only `core.base`.
- **`Stepper.initialize`:** it builds the event locator, the renewal clock, and the delay history. These read
  compiled events and renewals.
- **`Backend.lower`:** it takes the compiler's kernels (`Kernel`, `Reference`, `ReductionPass`, `Draws`). A `Kernel`
  imports `core.laws` and the compiler's printing.
- **Formalism names in `base`:** `HilbertSpace` and `HilbertFamily` are used only by quantum, and `Structure` only by
  rewriting. `engine/compile/_validation.py` maps the domain classes to their kinds by `isinstance`.

## Goals / Non-Goals

**Goals:**
- `core.base` holds the base classes of the framework and the helpers they are declared with, and nothing else.
- Every base is something no other base is, and relates to the others only by having them.
- Every class of the framework derives from the base it is.

**Non-Goals:**
- Changing what a class does, beyond detaching `System` and `Interaction` from components.
- Fixing the classes outside `base` that derive wrongly, such as `Invariant(Law)` or `Wiener(Element)`. The change
  `fix-class-relations` fixes them.

## Decisions

- **The bases and their parents.** Inheritance means *is*, and membership means *has*:

  ```text
  BaseEstimator
  ├── BaseComponent      a part of a system
  ├── BaseInteraction    a relation between parts
  ├── BaseSystem         the whole
  ├── BaseSpace          where the parts live
  ├── BaseTime           how time advances
  ├── BaseSolver         advances a problem
  └── BaseBackend        writes numeric code
  BaseLaw                a rule of change
  BaseDomain             the values of a variable
  BaseProblem            a compiled system
  ```

  | Base | Has |
  |---|---|
  | `BaseSystem` | components, interactions, subsystems, one space, one time |
  | `BaseComponent` | a label, declared states and parameters, each with a `BaseDomain`, and laws |
  | `BaseInteraction` | a label, roles bound to components, declared parameters, and laws |
  | `BaseProblem` | the form, the layouts, and the initial vectors of a compiled system |
  | `BaseSolver` | the `BaseProblem` it advances |
  | `BaseBackend` | none of the others; it writes the numeric functions a problem holds |

  - `BaseEstimator` is the parent of the bases that are configured by constructor arguments, for its parameter
    handling and for fitting to data later.
  - Laws are frozen dataclasses of expressions, domains are values, and a problem is a compiled record, so these
    three derive from nothing.
  - *Alternative:* keep `System` and `Interaction` under one component base. Rejected because a system is the whole
    and an interaction is a relation, not a part.

- **Shared declarations are helpers, not a base.** `BaseComponent` and `BaseInteraction` both have declared variables
  and laws, and neither is the other. The declaration-based constructor, `get_params`, `get_default`,
  `get_argument_names`, `declarations`, and `define` become functions of `core/base/_variables.py`. Each base calls
  them in a one-line method.
  - *Alternative:* a common parent of the two. Rejected because nothing names what both *are*, only what both
    *have*.

- **A system has its subsystems.** `BaseSystem` takes `components=`, which may hold elements, groups, and systems,
  and `interactions=`. A nested system is held as a part; it does not become a component.

- **What `base` holds.** `core/base` holds one module per base and the helper modules the bases are declared with.
  Modules import downward in the order of the table, and `core.base` imports no other package of `skcomplex`.

  | Module | Holds |
  |---|---|
  | `_errors.py` | the error hierarchy |
  | `_registry.py` | `Registry` |
  | `_kinds.py` | `Term`, `LawKind`, `LawClass`, `TimeKind`, `Trait`, `SpaceKind`, `ComponentKind`, `DomainKind`, `Binding`, `Requires`, `Form`, `Supports`, `OrderKind`, `NoiseKind`, `NUMERIC_DOMAINS`, `vocabulary` |
  | `_domain.py` | `BaseDomain`, `Domain`, `Categorical`, `Tensor`, `wrap_angle` |
  | `_symbolic.py` | `symbol`, `vector`, `Expression`, `SymbolicVector`, and the symbol helpers |
  | `_variables.py` | `state`, `parameter`, `role`, `Variable`, `VariableKind`, and the declaration helpers |
  | `_visuals.py` | `VisualSpec` |
  | `_layout.py` | `Layout`, `Entry`, `Vector`, `Indices` |
  | `_law.py` | `BaseLaw`, `register_laws`, `law_classes`, `is_registered`, `has_trait` |
  | `_space.py` | `BaseSpace` |
  | `_time.py` | `BaseTime` |
  | `_component.py` | `BaseComponent` |
  | `_interaction.py` | `BaseInteraction` |
  | `_system.py` | `BaseSystem` |
  | `_problem.py` | `BaseProblem` |
  | `_solver.py` | `BaseSolver` |
  | `_backend.py` | `BaseBackend` |

- **`BaseProblem` is what every solver reads.** It is a frozen dataclass of `form`, `states`, `parameters`,
  `initial_state`, `initial_parameters`, and `time`. `engine.compile.Problem` derives from it and keeps every other
  field. The kinetic, quantum, and rewriting problems derive from `BaseProblem`, not from `Problem`, and keep only
  the fields they use.

- **`BaseSolver` is the interface; the run machinery stays in the engine.**
  - `BaseSolver[P: BaseProblem]` declares `form`, `forms`, `supports`, and the flags. It defines the capability
    check, and the abstract `initialize`, `advance`, `plan`, and `initial_state`.
  - `engine/solvers/_stepper.py` keeps `Solver(BaseSolver[Problem])`, with events, renewals, delays, and the solver
    registry. Every built-in solver derives from it, and the solvers of quantum, rewriting, and kinetics derive
    from `BaseSolver` of their own problem.
  - *Alternative:* move the event, renewal, and delay machinery into `base`. Rejected because it reads the compiler's
    event and renewal types.

- **`BaseBackend` declares the contract; the kernels stay in the compiler.** `BaseBackend` declares `name`,
  `supports`, and the abstract `lower`. Its argument and return types are named by protocols in
  `core/base/_backend.py`, `KernelSource` and `Lowering`. The compiler's `Kernel` and `Lowering` satisfy them. A
  backend therefore has no class relation to a solver.

- **An external library is reached by a solver.** `ExternalBackend` advances a problem by handing it to another
  library, so it is a solver. It becomes `ExternalSolver(BaseSolver)` in `engine/solvers`, and `LammpsBackend`
  becomes `LammpsSolver`.

- **A system pass is a pass.** `SystemPass` derives from `Pass`, and the registry of passes holds both.

- **A domain is a base.** `BaseDomain` declares `kind`, `contains`, `encode`, and `decode`. `AnyDomain` is removed,
  and validation reads `domain.kind`.

- **Errors stay in one hierarchy, except those of one formalism.** The shared errors stay in `_errors.py`. An error
  that only one formalism raises moves to that formalism.

- **Base names carry a `Base` prefix**, after `sklearn.base.BaseEstimator`, and no alias keeps an old name. A codemod
  in the scratchpad rewrites the references; it never enters the repository.

- **Tests follow their modules.** The tests of every moved module move to the mirror of its new path, under
  `tests/core/base` or the formalism that now owns it.

## Risks / Trade-offs

- [Detaching `System` and `Interaction` from `Component` breaks code that checks `isinstance(x, Component)`.] → Each
  check is rewritten against the base it means. Where a check means "anything a system holds", it names the bases in
  a tuple.
- [The rename touches about 1,250 references.] → One codemod renames every reference. Then `ruff`, mypy, the import
  contracts, and the tests of the touched packages check the result.
- [The `KernelSource` protocol may not cover everything a backend reads.] → The protocol lists the attributes that
  `NumpyBackend` and `NumbaBackend` read. A backend that reads more fails the type check.
