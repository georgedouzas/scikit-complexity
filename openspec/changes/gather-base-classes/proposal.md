# Proposal

## Why

`skcomplex.core.base` holds none of the base classes of the framework, and the bases relate to each other wrongly:

- `Interaction` and `System` derive from `Component`. But `System` uses only the label of `Component`, and takes its
  components and interactions as two separate lists.
- `ExternalBackend` derives from `Stepper`, so a class named for a backend is a solver.
- The quantum, kinetic, and rewriting problems derive from the compiler's 50-field `Problem`, whose fields they do not
  use.

Meanwhile `base` holds a mix of terms, declarations, and domains, some of which only one formalism uses.

## What Changes

- **BREAKING** `skcomplex.core.base` holds the base classes of the framework. No base derives from another; each
  relates to the others by having them.
  - `BaseComponent`, `BaseInteraction`, `BaseSystem`, `BaseSpace`, `BaseTime`, `BaseSolver`, and `BaseBackend`
    derive from `sklearn.base.BaseEstimator`, for its parameter handling and for fitting later.
  - `BaseLaw`, `BaseDomain`, and `BaseProblem` derive from nothing.

  | Base | Replaces | Is the base of |
  |---|---|---|
  | `BaseComponent` | `Component` | `Element`, `Group` |
  | `BaseInteraction` | `Interaction` as a base | `Interaction`, `Potential`, `PairPotential` |
  | `BaseSystem` | (new) | `System` |
  | `BaseSpace` | `Space` | the spaces |
  | `BaseTime` | `Time` | `ContinuousTime`, `DiscreteTime`, `StaticTime` |
  | `BaseLaw` | `Law` | the laws |
  | `BaseDomain` | (new) | the domains |
  | `BaseProblem` | (new) | the compiler's `Problem`, `KineticProblem`, `QuantumProblem`, `RewritingProblem` |
  | `BaseSolver` | `Stepper`, as an interface | the solvers |
  | `BaseBackend` | `Backend` | `NumpyBackend`, `NumbaBackend` |

- **BREAKING** `System` and `Interaction` no longer derive from a component. A system has components, interactions,
  subsystems, a space, and a time. An interaction has roles bound to components.
- `BaseComponent` and `BaseInteraction` both have declared variables and laws. Their shared code (the
  declaration-based constructor, `get_params`, and `define`) becomes helper functions of `core/base/_variables.py`
  that both call.
- `BaseProblem` is the part of a compiled system that every solver reads:
  - the form;
  - the state and parameter layouts;
  - the initial vectors;
  - the time symbol.

  `Layout` moves to `base` with it.
- `BaseSolver` is the solver interface over a `BaseProblem`. `engine.solvers` keeps `Solver(BaseSolver)`, which runs
  events, renewals, and delays, and every built-in solver derives from it.
- **BREAKING** `ExternalBackend` becomes `ExternalSolver` in `engine/solvers`, and `LammpsBackend` becomes
  `LammpsSolver`.
- `SystemPass` derives from `Pass`.
- `base` keeps the helpers these classes are declared with:
  - the errors;
  - `Registry`;
  - the kinds;
  - `Form` and `Supports`;
  - `state`, `parameter`, and `role`;
  - `symbol` and `vector`;
  - the numeric, categorical, and tensor domains.
- **BREAKING** The types and errors of one formalism leave `base` for that formalism:
  - `HilbertSpace`, `HilbertFamily`, and `HilbertSpaceError` go to `quantum`;
  - `Structure`, `RewritingError`, and `WordProblemError` go to `rewriting`;
  - `HorizonError` goes to `control`.
- Every domain names its kind, so validation reads it instead of listing the domain classes.
- The constitution names the base classes, their parents, and what `base` holds.
- The classes outside `base` that derive wrongly are fixed in the follow-up change `fix-class-relations`.

## Capabilities

### New Capabilities

### Modified Capabilities
- `package-layout`: adds the base classes, their relations, and what `core.base` holds.
- `system-modeling`: a system holds nested systems without being a component.

## Impact

- **Code:**
  - every module that imports `Component`, `Interaction`, `Law`, `Space`, `Time`, `Stepper`, `Backend`, `Problem`,
    `Layout`, or `ExternalBackend`, about 1,250 references across `src`, `tests`, and `docs`;
  - the modules that import the moved formalism names;
  - `engine/compile/_validation.py`, which today lists the domain classes.
- **Tests:** the tests of the moved modules move with them into `tests/core/base`, and into the formalisms.
- **Constitution:** the project profile names the base classes, which bumps it to 1.9.0.
- **Other changes:** the active changes that name the old classes need their text updated when they are applied.
