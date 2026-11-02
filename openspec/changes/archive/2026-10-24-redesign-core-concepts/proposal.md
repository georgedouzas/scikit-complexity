# Proposal

## Why

The core grew through many parallel changes merged by union, and its packages no longer follow one concept model.
- **Component hierarchy:** `Component` is the base of `Element` and `Interaction`, while `Group` and `System` derive from
  `BaseEstimator` directly.
- **No law base class:** each law is a standalone dataclass, and the compiler and validation recognise law kinds
  through hand-kept lists.
- **Formalisms scattered:** a formalism such as rewriting spreads over `laws`, `components`, `interactions`, `compile`,
  `solvers`, and `viz`.
- **Oversized compiler:** `compile/_compiler.py` holds 4 348 lines that mix the shared pipeline with the lowering of
  every formalism.
- **Disordered modules:** private helpers of other concepts sit between classes.
- **Magic in fundamental classes:** `Component.__init_subclass__` turns subclasses into dataclasses and injects
  companion fields.

The four reviews of the core also found defects that grew from this structure:
- validation and compilation plan partitions with two different rules;
- the NumPy and Numba steppers count steps differently, so one seed gives different paths;
- the Numba backend falls back to Python silently;
- validation swallows invalid lattices.

The 518 registered models now show which concepts the core must support, so the core can be redesigned around those
concepts before more changes build on the current layout.

## What Changes

- **Formalisms as plug-ins:** add `skcomplex.core.formalisms`, which holds one subpackage per domain-agnostic
  formalism:
  - `agents`, `lattices`, `groups`, `rewriting`, `quantum`, `fields`, `kinetics`, `fractional`, `topology`, `particles`,
    and `control`.
  - Each formalism registers its laws, components, lowering passes, forms, solvers, backends, visuals, and validation
    checks through one declaration.
  - The rest of the core never imports a formalism.
  - **BREAKING:** the import paths of formalism-specific names change. For example, `LocalRule` moves from
    `skcomplex.core.laws` to `skcomplex.core.formalisms.lattices`.
- **One component hierarchy:** `Element`, `Group`, `Interaction`, and `System` all derive from `Component`, which
  depends only on `base`.
  - Components take their declared values through one explicit constructor.
  - A component class lists its declared variables, and order and lag companions derive from that list.
  - **BREAKING:** `Component` no longer turns subclasses into dataclasses, and no longer injects companion fields as
    class attributes.
- **One law hierarchy:** every law derives from `Law` and declares its kind and its traits as class variables. The
  compiler and validation build their tables from the registered law classes instead of hand-kept lists.
- **Compiler as a pipeline:** `compile` becomes an explicit pipeline of passes with declared inputs and outputs.
  - The system is flattened once per compile.
  - Validation and compilation share one partition planner.
  - The lowering of each formalism moves to that formalism.
- **Solvers by form:** solvers split into modules by form. The NumPy and Numba backends share one step count.
- **Explicit diagnostics:** a backend that cannot serve a form, a Numba kernel that falls back to Python, and an
  invalid lattice are reported, never handled silently.
- **BREAKING:** public names spell neighbour one way: `neighbors`, `Neighborhood`, and the binding `over='neighbors'`.
  `neighbor_sum` folds into `neighbors(..., reduce='sum')`, and the neighbourhood keyword of `LocalRule` and `Energy` is
  `over=`.
- **Module order:** every core module follows the constitution's order. Private helpers come first, and a helper that
  serves another concept moves to the module of that concept.
- **Models and docs:** all models, gallery examples, tests, and documentation move to the new import paths.

## Capabilities

### New Capabilities

- `formalism-plugins`: how a formalism declares and registers its laws, components, passes, forms, solvers, backends,
  visuals, and checks, and how the core resolves them without importing any formalism.

### Modified Capabilities

- `package-layout`:
  - the core's package list, which now includes `formalisms` and drops the stale `model` package;
  - module-level rules: one concept per module, and base classes that import only `base`.
- `system-modeling`:
  - one component hierarchy, with systems and groups as components;
  - one law hierarchy;
  - group arguments checked when the group is created;
  - formalism primitives imported from their formalism.
- `model-compilation`:
  - one partition planner shared with validation;
  - one flattening per compile;
  - reported backend substitution and Numba fallback.
- `simulation`: the same seed gives the same path on every backend.
- `system-validation`: invalid spaces are reported as issues.

## Impact

- **Code:**
  - every package of `skcomplex.core`;
  - all 518 model factories and their gallery examples, whose imports are rewritten;
  - the tests, which move to mirror the new tree;
  - the user guide and the generated compatibility page.
- **Public API:**
  - import paths of formalism-specific names;
  - the base-class mechanics of components;
  - the exports of compiler internals, which leave `laws`, `components`, and `base`.

  Model behaviour and numeric results do not change, except where a defect is fixed:
  - the NumPy and Numba step counts agree;
  - an invalid lattice is reported by validation instead of failing later.
- **Constitution:** the Project Profile's layout, ecosystem contract, and extension points are amended.
- **Other changes:** the 15 active changes were planned against the current layout, and their designs need their paths
  updated after this change lands. They are:
  - `extend-agent-spaces`, `adopt-pyclaw-finite-volumes`, `add-variational-mechanics`, `add-tensor-fields`,
    `add-static-models`, `add-phase-space-fields`, `add-nonlocal-operators`, and `add-multiscale-networks`;
  - `add-molecular-structures`, `add-link-fields`, `add-learning-dynamics`, `add-higher-order-interactions`,
    `add-constraint-based-models`, `add-acausal-components`, and `add-unblocked-models`.
- **Dependencies:** none are added or removed.
