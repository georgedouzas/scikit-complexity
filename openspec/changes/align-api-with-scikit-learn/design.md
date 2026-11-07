## Context

0.6.0 makes every public object an estimator with `prepare` and `run`: parts, `Simulation`, about fifty analysis
classes, the experiment classes, and the compilers. Compatibility is spread over role specs, solver supports, the
compiler's form inference, the inventory's system vocabulary, and a generated compatibility page that lists every
declaration. scikit-learn separates estimators from the functions that evaluate them, and tells functions what an
estimator can do through tags.

## Goals / Non-Goals

**Goals:**
- One mental model shared with scikit-learn: estimators for what is defined, functions for what is done to them.
- One source of truth for compatibility, the tags, read by every function and by the documentation.
- Documentation and examples that use the new API, and compatibility pages that help a reader decide.

**Non-Goals:**
- Changing the numerics of any solver, backend, law, or system.
- Changing what a system defines, beyond the tags its parts declare.

## Correspondence with scikit-learn

| scikit-learn | scikit-complexity |
|---|---|
| `BaseEstimator`, `get_params`, `set_params`, `clone` | the base of every part |
| `fit` | `prepare` |
| `transform`, `predict` | `run` of a part |
| `NotFittedError` | the error for running an unprepared part |
| `Pipeline`, `ColumnTransformer` | `System` |
| `_parameter_constraints`, `Interval`, `StrOptions` | `ParameterSpec`, `StateSpec`, domains |
| estimator tags, `__sklearn_tags__` | capability tags, `get_tags()` |
| `check_estimator` | the conformance suite, selected by tags |
| `LinearModel`, `Ridge`, `Lasso` | general systems of the core, field realizations such as `SIRSystem` |
| `sklearn.utils.all_estimators()` | `all_systems()` |
| `cross_validate` | `simulate` |
| `cv=` splitters | `solver=` |
| `n_jobs`, joblib backends | `backend=` |
| `scoring=`, `make_scorer` | `observables=`, and the analysis binding helper |
| `sklearn.metrics` | analysis functions |
| `validation_curve`, `learning_curve`, repeated CV | sweeps, ensembles, scenarios, sensitivity |
| `GridSearchCV` | calibration, which stays an estimator |
| `*Display.from_estimator` | visual observers, such as Rerun |
| fitting internals | compilers and problem forms, private |

## Decisions

- **Tags through a plain method.** scikit-learn uses `__sklearn_tags__`. The project forbids magic methods, so parts
  define `get_tags()`, overriding a private `_get_tags()` hook per class the way specs are declared today, and return
  a frozen `Tags` dataclass from `skcomplex.core.specs`.
- **Tags come from existing declarations.** The fields map to what exists: state specs give domains, spaces give
  space, law classes give law kinds, solver supports become supported tags. The inventory's `system` vocabulary is
  generated from the tags of each system instead of written by hand, and the inventory check compares them.
- **System tags are derived, never declared**, so they cannot drift from what the compiler does. The compiler's form
  inference reads the tags.
- **Analyses as functions.** Each analysis class becomes a function named after it in snake case, with the same
  settings as keyword arguments and a frozen result value with the same fields as today's fitted attributes. The base
  `BaseAnalysis` and families with a choice argument become one function with that argument. The binding helper
  `bind_analysis(function, **settings)` returns a callable that experiments and `simulate` accept.
- **Experiments as functions.** `sweep`, `ensemble`, `compare_scenarios`, `sensitivity`, and `basin_stability`
  return tidy tables. `Calibration` stays an estimator.
- **Package layout, which needs the user's approval before any move.** Proposed: `simulate` and `Trajectory` in
  `skcomplex.core.engine` (no move), analysis functions in `skcomplex.core.analysis` (no move), experiment functions
  in `skcomplex.core.experiments` (no move), compilers and problems made private inside `skcomplex.core.engine` (no
  move, only names lose their export). No package is created or moved by this design. Any later move is asked first.
- **Documentation.** The guide's parts keep their task order, rewritten to the new API. The compatibility page and its
  generator are replaced by one generated page per kind of part, under a "Parts" section of the navigation, with
  executed examples of a compatible and an incompatible combination per page. A migration page maps removed names to
  their replacements.
  `docs/generate_parts.py` writes the pages of elements, interactions, laws, solvers, and backends, one private
  function per page. The analyses, experiments, and systems pages come in later waves, once their functions and
  classes declare tags, and the generator should add a function for each of them then.
- **Release.** One breaking release, 0.7.0. No aliases for removed names, as in earlier breaking releases.

- **System classes.** `BaseSystem` holds the verbs, the derived tags, nesting, and a private method that passes model
  parameters down to the parts in `prepare`. Each factory function becomes a class in the same self-contained module,
  named after the system in CamelCase with the suffix `System`. Its class attributes give `field`, `topics`,
  `references`, and the tuned run settings (`until`, `dt`, solver hint, observables, initial state). Model parameters
  become constructor arguments with the factory's defaults.
- **Discovery.** `all_systems()` reads field and topics statically from the class definitions, as the registration
  reader does today, so listing by field or topics imports no system module. The registry, `register_system`,
  `create_system`, `create_simulation`, and `describe` are removed. The conformance suite, the documentation generator,
  and the inventory and gallery checks use `all_systems()`.
- **General systems.** The cross-field duplicate scan of the consolidation lists shared structures. A general system
  enters the core in a later task only when two fields share the structure, starting with mass-action compartments.

## Risks / Trade-offs

- **Size:** about fifty analyses, the experiments, the catalog, every gallery example, and most tests change. The work
  runs in waves by package, with the documentation last.
- **Tags too fine or too coarse:** tags must stay few and orthogonal. A new tag needs a function that reads it.
- **Functions with many arguments:** `simulate` takes several keyword arguments. That is the scikit-learn trade, and
  the tuned settings of a catalog entry keep the common case short.
