# Design

## Context

- **Verbs today.** Every family names its life cycle its own way:

  | Family | Prepare | Act | Finish |
  |---|---|---|---|
  | solvers | `initialize`, `plan`, `initial_state` | `advance` | none |
  | backends | none | `lower` | none |
  | visuals | `views` | `log`, `log_agents` | none |
  | observers | `on_start` | `on_frame` | `on_stop` |
  | clocks | `start` | `wait` | none |
  | compilers | `claims` | `compile` | none |
  | designs | none | `rows` | none |
  | surrogates | `fit_dmd`, `fit_sindy`, `fit_reservoir` | `forecast` | none |
  | passes | none | `run` | none |

- **What the children define that their base does not:**
  - spaces define `size` in 5 of 6, the discrete neighbourhood methods in 4, and `positions`, `n_dim`, `check`, and
    `default_visual` in 2 or 3;
  - the runtime calls twenty problem methods that only the compiler's `Problem` defines;
  - laws name their changed states `target` or `targets`;
  - the copied helpers of visuals, `_colors` and `_layout`.
- **Constructors that work:**
  - `BaseComponent.__init__` rejects unknown and missing values;
  - `BaseInteraction.__init__` checks its layer;
  - spaces build their geometry in `__init__`;
  - 41 `__post_init__` validate records and laws.
- **Magic methods:** about 90 hand-written ones in `core`, `engine`, and `formalisms`.

## Goals / Non-Goals

**Goals:**
- A modeller learns three verbs once and uses them on every configurable object, as with `fit` and `predict` in
  scikit-learn.
- What a user gives and what the library computes stay apart: arguments as given, results ending in `_`.
- Behaviour reads as named methods, never as hidden magic.

**Non-Goals:**
- New features. Every verb and attribute exists today under some name.
- The quantity classes, which `fix-class-relations` reworks.

## Decisions

- **Three verbs, written once on each base.** For every base on `BaseEstimator`:
  - `check()` calls `_check()`, then returns `self`. It raises a named error for an argument the object cannot
    take.
  - `fit(*context)` calls `check()` and then `_fit(*context)`, then returns `self`. Calling it again refits from the
    current arguments.
  - `run(*inputs, final=False)` calls `_run(*inputs)`, and `_release()` when `final` is true, then returns the
    hook's result.

  The base provides trivial hooks:
  - `_check` accepts;
  - `_fit` derives nothing;
  - `_run` returns nothing;
  - `_release` releases nothing.

  So every object answers every verb, and a child overrides only the hooks its family needs. Checking, fitting, and
  running in the order the family requires stays inside the base.
  - *Alternative:* a fourth verb, `close`. Rejected by the maintainer in favour of `run(..., final=True)`.

- **What each family fits and runs.**

  | Base | `fit(context)` derives | `run(input)` does |
  |---|---|---|
  | `BaseComponent`, `BaseInteraction` | `declarations_` | `run(s)` returns the laws, from the user's `laws` hook |
  | `BaseSpace` | `size_`, `n_dim_`, `shape_`, `positions_`, `periodic_`, `default_visual_`, and the neighbour tables | `run(site, neighborhood=None)` returns the neighbours of a site |
  | `BaseTime` | `time_kind_`, `schedule_` | `run(until)` returns the frame times up to a time |
  | `BaseSolver` | `fit(problem, rng, until)`: `problem_`, `rng_`, `initial_state_` | `run(t, x, p, until)` returns the state at `until` and sets `events_`, `stopped_at_`, `snapshot_`, `agents_` |
  | `BaseBackend` | `fit(form, size)` | `run(kernels)` returns the numeric functions |
  | `BaseSystemCompiler` | `fit(system)`: `claims_` | `run(system)` returns the problem |
  | `BasePass` | none | `run(compilation)` writes the fields it declares |
  | `BaseVisual` | `fit(problem)`: `views_` | `run(frame)` logs the frame |
  | `BaseObserver` | `fit(problem)` | `run(frame)` reacts to a frame, and `run(frame, final=True)` ends the run |
  | `BaseClock` | `fit()`: `started_` | `run(time)` waits until a frame's time |
  | `BaseDesign` | `fit()`: `rows_` | `run()` returns the rows |
  | `BaseSurrogate` | `fit(series)`: `system_`, `labels_`, `dt_`, `training_error_`, `validation_error_` | `run(steps)` forecasts, simulating the fitted system unless the surrogate has a faster `_run` |

  - The runtime calls `fit` before the first frame, `run` at every frame, and `run(..., final=True)` at the last.
  - `run(final=True)` releases the Rerun viewer, the LAMMPS process, and the agent model.
  - Child-only methods become private or move to their user. These include `CayleySpace.words`,
    `MeshSpace.skfem_mesh`, `MesaSolver.create_agent`, and the quantum problem's `expect`.
  - `engine/analysis/_quantum.py` moves to `formalisms/quantum`, so the engine calls no method of a formalism's
    problem.
  - `SiteSpace` is removed, since `BaseSpace` defines the neighbourhood hooks for every space.
  - Extension hooks a user writes to define a subclass stay public on the class that calls them, as `fit` is public
    in scikit-learn. These are `laws`, `energy`, and `pair_energy`.

- **Trivial constructors and fitted attributes.**
  - `__init__` assigns each argument to the attribute of the same name, `self.x = x`, and nothing else. Components
    and interactions keep their single declaration-based constructor, which stores the label and every declared value
    the same way.
  - Every attribute that is not an argument ends in `_` and is set only by `fit` or `run`.
  - The checks of today's constructors move to `_check`. Validation calls `check()` on every part of a system before
    compiling, so an invalid part is reported with its path, and `compile_system` validates too.

- **Values: fields and `check()`.**
  - Laws, domains, problems, terms, compiled entries, and kernels are values. They are compared by content and stay
    frozen dataclasses without hand-written methods.
  - A value base defines the fields every child has, and the single method `check()`, which calls `_check()`:
    - `BaseLaw` defines `targets`, the states a law changes, which replaces the `target` and `targets` of its
      children;
    - `BaseDomain` defines `value`, `domain_kind`, and `sage_domain`, and keeps `contains`, `encode`, and `decode`,
      the conversions of stored numbers;
    - `BaseProblem` defines `elements`, `system`, `symbols`, `stochastic`, `objectives`, `adaptive_parameters`, and
      the readers the runtime calls, with trivial defaults.
  - `__post_init__` goes. Validation calls `check()` on every law of a system, and a builder function checks the
    records it builds.

- **Terms are strings of their value.**
  - `Term` derives from `str` and holds its value.
  - `Term.register(name, value)` registers a term and keeps its name, which the property `name` reads back.
  - A term pickles, copies, hashes, compares, and prints as its value with no hand-written method, so messages that
    format a term need no change.
  - Every `is` comparison of a term becomes `==`.
  - A term equals a plain string of its value. Vocabularies are never mixed in one collection, so this is harmless.
  - *Alternative:* a frozen dataclass of name and value. Rejected because it prints as its fields, so every message
    that formats a term would change.

- **The magic methods that stay.** No class defines a magic method except `__init__` and the arithmetic of the
  quantum operators, which the modeller writes as mathematics, `σx * σy + h σz`.
  - **Collections:** they are read by name.
    - A trajectory reads a variable with `values(name)`.
    - A layout gives `size` and `entries`.
    - A neighbourhood gives `offsets`.
    - Sections and snapshots give `frames`.
  - **The symbol namespace of laws:** it stores every symbol as a plain attribute in `__init__`, so `s.x` and
    `s.target.n` read attributes with no `__getattr__`, and an unknown name raises Python's own `AttributeError`,
    which names close matches.
  - **Probing a component's laws before compiling:** it builds the namespace from the component's declared names and
    roles, with `other` for the neighbour of a member, instead of inventing a symbol for any name.

  The rest become named methods, as follows:
  - **`__call__`:** a callable becomes a `run` method on a configurable object, or `evaluate` on a value.
  - **`__repr__`:** printing comes from scikit-learn's parameters or the dataclass fields.
  - **`__str__`:** a named property, such as `text`.
  - **`__eq__` and `__hash__`:** the class becomes a value record, or its dictionaries are keyed by an explicit `key`.
  - **Design arithmetic:** designs combine with `product` and `concat` instead of `*` and `+`.

## Risks / Trade-offs

- [Every family's verbs change at once.] → A codemod in the scratchpad renames the calls by their resolved class.
  The tests, the doctests, the user guide, and the gallery check that every caller moved.
- [Errors move from construction to validation.] → Validation runs `check()` on every part of a system before
  compiling, so no unchecked part reaches a solver, and the errors gain the path of the part.
- [Fitted attributes replace methods with arguments, such as `neighbors(site)`.] → What takes an argument stays a
  `run` input, and only facts of the whole object become fitted attributes.
- [`trajectory['a.n']` is the most-used notation in tests, models, and the guide.] → The codemod rewrites every
  subscript of a trajectory to `values`, and the doctests and markdown tests check the documents.
- [Code compares terms with `is`.] → The codemod rewrites every `is` comparison of a term, and mypy and the tests
  check the result.
