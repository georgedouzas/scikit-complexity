# Design

## Context

- **What the children define that their base does not.** An inventory of the public methods of every family, outside
  the bases:
  - **Spaces:**
    - `size` in 5 of 6 spaces;
    - the discrete neighbourhood methods (`index_table`, `compute_ball`, `list_neighbor_pairs`) in 4;
    - `dimensions`, `default_visual`, `positions`, `n_dim`, and `check` in 2 or 3 each, with `torus` written twice
      identically.
  - **Problems:** the runtime calls `observable_getters`, `adaptive_parameters`, `stochastic`, `lattice_shape`,
    `objectives`, `max_speed`, and others on any problem. Only the compiler's `Problem` defines them, so the runtime
    falls back on `getattr`. The quantum problem adds ten methods that `engine.analysis` calls by duck typing.
  - **Laws:** they name what they change as `target` in some laws and `targets` in others. Twenty laws validate in
    `__post_init__`, three of them with the same body.
  - **The other families:**
    - two visuals implement `log_agents`, and the `_colors` and `_layout` helpers are copied across visuals;
    - the optimal control solvers have `prepare` and `solve`, and the external solvers `check` and `build`, all
      called only by their own base;
    - the surrogates are frozen dataclasses built by `fit_dmd`, `fit_sindy`, and `fit_reservoir`, and two of three
      have `forecast`.
- **Magic methods:** about 90 hand-written ones in `core`, `engine`, and `formalisms`.

  | Method | Count |
  |---|---|
  | `__post_init__` | 41 |
  | `__repr__` | 31 |
  | `__len__` | 9 |
  | `__call__` | 6 |
  | `__getattr__` | 3 |
  | `__eq__`, `__hash__` | 3 each |
  | `__str__` | 3 |
  | `__reduce__`, `__copy__`, `__deepcopy__`, `__new__`, and `__setattr__` on terms | several |
  | arithmetic | quantum operators and designs |

## Goals / Non-Goals

**Goals:**
- A modeller who learns one member of a family knows the others: the same questions get the same names.
- Every public method of a family is written once, on its base.
- Behaviour reads as named methods, never as hidden magic.

**Non-Goals:**
- New features. Every method kept here exists today under some name.
- The quantity classes, which `fix-class-relations` reworks.

## Decisions

- **The template method, as scikit-learn writes it.** A base defines each public method or property once:
  - it checks the call;
  - it calls a private hook `_name` that a child overrides;
  - it returns the result in the family's shape.

  A hook that every child must provide raises `NotImplementedError` in the base. A hook that some children lack has
  a trivial default in the base. A child adds no public method, apart from:
  - constructors that build it, such as `NetworkSpace.erdos_renyi` and `MeshSpace.rectangle`;
  - the extension hooks a user implements to define a subclass, such as `laws`, `energy`, and `pair_energy`. These
    are public on the class that calls them, as `fit` is in scikit-learn.

- **The family APIs.** The modeller's questions decide the names. A property states a fact and a method computes:

  | Base | Public API, implemented on the base | Hooks a child overrides |
  |---|---|---|
  | `BaseSpace` | `n_dim`, `size`, `shape`, `torus`, `default_visual`, `positions()`, `neighbors(site, neighborhood=None)`, `check()` | `_n_dim`, `_size`, `_shape`, `_periodic`, `_positions`, `_neighbors`, `_check`, and the compiler's `_index_table`, `_ball`, `_neighbor_pairs` |
  | `BaseTime` | `time_kind`, `schedule` | none |
  | `BaseLaw` | `targets`, `check()` | `_targets`, `_check` |
  | `BaseDomain` | `value`, `domain_kind`, `sage_domain`, `contains()`, `encode()`, `decode()` | the same names with `_` |
  | `BaseProblem` | `elements`, `system`, `symbols`, `stochastic`, `objectives`, `adaptive_parameters`, `observable_getters()`, `lattice_shape()`, `max_speed()`, `state_symbols`, `parameter_symbols` | the compiler's `Problem` and the formalism problems |
  | `BaseComponent` | `default_visual`, and the existing declaration API | `laws` |
  | `BaseInteraction` | `energy(s)`, and the existing declaration API | `laws`, `energy` |
  | `BaseSolver` | the existing interface | `_prepare`, `_solve`, `_check`, `_build`, from the optimal control and external solvers |
  | `BaseVisual` | `views()`, `log()`, `log_agents()` | `_views`, `_log`, with `_colors` and `_layout` written once on the base |
  | `BaseObserver` | `on_start()`, `on_frame()`, `on_stop()`, `trajectory` | the same names with `_` |
  | `BaseDesign` | `rows()`, `product(other)`, `concat(other)` | `_rows` |
  | `BaseSurrogate` | `fit(series)`, `forecast(...)`, `system_`, `training_error_`, `validation_error_` | `_fit` |

  - Trivial defaults keep the same calls valid for every child:
    - `size` and `positions()` of a continuous space are `None` and an empty array;
    - `neighbors` of a mesh are the nodes sharing a cell;
    - `energy` of an interaction without a potential is `None`;
    - `log_agents` of a visual without agents logs nothing;
    - `trajectory` of an observer that records nothing is `None`.
  - `SiteSpace` is removed, since `BaseSpace` now defines the discrete neighbourhood hooks, and `NetworkSpace` is a
    space of sites like the lattices.
  - The methods only one child had become private, or move to their user. These include `CayleySpace.words`,
    `MeshSpace.skfem_mesh`, the quantum problem's `expect` and `state_of`, and `MesaSolver.create_agent`.
  - `engine/analysis/_quantum.py` moves to `formalisms/quantum`, so the engine stops calling quantum methods by duck
    typing.

- **Values and behaviour.**
  - A plain value record holds data and is compared by its fields. It stays a frozen dataclass and has no
    hand-written methods. These are terms, domains, laws, compiled entries and kernels, and problems.
  - A configurable object with behaviour derives from its base on `BaseEstimator`, takes its settings in an explicit
    `__init__` that stores them unmodified, and prints through scikit-learn's parameter printing.
  - `BaseLaw` stays a value base, as two laws with the same target and expression are the same law.

- **Validation leaves `__post_init__`.** The checks of laws and records move into the private `_check` hook behind a
  public `check()`. Validation calls it on every law of a system, so an error names the component that holds the
  value, not only the value. A record that a builder function creates is checked by that function.

- **Terms are value records.**
  - `Term(name, value)` becomes a frozen dataclass. The constants are class attributes made by
    `Term.register(name, value)`.
  - Pickling, copying, and comparison work by value with no hooks, so `__new__`, `__reduce__`, `__copy__`,
    `__deepcopy__`, and `__setattr__` go.
  - Every `is` comparison of a term becomes `==`.

- **The magic methods that stay.** The code defines no magic method except these:
  - `__init__`;
  - the arithmetic of the quantum operators, which the modeller writes as mathematics, `σx * σy + h σz`;
  - the collection protocol of collections the modeller indexes or iterates, `trajectory['a.n']`, `len(layout)`,
    `for entry in layout`;
  - `__getattr__` of the symbol namespace of laws, `s.x` and `s.target.n`, the notation laws are written in.

  The remaining magic methods change as follows:
  - **`__call__`:** each callable object gets a named method, such as `evaluate(t, x, p)`.
  - **`__eq__` and `__hash__`:** a class that defines them by hand either becomes a value record or keys its
    dictionaries by an explicit `key` property.
  - **`__repr__` and `__str__`:** printing comes from the parameters or the fields, and a readable string comes from
    a named property such as `text`.
  - **Design operators:** designs combine with `product` and `concat`. `GridDesign(...) * RowsDesign(...)` becomes
    `GridDesign(...).product(RowsDesign(...))`.

- **Surrogates are estimators.**
  - `BaseSurrogate(BaseEstimator)` takes its settings in `__init__` and trains with `fit(series)`.
  - `fit` stores `system_`, `labels_`, `states_`, `dt_`, `training_error_`, and `validation_error_`.
  - `forecast(steps)` simulates the fitted system from its last training state, so all three surrogates forecast.
  - The DMD and reservoir surrogates keep their own, faster `_forecast`.
  - `DMDSurrogate`, `SINDySurrogate`, and `ReservoirSurrogate` replace `fit_dmd`, `fit_sindy`, and
    `fit_reservoir`.

## Risks / Trade-offs

- [Laws and records lose their construction-time checks.] → Validation runs `check()` on every law of a system
  before compiling. `compile_system` validates too, so no unchecked law reaches a solver. The errors gain the
  owner's path.
- [Code compares terms with `is`.] → A codemod in the scratchpad rewrites every `is` and `is not` comparison of a
  term, and mypy and the tests check the result.
- [Moving quantum analysis out of the engine changes an import path.] → The functions keep their names, and the
  guide shows the new import.
- [Callers that kept `getattr` fallbacks drift back.] → The base APIs are typed, so mypy rejects a call of a method
  the base does not define.
