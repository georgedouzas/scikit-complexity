## Why

The library follows scikit-learn's conventions only half way. Every public object is an estimator with `prepare` and
`run`, including things that are not models: a simulation, an analysis, an experiment, a compiler. In scikit-learn only
what is defined is an estimator. What is done to an estimator is a function or a helper object: `cross_validate`
evaluates an estimator, `sklearn.metrics` scores its results, `model_selection` explores its parameters. Following that
split makes the library easier to learn for anyone who knows scikit-learn, and shrinks the surface of classes.

The library also lacks one place that says what a part can be combined with. Compatibility lives in solver supports,
role specs, the compiler's form inference, the inventory's system vocabulary, and a generated compatibility page that
lists every declaration of every class, which readers cannot use to decide anything. scikit-learn solves the same
problem with estimator tags, which functions and meta-estimators read to decide what is allowed.

## What Changes

- **Estimators stay for what is defined.** Elements, interactions, laws, and systems keep `prepare` and `run`, in the
  roles of `fit` and `transform`.
- **BREAKING: systems form a class hierarchy.** `BaseSystem` holds the verbs, derived tags, nesting, and passing model
  parameters down to parts. `System(BaseSystem)` is the generic composite of elements and interactions, in the role of
  `Pipeline`. General systems named after their mathematics, such as `MassActionSystem`, live in the core and enter it
  only when two fields share the structure. Every ready-made system becomes a class in its field, such as
  `SIRSystem(MassActionSystem)` or `LorenzSystem(BaseSystem)`, whose constructor stores the model's own parameters and
  whose `prepare` builds its parts, so `get_params`, `set_params`, and `clone` work on parameters such as `beta`. A
  variant subclasses a system of its own field only. No class is chosen by mathematical form, which the tags give.
- **BREAKING: ready-made systems are used directly.** Users import a system class from its subfield, such as
  `from skcomplex.systems.biology.epidemiology import SIRSystem`. `create_system`, `create_simulation`, `describe`,
  `register_system`, and the registry are removed. A class carries its field, topics, references, and tuned run
  settings as class attributes. `all_systems()` lists and filters the system classes, in the role of
  `sklearn.utils.all_estimators()`.
- **BREAKING: simulating becomes a function.** `simulate(system, *, until, initial_state, solver, backend,
  observables, random_state)` returns a `Trajectory`, in the role of `cross_validate`. Solvers take the role of
  splitters (`cv=`), backends the role of `n_jobs`, observables the role of `scoring=`. The `Simulation` class is
  removed. Arguments left out come from the tuned run settings of the system class, then from its tags.
- **BREAKING: analyses become functions of results.** Analyses of trajectories, arrays, and graphs become functions,
  in the role of `sklearn.metrics`, such as `lyapunov_exponents(trajectory, ...)`. The `*Analysis` classes are removed.
  Settings worth reusing are bound by a helper in the role of `make_scorer`.
- **BREAKING: experiments become model-selection functions.** Parameter sweeps, ensembles, scenarios, and sensitivity
  become functions in the role of `validation_curve` and `cross_validate` with repeats. Calibration stays an estimator,
  in the role of `GridSearchCV`, since it returns a fitted system.
- **BREAKING: compilers become private machinery.** Problem forms and compilers stay internal, as scikit-learn's
  fitting internals are. A system package that brings its own form registers it through one documented hook.
- **Capability tags.** Every part declares tags through a plain method `get_tags()` that returns a frozen `Tags`
  value: time, state domains, space, law kinds, randomness, structure, and flags such as differentiable. A `System`
  derives its tags from its parts. Interactions declare the tags they accept on each role, solvers and backends the tags
  they support, analyses and experiments the tags they need, so incompatible combinations fail before compiling with a
  message that names the tag. Topical catalog tags, such as `epidemic`, stay catalog metadata, apart from capability
  tags.
- **The API rule changes.** The rule "no public functions outside `skcomplex.systems`" becomes "estimators for what is
  defined, functions for what is done to them", as in scikit-learn.
- **Documentation.**
  - The README, the user guide, every docstring example, and every gallery example use the new API.
  - The compatibility page is replaced by pages per kind of part (elements, interactions, laws, solvers, backends,
    analyses, experiments), each generated from tags, with a table of what each class declares or accepts and worked
    examples of compatible and incompatible combinations.
  - A migration page lists every removed name and its replacement.
  - A systems page explains `BaseSystem`, the composite `System`, general systems, and field realizations.
  - The catalog pages show each system's class, import path, model parameters, and capability tags.

## Capabilities

### New Capabilities

- `capability-tags`: tags declared by parts, derived by systems, and read by simulation, solvers, backends, analyses,
  experiments, the conformance suite, and the catalog.

### Modified Capabilities

- `base-interfaces`: the base of parts gains `get_tags()`, estimators are only the parts and calibration, and systems
  form the `BaseSystem` hierarchy.
- `model-catalog`: ready-made systems are classes used directly and found with `all_systems()`.
- `simulation`: `simulate` replaces the `Simulation` class.
- `analysis-interface`: analyses are functions of results.
- `experiments`: experiments are functions, except calibration.
- `package-layout`: the API rule on public functions, and the packages of simulation, analysis, and experiments.
- `user-documentation`: the guide and examples follow the new API, and compatibility is documented per part from tags.

## Impact

Every caller of `Simulation`, `create_system`, `create_simulation`, `describe`, the `*Analysis` classes, the experiment
classes, and the public compilers breaks. Every factory function of `skcomplex.systems` becomes a class. The catalog
pages, the gallery, the README, the guide, and most tests change. The release after this change
is 0.7.0 with a migration page. The internal engine, the solvers, the backends, and every system definition keep their
behaviour.
