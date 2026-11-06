# Proposal

## Why

0.5.0 is on hold until a discussion of the documentation and the API decides what it carries. This change collects what
comes up in that discussion, so that nothing is lost and each item ends with a decision. Decided items have specs, a
design, and tasks. Open items stay listed until they are decided.

## What Changes

Every item is decided.

### Documentation decisions

- Say "in any field of science", not "from any science", in the README, the guide, and the package docstring.
- Use no semicolons in prose.
- Set `relaxed_headers` on superfences, so that `python continuation` blocks render on the documentation site.
- Keep the README as code only, with no output and no `print`, on GitHub and on the documentation site.
- Run the Python blocks of the guide when the documentation builds, and show the value of each top-level expression below
  them, with no `print` in the source. A failing block stops the build.
- Remove hard-coded counts of models from prose.

### API decisions

- Rename `create` of `skcomplex.systems` to `create_system`.
- Replace `run` of `skcomplex.systems` with `create_simulation(name, **parameters)`, which returns an unprepared
  `Simulation` that holds the model as `system` and its tuned `until`, `dt`, solver, observables, initial state and seed
  `random_state=0`. It runs with the two verbs, `create_simulation('sir').prepare().run()`, and its settings change with
  `set_params`. `create_system` stays for changing or reusing the model itself. `Simulation` gains an `until` parameter
  that `run()` uses when it is given none.
- Restructure the packages into two. `skcomplex.core` holds everything general: `base`, `errors`, `types`, `specs`,
  `components` (the former `skcomplex.core`: elements, interactions, laws, spaces, `System`, and their own base, types
  and specs), `engine`, `analysis` and `experiments`. `skcomplex.systems` is the former `skcomplex.models`, the
  catalog of ready-made systems. The tests mirror it as `tests/core/...` and `tests/systems/...`. No aliases for the
  old paths.
- Rename "model" to "system" everywhere, in identifiers such as `register_model`, `ModelEntry`, `ModelCatalog`,
  `ModelListing` and `MODELS`, in the docs section and pages, and in all prose, including scientific names. Library
  identifiers of other packages, such as `mesa.Model`, and the titles of cited references keep their wording.
- Rewrite the user guide task-first, from the modeler's view and in the style of the README, keeping every current
  topic as a section.

### Decisions on the former open items

- Replace the arithmetic magic methods of `OperatorSpec` with the named methods `add`, `subtract`, `multiply`,
  `divide` and `negate`. A scalar operand stands for that multiple of the identity, as before.
- Keep `EarlyWarningAnalysis` and `CracklingNoiseAnalysis` in `skcomplex.systems`, since each serves the systems of one
  subfield. `skcomplex.core.analysis` holds the general analyses only.
- Keep `SubshiftSpec` as a declaration that a run does not enforce. A run checks that every cell holds a value of the
  alphabet, and `AutomatonAnalysis` decides whether a configuration is admissible, as its docstring already states.
  Enforcing forbidden patterns at every step would cost a scan of every neighbourhood per step for no current system.
- Remove `compile_system`, since the API has no public functions outside `skcomplex.systems`. Callers use a compiler,
  `PassCompiler(backend=..., solver=..., observables=...).prepare(system).run(initial_state).problem_`.
- Start the guide with the easiest custom system, one element and one interaction, and introduce laws, specs and
  annotations after it.
- Keep the gallery scope. The published documentation builds a sample, and a full build runs on demand.
- Fix the five failures of the full gallery: `plot_carbon_13_mfa` and `plot_urban_scaling_interactions` show a blank
  view, `plot_dynamic_multispecies_fba` and `plot_quarter_car` exceed their budgets, and `plot_free_group_automaton`
  raises a `GAPError` about a missing directory.
- Split the user guide into one page per part, with the long part on custom systems split by section.
- Fix the popularity-similarity tests in `tests/systems/mathematics/networks/test_growth.py`, which fail on some
  full-suite runs on Linux.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `package-layout`: two subpackages, `skcomplex.core` and `skcomplex.systems`, "system" for every ready-made item,
  and an engine surface with no public function.
- `model-catalog`: `create_system` and `create_simulation` replace `create` and `run`.
- `simulation`: a simulation holds a default end time, and a compiler is passed to it, with no `compile_system`.
- `system-validation`: a compiler validates in `prepare`, with no `compile_system`.
- `user-documentation`: a task-led guide, a README of code only, executed guide blocks, and plain prose.

## Impact

Every import path changes, with no alias, and `create`, `run`, `register_model`, `ModelEntry`, `ModelCatalog`,
`ModelListing` and `MODELS` are renamed or removed. 0.5.0 already breaks callers, since it removes `check()`. The
capability directories keep their names, such as `model-catalog`, since OpenSpec does not rename them.
