# Proposal

## Why

0.5.0 is on hold until a discussion of the documentation and the API decides what it carries. This change collects what
comes up in that discussion, so that nothing is lost and each item ends with a decision. Decided items have specs, a
design, and tasks. Open items stay listed until they are decided.

## What Changes

The decisions come first. The numbered items after them are undecided, and a suggestion there is only a starting point.

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

### API

1. **`OperatorSpec` magic methods.** It has eight magic methods for quantum operator arithmetic, the one exception to the
   rule against magic methods. Suggestion: replace them with named methods.
2. **Analyses outside `skcomplex.core.analysis`.** `EarlyWarningAnalysis` and `CracklingNoiseAnalysis` live elsewhere.
   Suggestion: move them in. A package move needs the user's approval.
3. **`SubshiftSpec` in a run.** It declares what is admissible, but a run does not enforce it, so a run can leave the
   allowed set without an error. To decide: enforce it, or state that only the analysis reads it.
4. **`compile_system` is a function.** `skcomplex.core.engine.compile` exports `compile_system`, a public function
   rather than a class with `prepare` and `run`, and the guide uses it. To decide: replace it with the compiler classes
   the guide already shows, or keep it as the one shortcut.
5. **The first system.** To decide: whether the simple case still takes too many lines once `check` is gone.

### Documentation

6. **Gallery scope.** The published documentation builds a sample of 24 examples, while the catalog lists every system. To
   decide: whether a release builds all of them.
7. **Failures of the whole gallery.** The first full build ran 576 examples, and 5 failed: `plot_carbon_13_mfa` and
   `plot_urban_scaling_interactions` have a blank view, `plot_dynamic_multispecies_fba` and `plot_quarter_car` exceed
   their budgets, and `plot_free_group_automaton` raises a `GAPError` about a missing directory.
8. **User guide size.** The guide, now task-led, is over 9,000 lines on one page. To decide: whether to split it.

### Reliability

9. **Seeded tests.** The popularity-similarity tests in `tests/systems/mathematics/networks/test_growth.py` fail on some
   full-suite runs on Linux, with no cause found. The same source failed at `1f22bf9b` and passed at `1000e3b6`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `package-layout`: two subpackages, `skcomplex.core` and `skcomplex.systems`, and "system" for every ready-made item.
- `model-catalog`: `create_system` and `create_simulation` replace `create` and `run`.
- `simulation`: a simulation holds a default end time.
- `user-documentation`: a task-led guide, a README of code only, executed guide blocks, and plain prose.

## Impact

Every import path changes, with no alias, and `create`, `run`, `register_model`, `ModelEntry`, `ModelCatalog`,
`ModelListing` and `MODELS` are renamed or removed. 0.5.0 already breaks callers, since it removes `check()`. The
capability directories keep their names, such as `model-catalog`, since OpenSpec does not rename them.
