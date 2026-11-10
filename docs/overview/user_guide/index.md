# User guide

This guide follows the work of a modeler in order. It starts from a ready-made system, changes it, builds a system of
your own, and then simulates, analyses, evaluates, calibrates, and shares it. Each page builds on the ones before it.
The first run takes three lines.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import SIRSystem

simulate(SIRSystem()).get_values('I.x').max().round()
```

The epidemic of `SIRSystem` peaks at 304 infected people out of 1000.

## If you know scikit-learn

This section maps the names of scikit-learn to those of the library. What you define is an estimator, and what you do
to it is a function.

| scikit-learn | scikit-complexity |
| --- | --- |
| an estimator, such as `Ridge`, with `get_params`, `set_params`, and `clone` | a part of a system, an element, an interaction, a law, or a system, such as `SIRSystem` |
| `Pipeline` | `CompositeSystem`, the composite of elements, interactions, and nested systems |
| `fit` | `prepare` |
| estimator tags | capability tags, from `get_tags()` |
| `all_estimators()` | `all_systems()` |
| `cross_validate(model, X, y, cv=...)` | `simulate(system, solver=...)`, which returns a `Trajectory` |
| `n_jobs` | `backend=` |
| `sklearn.metrics` | the analysis functions, such as `lyapunov` and `entropy` |
| `validation_curve` and repeated cross-validation | the experiment functions, such as `sweep` and `ensemble` |
| `GridSearchCV` | `calibrate`, which returns the fitted system in its result |

A [system](glossary.md#system) is a class whose constructor takes the parameters of its model and stores them.
`set_params` changes them, and `clone` copies them. `simulate`, the analyses, and the experiments read the system and
return frozen results. They never change it.

## Two ways in

You can start from the catalog or from the core.

- **Select a ready-made system.** Import a system class from its subfield, such as `SIRSystem` from
  `skcomplex.systems.biology.epidemiology`, build it with the parameters you choose, and run it with `simulate`.
  `all_systems` lists the classes of the catalog by field, topic, or tag.
- **Build a custom system.** Write elements and interactions with `skcomplex.core.components` and the specs of
  `skcomplex.core.specs`, run them with `skcomplex.core.engine`, and study the results with `skcomplex.core.analysis`.

Every ready-made system is written with the same classes as a custom one, so the page of a system under
[Systems](../../systems/index.md) also serves as a pattern.

## The order of the guide

Read the pages in order. Each line says what you can do after reading the page.

1. [Select a ready-made system](select_a_ready_made_system.md). Find a system of the catalog, read what it models, and
   run it with its tuned settings.
2. [Change it](change_it.md). Give a ready-made system your own parameters and run settings, and nest it in a larger
   system.
3. [System classes](system_classes.md). Change what a ready-made system builds by subclassing it, and write a system
   class of your own.
4. [Learn the core components](core_components/index.md). Know how elements, interactions, laws, spaces, and systems
   fit together. The reference pages on [elements](core_components/elements.md),
   [interactions](core_components/interactions.md), [laws](core_components/laws.md),
   [spaces](core_components/spaces.md), and [systems](core_components/systems.md) explain each class, the page on
   [general realizations](core_components/general.md) explains the recipes that several fields share, and
   [Choosing parts](core_components/choosing_parts.md) leads from the features of a problem to its parts.
5. [Build a custom system](build_a_custom_system/index.md). Write a model of hares and lynx from core parts, as a
   subclass of a shared recipe, and as a system class.
6. [Check a system](check.md). Read the tags and requirements of parts, and fix the validation errors a builder meets
   most.
7. [Simulate](simulate.md). Choose the run settings, the solver, and the backend, and read the trajectory a run
   returns.
8. [Analyse](analyse.md). Find the analyses that apply to a system and read their results.
9. [Evaluate](evaluate.md). Compare a model with data and judge the fit and the identifiability of its parameters.
10. [Calibrate](calibrate.md). Fit the parameters of a system to data with each method of `calibrate`.
11. [Run experiments](run_experiments.md). Sweep parameters, run ensembles, measure sensitivity, and compare scenarios
    over many runs.
12. [Visualise](observe.md). Stream a run to the Rerun viewer and plot trajectories and results.
13. [Share](share.md). Turn a model into a system class of the catalog with its inventory entry and gallery example.
14. [Extend the library](extend_the_library.md). Give the general classes the words of your field, and find where a
    new part belongs.
15. [Case studies](case_studies/index.md). Follow one system of each of three fields through the whole workflow,
    against data or an exact result.

The [glossary](glossary.md) defines the terms of the guide, and [Troubleshooting](troubleshooting.md) shows each error
of the library with its cause and its fix.
