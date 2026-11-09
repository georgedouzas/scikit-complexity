# User guide

This guide follows what you do as a modeler, in order. Start with a ready-made system, change it, then build your own,
simulate it, observe it, analyse it, run experiments over many runs, and extend the library with parts of your own.
Each part builds only on the ones before it.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import SIRSystem

simulate(SIRSystem()).values('I.x').max().round()
```

## If you know scikit-learn

The library follows scikit-learn. What you define is an estimator, and what you do to it is a function.

| scikit-learn | scikit-complexity |
| --- | --- |
| an estimator, such as `Ridge`, with `get_params`, `set_params`, and `clone` | a part of a system, an element, an interaction, a law, or a system, such as `SIRSystem` |
| `Pipeline` | `System`, the composite of elements, interactions, and nested systems |
| `fit` | `prepare` |
| estimator tags | capability tags, from `get_tags()` |
| `all_estimators()` | `all_systems()` |
| `cross_validate(model, X, y, cv=...)` | `simulate(system, solver=...)`, which returns a `Trajectory` |
| `n_jobs` | `backend=` |
| `sklearn.metrics` | the analysis functions, such as `lyapunov` and `entropy` |
| `validation_curve` and repeated cross-validation | the experiment functions, such as `sweep` and `ensemble` |
| `GridSearchCV` | `calibrate`, which returns the fitted system in its result |

So a system is a class whose constructor takes the parameters of its model and stores them, `set_params` changes them,
and `clone` copies them. `simulate`, the analyses, and the experiments read the system and return frozen results, and
never change it.

## Two ways in

- **Select a ready-made system.** You import a system class from its subfield, such as `SIRSystem` from
  `skcomplex.systems.biology.epidemiology`, build it with the parameters you choose, and simulate it with `simulate`.
  `all_systems` lists the system classes of the catalog by field, topic, or tag.
- **Build a custom system.** You write elements and interactions with `skcomplex.core.components`, use specs from
  `skcomplex.core.specs`, simulate with `skcomplex.core.engine`, and study the results with `skcomplex.core.analysis`.

Every ready-made system is written with the same classes as a custom one, so the page of a system under
[Systems](../../systems/index.md) also works as a pattern. The page [System classes](system_classes.md) shows how a
system class builds its parts, and how to change a ready-made one by subclassing it. The section
[Choosing compatible components](build_a_custom_system/choosing_compatible_components.md) shows how to find which parts
of a custom system fit together.

Read the parts in order.

1. [Select a ready-made system](select_a_ready_made_system.md)
2. [Change it](change_it.md)
3. [System classes](system_classes.md)
4. [Build a custom system](build_a_custom_system/index.md)
5. [Simulate](simulate.md)
6. [Observe](observe.md)
7. [Analyse](analyse.md)
8. [Run experiments](run_experiments.md)
9. [Extend the library](extend_the_library.md)
