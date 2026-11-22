# User guide

This guide follows the work of a modeler in order. It starts from a ready-made system, changes it, builds a system of
your own, and then simulates, analyses, evaluates, calibrates, and shares it. Each page builds on the ones before it.
The first run takes three lines.

```python
from skcomplex.systems.biology.epidemiology import SIRSystem

SIRSystem().prepare().simulate().get_values('I.x').max().round()
```

The epidemic of `SIRSystem` peaks at 304 infected people out of 1000.

## If you know scikit-learn

This section maps the names of scikit-learn to those of the library. What you define is an estimator, and what you do
to it is a function.

| scikit-learn | scikit-complexity |
| --- | --- |
| an estimator, such as `Ridge`, with `get_params`, `set_params`, and `clone` | a component of a system, an element, an interaction, a space, or a system, such as `SIRSystem` |
| `Pipeline` | `System`, the composite of elements, interactions, and nested systems, with nested parameters such as `gravity__g` |
| `fit` | `prepare`, which checks and compiles a system once |
| `predict` | `simulate`, which returns a `Trajectory` |
| estimator tags | capability tags, from `get_tags()` |
| `all_estimators()` | `all_systems()` |
| `solver=` of `LogisticRegression` | `solver=` of a system, such as `GillespieSolver()` or `ScipyODESolver()` |
| `n_jobs` | `backend=` |
| `sklearn.metrics` | the analysis functions, such as `lyapunov` and `entropy` |
| `validation_curve` and repeated cross-validation | the experiment functions, such as `sweep` and `ensemble` |
| `GridSearchCV` | `calibrate`, which returns the fitted system in its result |

A [system](glossary.md#system) is a class whose constructor takes the parameters of its model and stores them.
`set_params` changes them, and `clone` copies them. `prepare` compiles the system once, and `simulate`, the analyses,
and the experiments read it and return frozen results. They never change its parameters.

## The running examples

Four small systems come back on many pages, each built on one kind of law. The README builds all four, and the pages
below take them further.

- **A falling ball.** `Element('ball', values={'x': 10.0, 'm': 1.0})` and a gravity whose `RateLaw` sets the second
  derivative `x.diff(t, 2)` to `-g`. It shows states, parameters, equations, and one prepared system run many times.
- **A reaction network.** The reaction `A + B -> C` as a `JumpLaw` that removes one `A` and one `B` and adds one `C`.
  It runs as random jumps with `GillespieSolver` and as rate equations with `ScipyODESolver`, by changing only
  `solver=`.
- **Heat in a rod.** One value `u` per cell of a `LatticeSpace`, with `RateLaw(u.diff(t), D * laplacian(u))`. It shows
  elements with one member per site and the operators of a space.
- **The Ising model.** One spin per site of a square lattice, with its energy over the temperature as an
  `ObjectiveLaw`, sampled by `MetropolisSolver`. It shows objectives, sampling solvers, and sweeps over a parameter.

The four look like this in one block. The README explains each line.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw, ObjectiveLaw, RateLaw
from skcomplex.core.components.laws.utils import laplacian, neighbors
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.engine.solvers import MetropolisSolver, ScipyODESolver
from skcomplex.core.systems import System

gravity = Interaction('gravity', roles={'target': 'ball'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t, 2), -9.81)])
ball = System(elements=[Element('ball', values={'x': 10.0, 'm': 1.0})], interactions=[gravity])

bind = Interaction(
    'bind',
    roles={'a': 'A', 'b': 'B', 'c': 'C'},
    laws=lambda s: [JumpLaw(0.01 * s.a.n * s.b.n, {s.a.n: -1, s.b.n: -1, s.c.n: +1})],
)
molecules = [Element(name, values={'n': n}) for name, n in (('A', 100), ('B', 80), ('C', 0))]
jumps = System(elements=molecules, interactions=[bind])
mean = System(elements=molecules, interactions=[bind], solver=ScipyODESolver())

start = np.zeros(21)
start[10] = 1.0
rod = Element('rod', values={'u': start}, space=LatticeSpace(21, boundary={'u': ('fixed', 0.0)}))
conduction = Interaction(
    'conduction',
    roles={'target': 'rod'},
    laws=lambda s: [RateLaw(s.target.u.diff(s.t), laplacian(s.target.u))],
)
heat = System(elements=[rod], interactions=[conduction])

nearest = LatticeNeighborhood(((-1, 0), (0, -1), (0, 1), (1, 0)))


def ising(s):
    spin = s.target.spin
    return [ObjectiveLaw(-spin * sum(neighbors(spin, nearest, reduce='each')) / 1.5, spin)]


spins = Element('spins', values={'spin': np.ones(256)}, space=LatticeSpace((16, 16)))
magnet = System(
    elements=[spins],
    interactions=[Interaction('ising', roles={'target': 'spins'}, laws=ising)],
    solver=MetropolisSolver(moves={'spins.spin': 'flip'}, neighborhood=nearest),
)
[system.prepare().form_ for system in (ball, jumps, mean, heat, magnet)]
```

The five systems are ordinary differential equations, a jump process, ordinary differential equations again, ordinary
differential equations with one equation per cell, and Monte Carlo sampling. Nobody named the problem: `prepare` read
it from the kinds of law and the solver.

## Two ways in

You can start from the catalog or from the core.

- **Select a ready-made system.** Import a system class from its subfield, such as `SIRSystem` from
  `skcomplex.systems.biology.epidemiology`, build it with the parameters you choose, and run it with
  `prepare().simulate()`.
  `all_systems` lists the classes of the catalog by field, topic, or tag.
- **Build a custom system.** Write elements and interactions with `skcomplex.core.components`, put them in a `System`
  of `skcomplex.core.systems`, run it with `prepare().simulate()`, and study the results with
  `skcomplex.core.analysis`.

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
   [Choosing components](core_components/choosing_components.md) leads from the features of a problem to its components.
5. [Build a custom system](build_a_custom_system/index.md). Write a model of hares and lynx from core components, as a
   subclass of a shared recipe, and as a system class.
6. [Check a system](check.md). Read the tags and requirements of components, and fix the validation errors a builder
   meets most.
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
    new component belongs.
15. [Case studies](case_studies/index.md). Follow one system of each of three fields through the whole workflow,
    against data or an exact result.

The [glossary](glossary.md) defines the terms of the guide, and [Troubleshooting](troubleshooting.md) shows each error
of the library with its cause and its fix.
