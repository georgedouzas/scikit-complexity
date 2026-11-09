# Observe

Record the quantities you care about at every frame, check that conserved ones stay constant, and watch a run live.

## Observables

Record the quantities you care about at every frame, such as the order of a group of oscillators, beside the states of
a run. An `ObservableSpec` is a label with an expression or a group reduction, which a run computes at every frame,
records in the trajectory under the path of the system, and logs to Rerun. Pass observables to `simulate` with
`observables`.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import I, exp, pi

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec


class OscillatorElement(Element):
    def __init__(
        self,
        label,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s):
        return [RateLaw(s.theta, s.omega)]


def phase_vector(member):
    return exp(I * member.theta)


def order_parameter(s):
    return abs(s.z)


values = {'theta': np.linspace(0, 1, 100), 'omega': np.linspace(0.9, 1.1, 100)}
oscillators = CompositeSystem(elements=[GroupElement('phases', element=OscillatorElement, size=100, values=values)])
observables = [ObservableSpec('z', phase_vector, group='phases'), ObservableSpec('order', order_parameter)]
order = simulate(oscillators, until=50, dt=1.0, observables=observables).values('order')
round(float(order[0]), 3), round(float(order[-1]), 3)
```

An observable is a frozen record of one of two kinds:

- **An expression of the system namespace.** Earlier observables appear in it by label, as `s.z` in `order_parameter`.
- **A reduction over the members of a group.** It reduces a variable, or an expression of one member such as
  `phase_vector`, by `count`, `sum`, `mean`, `min`, `max`, `var`, or a function.

Expressions and reductions given as functions must be module-level functions, so that systems pickle. A lambda raises
an error.

Above, the oscillators start close together and drift apart at their own frequencies, so the order falls from near 1.

### Laws that read observables

Let a law react to a macroscopic quantity, such as the mean field of all oscillators. A law reads an observable through
a role bound to the path of the observable, as `s.<role>.value`.

```python continuation
from sage.all__sagemath_symbolics import cos, sin

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.specs import RoleSpec


class MeanFieldInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        X: Annotated[str, RoleSpec()],
        Y: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.target = target
        self.X = X
        self.Y = Y
        self.K = K

    def _get_laws(self, s):
        return [RateLaw(s.target.theta, s.K * (s.Y.value * cos(s.target.theta) - s.X.value * sin(s.target.theta)))]


def cosine(member):
    return cos(member.theta)


def sine(member):
    return sin(member.theta)


values = {'theta': np.linspace(0, 6, 100), 'omega': np.linspace(0.9, 1.1, 100)}
coupled = CompositeSystem(
    elements=[GroupElement('phases', element=OscillatorElement, size=100, values=values)],
    interactions=[MeanFieldInteraction('coupling', target='phases', X='X', Y='Y', K=2.0)],
)
observables = [ObservableSpec('X', cosine, group='phases'), ObservableSpec('Y', sine, group='phases')]
synchronised = simulate(coupled, until=20, dt=1.0, observables=observables)
round(float(synchronised.values('X')[-1]), 2)
```

When the law reads the value depends on where the observable and the law live:

- A symbolic observable is an expression, or a reduction with a string `reduce` over an element group. Laws of its own
  partition read it at the current state. The compiled problem evaluates it before the law, so the value the law reads
  equals the recorded value at every frame.
- A law of another partition reads the observable as held at the last synchronisation point. This is how agents and
  equations exchange macroscopic quantities in a co-simulation.
- An observable with a Python function as `reduce`, or one over agents, is read only from another partition. Reading it
  in its own partition raises an error.

## Visualizing with Rerun

Watch a run live in the [Rerun](https://rerun.io) viewer, or save it to a file to open later. `RerunObserver` streams
frames to a spawned viewer, a running viewer, or a `.rrd` file. Pass it to `simulate` with `observers`.

```python
from typing import Annotated

import tempfile
from pathlib import Path

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.specs import StateSpec


class AtomElement(Element):
    def __init__(
        self, label, *, x: Annotated[object, StateSpec(vector=True, order=2, role='position')] = 0.0, x_d1: object = 0.0
    ):
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1


molecule = CompositeSystem(
    elements=[AtomElement('a', x=[0.0, 0.0, 0.0]), AtomElement('b', x=[1.5, 0.0, 0.0], x_d1=[0.0, 0.5, 0.0])],
    interactions=[
        RateInteraction('pull', target='a', sources=['b'], rate=lambda s: 4.0 * (s.sources[0].x - s.target.x)),
        RateInteraction('push', target='b', sources=['a'], rate=lambda s: 4.0 * (s.sources[0].x - s.target.x)),
    ],
    space=ContinuousSpace(n_dim=3),
)
path = Path(tempfile.mkdtemp()) / 'molecule.rrd'
trajectory = simulate(molecule, until=5, dt=0.02, observers=[RerunObserver('molecule', save_path=path)])
path.exists()
```

Entity paths follow the system tree, each element type chooses its visual, and parameters appear under `parameters/`.
The default visuals are these:

- Particles appear as 3D points, and populations as time series.
- Agents on grids and in continuous spaces appear as points colored by a state, and agents on networks as graph nodes.
- Reporters appear under `reporters/`.
- A group is logged as one batch per state. `PointsVisual(position='x', color_by='x_d1')` colors its points by speed.
- Points and agents in a continuous space without a given `radius` take a share of the extent of the view. A group of
  at most ten points draws the trail of its recent positions, which `trail=0` turns off.
- A one-dimensional lattice also logs a space-time image, one row per logged frame with the oldest at the top, holding
  at most `ImageVisual(history=...)` rows.
- A complex state of a lattice or a mesh appears as two images. Its modulus goes under `<path>/<state>/modulus`, colored
  over the range, and its phase under `<path>/<state>/phase`, colored by a cyclic palette over `(-π, π]`. A complex
  scalar appears as the series `<path>/<state>/real` and `<path>/<state>/imag`.

An element or a group sets its own visual with its `visual` argument, such as
`GroupElement(..., visual=VisualSpec('points', {...}))`. A `VisualSpec(kind, options)` names a visual of the
visualization layer, such as `'points'`, `'image'`, or `'agents'`. `RerunObserver` builds it with the path of the
element and its recording.

Every visual subclasses `BaseVisual`, and implements the hook `_log` for elements or `_log_agents` for agents. Its
settings `path` and `recording` name the element and the recording it logs to. `visual.prepare(problem)` finds the
element in a compiled problem and sets its views `views_`, and `visual.run(frame)` logs the element at a frame.

### Layouts and blueprints

Arrange the views of the viewer with a Rerun blueprint, the default one or your own. Every visual declares the Rerun
views that show what it logs, as its fitted attribute `views_`.

```python continuation
import rerun.blueprint as rrb

from skcomplex.systems.biology.ecology import LotkaVolterraSystem

predation = LotkaVolterraSystem()
observer = RerunObserver('lotka_volterra', save_path=path.with_name('lotka_volterra.rrd'), blueprint='default')
simulate(predation, until=20, observers=[observer])
type(observer.views_[0]).__name__
sorted(file.name for file in path.parent.glob('lotka_volterra.*'))
custom = rrb.Blueprint(rrb.TimeSeriesView(origin='/prey', name='prey'), collapse_panels=True)
observer = RerunObserver('lotka_volterra', save_path=path.with_name('prey.rrd'), blueprint=custom)
simulate(predation, until=20, observers=[observer])
path.with_name('prey.rbl').exists()
```

The views follow what is logged:

- Scalars go in time series views.
- Points go in 2D or 3D spatial views, by their dimension.
- Lattices and fields go in one 2D view per state, after the 2D views of their space-time images in one dimension.
- Meshes go in 3D views.
- Agents go in a 2D view, or a graph view on a network. Agents without a space get no spatial view.

`RerunObserver(blueprint='default')` arranges these views into a Rerun blueprint:

- Spatial views come first and side by side, with time series views below them.
- Scalar elements of one type share a time series view with at most six series.
- Parameters are left out unless `parameters=True`.
- The side panels are collapsed, the top panel is hidden, and the time panel follows the latest frame.

`primary` names the view to put first. The prepared observer keeps the views in this order as `views_`, so its first
view is the one that shows the signature behaviour of the system. `RerunObserver` sends the default blueprint or a
blueprint of your own, `blueprint_`, and writes it next to a saved recording as a `.rbl` file.

Open both files with `rerun lotka_volterra.rrd lotka_volterra.rbl` to see the layout.
