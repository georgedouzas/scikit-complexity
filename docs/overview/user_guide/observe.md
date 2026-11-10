# Observe

This tutorial records the quantities of a run that the states do not hold, watches a run in the
[Rerun](https://rerun.io) viewer with a layout of its views, and plots trajectories and experiment results with
matplotlib. It follows the hare and lynx model of the tutorial [Build a custom system](build_a_custom_system/index.md).

## The running example

`LotkaVolterraSystem` of `skcomplex.systems.biology.ecology` holds the predator and prey model of Lotka and Volterra
with the rates of Carpenter (2018). It starts from the pelt records of 1900, 30.0 thousand hares and 4.0 thousand lynx.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
simulate(system, until=20, dt=1.0).get_values('prey.x').round(1)[:6].tolist()
```

The hares rise from 30.0 to 81.2 thousand in 1903 and fall to 18.6 thousand in 1905, as in the build tutorial.

## Observables

An observable is a quantity that a run computes at every frame from the states, such as the total of two populations.
An [`ObservableSpec`][skcomplex.core.specs.ObservableSpec] pairs a label with an expression, a function of the
namespace `s` of the system that returns a symbolic expression. `simulate` takes the observables in `observables`,
records them in the trajectory under their labels, and logs them to Rerun when an observer runs.

The Lotka and Volterra model conserves a quantity along every cycle,
`V = b H - d ln H + a L - r ln L` for hares `H` and lynx `L`. Recording it checks that the solver keeps it constant.

```python continuation
import numpy as np
from sage.all__sagemath_symbolics import log

from skcomplex.core.specs import ObservableSpec


def total(s):
    return s.prey.x + s.predators.x


def conserved(s):
    return 0.024 * s.prey.x - 0.80 * log(s.prey.x) + 0.028 * s.predators.x - 0.55 * log(s.predators.x)


observables = [ObservableSpec('total', total), ObservableSpec('V', conserved)]
observed = simulate(system, until=20, dt=1.0, observables=observables)
observed.get_values('total').round(1)[:4].tolist(), np.ptp(observed.get_values('V')) < 1e-4
```

The two populations together grow from 34.0 to 103.7 thousand animals in 1903. The quantity `V` varies by less than
0.0001 over the twenty years, so the solver keeps it constant up to its tolerance. A drift of `V` would show that the
solver loses accuracy over long runs.

An observable must pickle with its system, so its functions are module-level functions. A lambda raises an error.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    simulate(system, until=1, observables=[ObservableSpec('total', lambda s: s.prey.x + s.predators.x)])
except ValidationError as error:
    message = str(error)
message.splitlines()[1][:80]
```

The message says that the expression of the observable `'total'` cannot be pickled, and asks for a module-level
function or a symbolic expression instead of a lambda or a local function.

### Observables of a group

A group holds many identical members, and an observable may reduce a variable of every member to one number. The
reduction is `count`, `sum`, `mean`, `min`, `max`, `var`, or a function. Here a group of 100 phase oscillators, each
turning at its own frequency, records the mean of the phase vector `exp(i theta)` of the members, and the order
parameter, its modulus, is an expression that reads the earlier observable by its label as `s.z`. An order parameter
near 1 means the phases sit together, and near 0 that they spread around the circle.

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
order = simulate(oscillators, until=50, dt=1.0, observables=observables).get_values('order')
round(float(order[0]), 3), round(float(order[-1]), 3)
```

The oscillators start within one radian of each other and drift apart at their own frequencies, so the order falls
from 0.958 to 0.12.

### Laws that read observables

A law may react to a quantity of the whole system, such as the mean field of all oscillators. It reads an observable
through a role bound to the label of the observable, as `s.<role>.value`. The interaction below pulls every oscillator
towards the mean phase, with the means `X` of `cos theta` and `Y` of `sin theta` as its roles.

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
X, Y = synchronised.get_values('X'), synchronised.get_values('Y')
round(float(np.hypot(X[0], Y[0])), 2), round(float(np.hypot(X[-1], Y[-1])), 2)
```

The phases start spread over the whole circle, with an order parameter of 0.04, and the coupling gathers them, so the
order parameter, the modulus of `(X, Y)`, ends at 1.0. When the law reads the value depends on where the observable and
the law live:

- A symbolic observable is an expression, or a reduction with a string `reduce` over an element group. Laws of its own
  partition read it at the current state. The compiled problem evaluates it before the law, so the value the law reads
  equals the recorded value at every frame.
- A law of another partition reads the observable as held at the last synchronisation point. This is how agents and
  equations exchange macroscopic quantities in a co-simulation.
- An observable with a Python function as `reduce`, or one over agents, is read only from another partition. Reading it
  in its own partition raises an error.

## Watching a run with Rerun

Rerun is a viewer for data that change in time. [`RerunObserver`][skcomplex.core.engine.viz.RerunObserver] logs the
frames of a run to it, and `simulate` takes it in `observers`. It sends the frames to a viewer it spawns with
`spawn=True`, to a running viewer with `connect=True`, or to a `.rrd` file with `save_path`. The blocks of this page
save to a temporary file, so they run without a viewer.

```python
import tempfile
from pathlib import Path

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
folder = Path(tempfile.mkdtemp())
observer = RerunObserver('pelts', save_path=folder / 'pelts.rrd', interval=5)
simulate(system, until=20, dt=0.1, observers=[observer])
(folder / 'pelts.rrd').exists(), observer.logged_frames_
```

The recording exists, and with `interval=5` the observer logged every fifth of the 201 frames, 41 in all. Open it with
`rerun pelts.rrd`. The entity paths follow the system tree, so the hares appear under `prey` and the lynx under
`predators`, and the parameters under `parameters/`. Each frame sets the timelines `sim_time`, the simulated time, and
`step`, the index of the frame.

### Visuals

A visual draws one element or group of a system in Rerun, and each type of element declares its default visual. The
populations of the example are scalar elements, so their visual is a
[`ScalarsVisual`][skcomplex.core.engine.viz.ScalarsVisual] that logs every state as a time series. The prepared observer
holds the visual of every element in `visuals_`.

```python continuation
[(type(visual).__name__, visual.path) for visual in observer.visuals_]
```

Both populations are drawn by a `ScalarsVisual`, one at the path `prey` and one at `predators`. The visuals of
[`skcomplex.core.engine.viz`][skcomplex.core.engine.viz] are named by a kind, which a
[`VisualSpec`][skcomplex.core.specs.VisualSpec] gives:

| Kind | Visual | What it draws |
| --- | --- | --- |
| `'scalars'` | `ScalarsVisual` | every state component of an element as a time series |
| `'points'` | `PointsVisual` | a vector state as points, and other states as time series |
| `'agents'` | `AgentsVisual` | the agents of a group as one batch per frame |
| `'histograms'` | `HistogramVisual` | histograms of agents or of groups without a space |
| `'image'` | `ImageVisual` | every state of a lattice group as an image coloured by value |
| `'sites'` | `SiteVisual` | every state of a non-square lattice group as coloured sites |
| `'cayley'` | `CayleyVisual` | every state of a lattice group on a group space as a graph |
| `'mesh'` | `MeshVisual` | every state of a finite element field as a triangle mesh |
| `'cell'` | `CellVisual` | a state on the edges or faces of a complex |
| `'phase_ring'` | `PhaseRingVisual` | the wrapped phases of oscillators against their position |
| `'axes'` | `AxisVisual` | the axes of a tensor state or the arrows of a vector state |
| `'bodies'` | `BodiesVisual` | bodies at positions computed from the states of an element |
| `'matrix'` | `MatrixVisual` | a matrix-valued state or parameter as an image |
| `'expectations'` | `ExpectationVisual` | the local expectations of operator elements and groups |
| `'geometry'` | `GeometryVisual` | the structure of a rewritten element |
| `'linear_program'` | `LinearProgramVisual` | the linear program of an element |

The default visuals follow the kind of element:

- Particles appear as 3D points, and populations as time series.
- Agents on grids and in continuous spaces appear as points coloured by a state, and agents on networks as graph nodes.
- Reporters appear under `reporters/`.
- A group is logged as one batch per state.
- Points and agents in a continuous space without a given `radius` take a share of the extent of the view. A group of
  at most ten points draws the trail of its recent positions, which `trail=0` turns off.
- A one-dimensional lattice also logs a space-time image, one row per logged frame with the oldest at the top, holding
  at most `ImageVisual(history=...)` rows.
- A complex state of a lattice or a mesh appears as two images. Its modulus goes under `<path>/<state>/modulus`,
  coloured over the range, and its phase under `<path>/<state>/phase`, coloured by a cyclic palette over `(-π, π]`. A
  complex scalar appears as the series `<path>/<state>/real` and `<path>/<state>/imag`.

An element or a group sets its own visual with its argument `visual`. Below, two atoms joined by a spring move in
three dimensions, and the second atom asks for points coloured by its velocity `x_d1` with
`VisualSpec('points', {'position': 'x', 'color_by': 'x_d1'})`.

```python continuation
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.specs import StateSpec, VisualSpec


class AtomElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        x: Annotated[object, StateSpec(vector=True, order=2, role='position')] = 0.0,
        x_d1: object = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1


speed = VisualSpec('points', {'position': 'x', 'color_by': 'x_d1'})
molecule = CompositeSystem(
    elements=[
        AtomElement('a', x=[0.0, 0.0, 0.0]),
        AtomElement('b', visual=speed, x=[1.5, 0.0, 0.0], x_d1=[0.0, 0.5, 0.0]),
    ],
    interactions=[
        RateInteraction('pull', target='a', sources=['b'], rate=lambda s: 4.0 * (s.sources[0].x - s.target.x)),
        RateInteraction('push', target='b', sources=['a'], rate=lambda s: 4.0 * (s.sources[0].x - s.target.x)),
    ],
    space=ContinuousSpace(n_dim=3),
)
atoms = RerunObserver('molecule', save_path=folder / 'molecule.rrd')
simulate(molecule, until=5, dt=0.02, observers=[atoms])
[type(visual).__name__ for visual in atoms.visuals_], type(atoms.views_[0]).__name__
```

The first atom keeps the default visual of a new type of element, a `ScalarsVisual` of time series. The second is drawn
by a `PointsVisual` with the colours of its speed, and its 3D spatial view comes first. Every visual subclasses
[`BaseVisual`][skcomplex.core.engine.viz.BaseVisual]. `prepare(problem)` finds the element at the entity path `path` in
a compiled problem and sets its views in `views_`, and `run(frame)` logs the element at one frame to the recording
`recording`. A new visual implements the hook `_log` for elements or `_log_agents` for agents, declares its views with
`_views`, and is given to a `VisualSpec` as its class.

### Layouts and blueprints

A Rerun blueprint arranges the views of the viewer. Every visual declares the views that show what it logs, and
`RerunObserver(blueprint='default')` arranges them into a blueprint. A blueprint of your own may replace it. Preparing
an observer on a compiled problem builds the blueprint without sending it anywhere, so it can be inspected first.

```python continuation
problem = simulate(system, until=0).problem
layout = RerunObserver(blueprint='default').prepare(problem)
[type(view).__name__ for view in layout.views_], [str(view.origin) for view in layout.views_]
```

The populations are scalars of one type, so they share one time series view, a `TimeSeriesView` with the origin `/`.
The views follow what is logged:

- Scalars go in time series views.
- Points go in 2D or 3D spatial views, by their dimension.
- Lattices and fields go in one 2D view per state, after the 2D views of their space-time images in one dimension.
- Meshes go in 3D views.
- Agents go in a 2D view, or a graph view on a network. Agents without a space get no spatial view.

The default blueprint arranges these views:

- Spatial views come first and side by side, with time series views below them.
- Scalar elements of one type share a time series view with at most six series.
- Parameters are left out unless `parameters=True`.
- The side panels are collapsed, the top panel is hidden, and the time panel follows the latest frame.

`primary` names the view to put first, so the first view shows the behaviour that defines the system. The observer
sends the blueprint, kept as `blueprint_`, with the recording, and writes it next to a saved recording as a `.rbl` file
with the same stem. Below, a blueprint of your own shows the hares alone, with the side panels collapsed.

```python continuation
import rerun.blueprint as rrb

hares_only = rrb.Blueprint(rrb.TimeSeriesView(origin='/prey', name='hares'), collapse_panels=True)
custom = RerunObserver('pelts', save_path=folder / 'hares.rrd', blueprint=hares_only)
simulate(system, until=20, observers=[custom])
sorted(file.name for file in folder.glob('hares.*'))
```

The folder holds the recording `hares.rrd` and its blueprint `hares.rbl`. Open both with
`rerun hares.rrd hares.rbl` to see the layout.

## Plots with matplotlib

A figure for a paper or a report is drawn from the arrays of a trajectory or an experiment result. The blocks below use
the backend `'Agg'` of matplotlib, which draws into files and opens no window. The data are the pelt records of 1900 to
1920, in thousands, as distributed with the case study of Carpenter (2018, *Predator-Prey Population Dynamics: the
Lotka-Volterra model in Stan*) in the file `knitr/lotka-volterra/hudson-bay-lynx-hare.csv` of the Stan example models.
The tutorial [Build a custom system](build_a_custom_system/index.md) traces their source.

```python
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

years = np.arange(1900, 1921)
hares = np.array(
    [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1]
    + [40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
)
lynx = np.array(
    [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4]
    + [8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]
)
system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
trajectory = simulate(system, until=20, dt=0.1)
times = 1900 + np.asarray(trajectory.times)

figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.plot(times, trajectory.get_values('prey.x'), label='hares, model')
left.plot(times, trajectory.get_values('predators.x'), label='lynx, model')
left.plot(years, hares, 'o', label='hares, pelts')
left.plot(years, lynx, 's', label='lynx, pelts')
left.set(xlabel='year', ylabel='thousands', title='Populations')
left.legend()
right.plot(trajectory.get_values('prey.x'), trajectory.get_values('predators.x'))
right.plot(hares, lynx, 'o')
right.set(xlabel='hares (thousands)', ylabel='lynx (thousands)', title='Phase plane')
path = Path(tempfile.mkdtemp()) / 'pelts.png'
figure.savefig(path)
plt.close(figure)
path.exists(), len(left.lines), len(right.lines)
```

The left panel draws the model against the records over time, four lines with the two series of points. The right
panel draws the phase plane, the lynx against the hares, where the model runs on a closed cycle and the records circle
around it. `get_values` returns the series of a state path and `times` the times of the frames, which the block shifts
to calendar years.

An experiment result gives its arrays in the same way. Below, a sweep of the growth rate of the hares is drawn with
the peak of the hares on one axis and the rate on the other, from `get_column` of the result.

```python continuation
from skcomplex.core.experiments import sweep


def hare_peak(trajectory):
    return float(trajectory.get_values('prey.x').max())


rates = np.linspace(0.4, 0.7, 7)
swept = sweep(system, {'r': rates}, until=20, dt=0.5, outputs=[hare_peak])
figure, axes = plt.subplots(figsize=(5, 4))
axes.plot(swept.get_column('r'), swept.get_column('hare_peak'), 'o-')
axes.set(xlabel='growth rate of the hares (per year)', ylabel='highest count of hares (thousands)')
figure.savefig(path.with_name('sweep.png'))
plt.close(figure)
swept.get_column('hare_peak').round(1).tolist()
```

The highest count of hares rises with their growth rate, from 65.0 thousand at `r = 0.4` to 98.6 thousand at
`r = 0.7`. The page [Run experiments](run_experiments.md) runs the other experiments, whose results hold arrays to plot
in the same way.
