# Learning by gradient steps

Fit the parameters of a system to data by gradient descent, as a neural network is trained. A system that learns is a
dynamical system in two times:

- The inner time steps a system through its inputs.
- The outer time steps through training. The trained parameters of the inner system are its states, and a gradient step
  on a loss of the runs of the inner system is its law.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import TrainingInteraction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DatasetSpec, ObservableSpec, ParameterSpec, StateSpec


class DecayElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        a: Annotated[float, ParameterSpec(adaptive=True)] = 0.5,
        observed: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label)
        self.x = x
        self.a = a
        self.observed = observed

    def _get_laws(self, s):
        return [UpdateLaw(s.x, s.a * s.x)]


def squared_error(s):
    return (s.unit.x - s.unit.observed) ** 2


series = 0.8 ** np.arange(1, 11)
system = System('decay', elements=[DecayElement('unit')], time=DiscreteTime())
training = TrainingInteraction(
    'fit',
    system='decay',
    trained=['unit.a'],
    loss=ObservableSpec('error', squared_error),
    data=DatasetSpec({'unit.observed': series[np.newaxis]}),
    learning_rate=0.05,
    batch_size=1,
)
fitting = System('fitting', systems=[system], interactions=[training], time=DiscreteTime())
trajectory = simulate(fitting, until=300, random_state=0)
round(float(trajectory.values('decay/unit.a')[-1]), 3), bool(trajectory.values('fit/loss')[-1] < 1e-6)
```

`TrainingInteraction`, of `skcomplex.core.components.interactions`, writes this with the existing parts:

- The inner system is a nested system, such as `decay` above.
- Its trained parameters are declared `ParameterSpec(adaptive=True)`.
- The interaction binds the inner system through its role `system`.
- The outer system runs under `DiscreteTime()`, one step per epoch.

`TrainingInteraction` returns one `GradientStepLaw`, of `skcomplex.core.components.laws`, which differs from
`GradientInteraction`:

- `GradientInteraction` moves the states of its target down the gradient of a potential, in the system's own time, and
  lowers to rate laws.
- `GradientStepLaw` takes optimizer steps on the adaptive parameters of a nested system once per epoch. It follows the
  gradient of a loss over unrolled runs of that system, computed by automatic differentiation with JAX.

The parts of training work as follows:

- **The inner system.** It is an ordinary system, a discrete map or ordinary differential equations written as symbolic
  laws, so it runs, is analysed, and is visualised alone, before or after training. The trained system of a frame is
  the nested system with the recorded values set through `set_params`. A system with noise, jumps, rules, or laws given
  as Python callables reports `learning-not-differentiable`.
- **The data.** `DatasetSpec(values, weights=None, times=None)`, of `skcomplex.core.specs`, maps dotted parameter paths
  of the inner system, its inputs and targets, to arrays of shape `(sequences, steps)` followed by the shape of the
  parameter. At inner step `k`, every parameter takes its value at step `k` before the laws of the step are evaluated,
  as a schedule sets a parameter, and the loss reads the states that the step gives. Every sequence starts from the
  initial states of the system. `weights` masks steps out of the loss. It is one array for every observable, or a
  mapping from the label of the loss or of a metric to its own weights. `times` gives the inner times of an ODE system,
  `1, 2, ...` by default.
- **The loss and the metrics.** `loss` and `metrics` are `ObservableSpec`s over the symbols of the inner system. The
  loss of a batch is the weighted mean of the observable over its sequences and steps. The trajectory records the
  trained parameters under their inner paths, such as `decay/unit.a`, the mean batch loss of every epoch as
  `<interaction>/loss`, and every metric over the whole dataset as `<interaction>/<metric>`. The same observables given
  to `simulate` with the trained system record the loss in a forward run.
- **The optimizer.** `optimizer` is one of `OPTIMIZERS`, `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'`, run by
  Optax. Its hyperparameters, `learning_rate`, `momentum`, `beta1`, and `beta2`, the decay of `'rmsprop'`, are
  parameters of the interaction. So a schedule such as `{'fit.learning_rate': ...}` changes them at the next frame.
  `batch_size` sets the number of sequences of a batch, shuffled by the random stream of the run, so a seed
  repeats a run.
- **Truncation.** `truncation` cuts backpropagation through time after that many inner steps, and `None` follows the
  gradient through the whole run. A truncation of 1 holds the previous inner states as constant inputs, Elman's
  original rule.
- **The solver.** `GradientDescentSolver`, of `skcomplex.core.engine.solvers`, is the default solver of a system that
  trains. It unrolls a discrete map with `jax.lax.scan` and an ODE with diffrax, computes in double precision by
  default, and keeps the state of the optimizer between frames. The inner system always compiles with `JaxBackend`, so a
  backend passed to `simulate` is ignored.
- **Flat functions.** `heaviside`, `sign`, `floor`, and `ceil` have a zero gradient almost everywhere. So a loss that
  reads a parameter only through them does not train it, without any error.
- **Local rules.** Local learning rules, such as Hebb's and Oja's, change a weight from its own activity. So they are
  update or rate laws on weight states, as in `HopfieldSystem`, and need no training interaction.

`ElmanNetworkSystem` trains Elman's simple recurrent network on the temporal XOR stream. Its visual draws the loss
and the metrics as the first view, and the trained matrices as heat maps through `MatrixVisual`, chosen by
`VisualSpec('matrix')`.

## Shaped parameters and coupling tensors

Give an interaction a weight matrix, or a tensor of higher order, whose shape follows the groups it binds.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import contraction, count, total
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class NodeElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


class DriveInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        W: Annotated[object, ParameterSpec(shape=('target', 'source'))] = 0.0,
    ):
        super().__init__(label)
        self.target = target
        self.source = source
        self.W = W

    def _get_laws(self, s):
        return [RateLaw(s.target.x, -s.target.x + total(s.source.x, weights=s.W))]


W = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 2.0]])
system = System(
    elements=[
        GroupElement('cells', element=NodeElement, size=3),
        GroupElement('inputs', element=NodeElement, size=2, values={'x': [1.0, 2.0]}),
    ],
    interactions=[DriveInteraction('drive', target='cells', source='inputs', W=W)],
)
schedule = {'drive.W': lambda t: W if t < 20 else 2 * W}
trajectory = simulate(system, until=40, dt=0.1, schedule=schedule)
trajectory.values('cells.x')[200].round(3).tolist(), trajectory.values('cells.x')[-1].round(3).tolist()
```

`ParameterSpec(shape=('target', 'source'))` declares a parameter of an interaction whose shape is the sizes of the
groups its roles bind. Here it is a weight matrix with one row per target member and one column per source member. It
behaves like any parameter:

- It holds a block of the parameter vector in row-major order.
- Trajectories read it with its shape, and a schedule, such as the doubling of `W` at time 20 above, or an experiment
  sets it with its shape.
- With `adaptive=True`, a training interaction trains it.
- The weights of `total` and `mean`, and of `CouplingInteraction(mode='group')`, read it.

A shaped parameter whose role binds a single element reports the issue `parameter-shape`, with the fix of binding a
group.

A role repeated `k` times, `shape=('target',) * k`, gives a coupling tensor of order `k` among the members of a group.
`contraction(T, a, b)`, of `skcomplex.core.components.laws.utils`, returns `sum_jk T_ijk a_j b_k` for every member `i`,
with one expression per index after the first. Matrices keep `total(x, weights=W)`.

```python continuation
class TripletInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        T: Annotated[object, ParameterSpec(shape=('target',) * 3)] = 0.0,
    ):
        super().__init__(label)
        self.target = target
        self.T = T

    def _get_laws(self, s):
        x = s.target.x
        return [RateLaw(x, x * (1 - x) + x * contraction(s.T, x, x) / count(over=s.target))]


species = GroupElement('species', element=NodeElement, size=4, values={'x': 0.1})
triplets = TripletInteraction('triplets', target='species', T=-np.ones((4, 4, 4)))
community = System(elements=[species], interactions=[triplets])
trajectory = simulate(community, until=50, dt=0.1)
trajectory.values('species.x')[-1].round(3).tolist()
```

A tensor of order `k` over `N` members holds `N^k` entries, a million for order 3 over 100 members. `HigherOrderRandomLotkaVolterraSystem`
builds the random community of Sidhom and Galla (2024) on such a tensor.
