# Learning by gradient steps

This page fits the parameters of a system to data by gradient descent, as a neural network is trained. A system that
learns is a dynamical system in two times. The inner time steps a system through its inputs. The outer time steps
through training, in epochs, and its states are the trained parameters of the inner system. Its law is a gradient step
on a loss of the runs of the inner system.

## Training a system

This section fits the decay rate of the map `x -> a x` to the series `0.8^k`, starting from the guess `a = 0.5`. The
[`Element`](../core_components/elements.md#element) `unit` holds the value `x`, the rate `a`, and the observed value,
the interaction `decay` sets an [`UpdateLaw`](../core_components/laws.md#updatelaw), and a nested system `decay` holds
both. A [`TrainingInteraction`](../core_components/interactions.md#traininginteraction) binds the nested system, names
the trained rate by its path `unit.a`, reads the observed series from a `DatasetSpec`, and minimises the squared error
over 300 epochs.

```python
import numpy as np

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction, TrainingInteraction
from skcomplex.core.components.laws import DatasetSpec, UpdateLaw
from skcomplex.core.systems import System


def squared_error(s):
    return (s.unit.x - s.unit.observed) ** 2


series = 0.8 ** np.arange(1, 11)
unit = Element('unit', values={'x': 1.0, 'a': 0.5, 'observed': 0.0})
decay = Interaction('decay', roles={'target': 'unit'}, laws=lambda s: [UpdateLaw(s.target.x, s.target.a * s.target.x)])
system = System('decay', elements=[unit], interactions=[decay])
training = TrainingInteraction(
    'fit',
    system='decay',
    trained=['unit.a'],
    loss=Observable('error', squared_error),
    data=DatasetSpec({'unit.observed': series[np.newaxis]}),
    learning_rate=0.05,
    batch_size=1,
)
fitting = System('fitting', systems=[system], interactions=[training])
trajectory = fitting.prepare().simulate(until=300, random_state=0)
round(float(trajectory.get_values('decay/unit.a')[-1]), 3), bool(trajectory.get_values('fit/loss')[-1] < 1e-6)
```

The trained rate is 0.8, the rate of the data, and the loss of the last epoch is below `1e-6`.
`TrainingInteraction`, of `skcomplex.core.components.interactions`, writes this with the existing components.

- The inner system is a nested system, such as `decay` above.
- The interaction names the trained parameters by their paths in `trained`, such as `unit.a`, and no mark goes on the
  value.
- The interaction binds the inner system through its role `system`.
- Each system keeps the time its laws imply. The inner map steps through its inputs, and the outer system steps once
  per epoch, so no time is given to either.

`TrainingInteraction` states no laws of its own, since `GradientDescentSolver` reads its settings, and it differs
from [`GradientInteraction`](../core_components/general.md#gradientinteraction).

- `GradientInteraction` moves the states of its target down the gradient of a potential, in the own time of the
  system, and compiles to rate laws.
- `TrainingInteraction` takes optimizer steps on the trained parameters of a nested system once per epoch. It follows
  the gradient of a loss over unrolled runs of that system, computed by automatic differentiation with JAX.

## The components of training

This section describes each component of a training run and the options it takes.

- **The inner system.** It is an ordinary system, a discrete map or ordinary differential equations written as
  symbolic laws, so it runs, is analysed, and is visualised alone, before or after training. The trained system of a
  frame is the nested system with the recorded values set through `set_params`. A system with noise, jumps, rules, or
  laws given as Python callables reports `learning-not-differentiable`.
- **The data.** `DatasetSpec(values, weights=None, times=None)`, of `skcomplex.core.specs`, maps dotted parameter
  paths of the inner system, its inputs and targets, to arrays of shape `(sequences, steps)` followed by the shape of
  the parameter. At inner step `k`, every parameter takes its value at step `k` before the laws of the step are
  evaluated, as a schedule sets a parameter, and the loss reads the states that the step gives. Every sequence starts
  from the initial states of the system. `weights` masks steps out of the loss. It is one array for every observable,
  or a mapping from the label of the loss or of a metric to its own weights. `times` gives the inner times of an ODE
  system, `1, 2, ...` by default.
- **The loss and the metrics.** `loss` and `metrics` are `Observable`s over the symbols of the inner system. The
  loss of a batch is the weighted mean of the observable over its sequences and steps. The trajectory records the
  trained parameters under their inner paths, such as `decay/unit.a`, the mean batch loss of every epoch as
  `<interaction>/loss`, and every metric over the whole dataset as `<interaction>/<metric>`. The same observables given
  to `simulate` with the trained system record the loss in a forward run.
- **The optimizer.** `optimizer` is one of `OptimizerKind`, `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'`, run by
  Optax. Its hyperparameters, `learning_rate`, `momentum`, `beta1`, and `beta2`, the decay of `'rmsprop'`, are
  parameters of the interaction. So a schedule such as `{'fit.learning_rate': ...}` changes them at the next frame.
  `batch_size` sets the number of sequences of a batch, shuffled by the random stream of the run, so a seed repeats a
  run.
- **Truncation.** `truncation` cuts backpropagation through time after that many inner steps, and `None` follows the
  gradient through the whole run. A truncation of 1 holds the previous inner states as constant inputs, the original
  rule of Elman.
- **The solver.** `GradientDescentSolver`, of `skcomplex.core.engine.solvers`, is the default solver of a system that
  trains. It unrolls a discrete map with `jax.lax.scan` and an ODE with diffrax, computes in double precision by
  default, and keeps the state of the optimizer between frames. The inner system always compiles with `JaxBackend`, so
  a backend passed to `simulate` is ignored.
- **Flat functions.** `heaviside`, `sign`, `floor`, and `ceil` have a zero gradient almost everywhere. So a loss that
  reads a parameter only through them does not train it, and no error says so.
- **Local rules.** Local learning rules, such as those of Hebb and Oja, change a weight from its own activity. So they
  are update or rate laws on weight states, as in `HopfieldSystem`, and need no training interaction.

`ElmanNetworkSystem` of the catalog trains the simple recurrent network of Elman on the temporal XOR stream. Its
visual draws the loss and the metrics as the first view, and the trained matrices as heat maps through `MatrixVisual`,
chosen by `VisualSpec('matrix')`.

## Shaped parameters

This section gives an interaction a weight matrix whose shape follows the groups it binds. A parameter of an
interaction given as an array keeps its shape, here one row per target member and one column per source member. Three
cells, the members of an [`Element`](../core_components/elements.md#element) with `size=3`, relax towards the weighted
sum of two inputs through an [`Interaction`](../core_components/interactions.md#interaction) subclass with a
[`RateLaw`](../core_components/laws.md#ratelaw). A schedule doubles the weights at time 20.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import contraction, count, total
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class Drive(Interaction):
    def __init__(self, label, *, target: Role, source: Role, W=0.0):
        super().__init__(label)
        self.target = target
        self.source = source
        self.W = W

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [RateLaw(s.target.x, -s.target.x + total(s.source.x, weights=s.W))]


W = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 2.0]])
system = System(
    elements=[Element('cells', values={'x': 0.0}, size=3), Element('inputs', values={'x': [1.0, 2.0]}, size=2)],
    interactions=[Drive('drive', target='cells', source='inputs', W=W)],
).prepare()
schedule = {'drive.W': lambda t: W if t < 20 else 2 * W}
trajectory = system.simulate(until=40, dt=0.1, schedule=schedule)
trajectory.get_values('cells.x')[200].round(3).tolist(), trajectory.get_values('cells.x')[-1].round(3).tolist()
```

At time 20 the cells sit at 1.0, 1.5, and 4.0, the products of `W` with the inputs 1 and 2. At time 40, after the
doubling, they sit at 2.0, 3.0, and 8.0. A shaped parameter behaves like any parameter.

- It holds a block of the parameter vector in row-major order.
- Trajectories read it with its shape, and a schedule, such as the doubling of `W` above, or an experiment sets it
  with its shape.
- A training interaction trains it when it names it in `trained`.
- The weights of `total` and `mean` read it.

## Coupling tensors

This section couples the members of a group by a tensor of higher order. An array with `k` indices, one per member,
gives a coupling tensor of order `k` among the members of a group.
`contraction(T, a, b)`, of `skcomplex.core.components.laws.utils`, returns `sum_jk T_ijk a_j b_k` for every member
`i`, with one expression per index after the first. Matrices keep `total(x, weights=W)`. Four species grow
logistically and compete in triplets, each triplet with the coefficient -1.

```python continuation
class Triplets(Interaction):
    def __init__(self, label, *, target: Role, T=0.0):
        super().__init__(label)
        self.target = target
        self.T = T

    def _prepare(self, s):
        super()._prepare(s)
        x = s.target.x
        self.laws_ = [RateLaw(x, x * (1 - x) + x * contraction(s.T, x, x) / count(over=s.target))]


species = Element('species', values={'x': 0.1}, size=4)
triplets = Triplets('triplets', target='species', T=-np.ones((4, 4, 4)))
community = System(elements=[species], interactions=[triplets]).prepare()
trajectory = community.simulate(until=50, dt=0.1)
trajectory.get_values('species.x')[-1].round(3).tolist()
```

Every species settles at 0.39. With all coefficients -1 the rate is `x (1 - x - 4 x^2)`, whose positive root
`(sqrt(17) - 1) / 8` is 0.390.

A tensor of order `k` over `N` members holds `N^k` entries, a million for order 3 over 100 members.
`HigherOrderRandomLotkaVolterraSystem` builds the random community of Sidhom and Galla (2024) on such a tensor.
