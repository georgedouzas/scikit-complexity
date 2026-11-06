# Connected components

Assemble a system from components that connect through ports, as engineers do with circuits and plants. Each
component states equations and leaves open which variable each equation determines. So one resistor gives a current
under a voltage source and a voltage under a current source.

```python
from typing import Annotated, ClassVar

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ConnectorSpec, ParameterSpec, StateSpec

CHEMICAL = ConnectorSpec('chemical', potential='mu', flow='n')


class ReservoirElement(Element):
    ports: ClassVar = {'a': CHEMICAL}

    def __init__(
        self,
        label,
        *,
        a_mu: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        a_n: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        level: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label)
        self.a_mu = a_mu
        self.a_n = a_n
        self.level = level

    def _get_laws(self, s):
        return [EquationLaw(s.a_n, s.a_mu - s.level)]


system = System(
    elements=[ReservoirElement('first', level=1.0), ReservoirElement('second', level=3.0)],
    interactions=[ConnectionInteraction('node', members=['first', 'second'], ports=['a', 'a'])],
)
trajectory = Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_
assert trajectory.values('first.a_mu')[-1] == 2.0
assert trajectory.values('second.a_n')[-1] == -1.0
trajectory.values('first.a_mu')[-1], trajectory.values('second.a_n')[-1]
```

The two reservoirs share one node. The node makes their potentials equal and their flows sum to zero, so the potential
settles between the two levels and a flow runs from one reservoir into the other.

## Algebraic states

`StateSpec(algebraic=True)` declares a state that no law integrates and that the equations of the system determine at
every time.

- An algebraic state has order 1 and no lags.
- Its initial value is the starting guess of a nonlinear solve.
- A law other than an `EquationLaw` that targets it raises `AlgebraicStateError`.
- The trajectory records it at every frame, so observables and visuals read it like any other state.
- The simulation keeps it at its initial value in `state_`, since the solvers integrate only the other states.
- A law given as a function that reads values by name sees that initial value. Read it through a symbolic law instead.

## Ports and connections

`ConnectorSpec(name, potential, flow)` names the variables of a port, and the core holds no named connector. An element
type declares its ports in the class variable `ports`, a mapping from the name of a port to its connector. It takes one
algebraic state per variable of a port, named `<port>_<variable>`. A flow is positive into the element.

`ConnectionInteraction(label, members=..., ports=...)` is one node.

- It binds the elements of the node through a variadic role, and names the port of each in the same order.
- It returns the equations that make the potentials equal and the flows sum to zero.
- A node with many ports is one connection.
- A port that no connection names has a zero flow.

A circuit takes one connector and one element class per component. Here a capacitor discharges through a resistor.

```python
from typing import Annotated, ClassVar

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ConnectorSpec, ParameterSpec, StateSpec

ELECTRICAL = ConnectorSpec('electrical', potential='v', flow='i')


class TwoPinElement(Element):
    ports: ClassVar = {'p': ELECTRICAL, 'n': ELECTRICAL}

    def __init__(
        self,
        label,
        *,
        p_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
    ):
        super().__init__(label)
        self.p_v = p_v
        self.p_i = p_i
        self.n_v = n_v
        self.n_i = n_i

    def _get_laws(self, s):
        return [EquationLaw(s.p_i + s.n_i, 0)]


class ResistorElement(TwoPinElement):
    def __init__(
        self,
        label,
        *,
        p_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        R: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.R = R

    def _get_laws(self, s):
        return [*super()._get_laws(s), EquationLaw(s.p_v - s.n_v, s.R * s.p_i)]


class CapacitorElement(TwoPinElement):
    def __init__(
        self,
        label,
        *,
        p_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        u: Annotated[float, StateSpec()] = 0.0,
        C: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.u = u
        self.C = C

    def _get_laws(self, s):
        return [*super()._get_laws(s), EquationLaw(s.C * derivative(s.u, 1), s.p_i), EquationLaw(s.u, s.p_v - s.n_v)]


class GroundElement(Element):
    ports: ClassVar = {'p': ELECTRICAL}

    def __init__(
        self,
        label,
        *,
        p_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
    ):
        super().__init__(label)
        self.p_v = p_v
        self.p_i = p_i

    def _get_laws(self, s):
        return [EquationLaw(s.p_v, 0)]


circuit = System(
    elements=[CapacitorElement('tank', u=1.0, C=0.5), ResistorElement('load', R=2.0), GroundElement('ground')],
    interactions=[
        ConnectionInteraction('top', members=['tank', 'load'], ports=['p', 'p']),
        ConnectionInteraction('bottom', members=['tank', 'load', 'ground'], ports=['n', 'n', 'p']),
    ],
)
trajectory = Simulation(circuit, dt=0.1).prepare().run(until=1.0).trajectory_
assert np.isclose(trajectory.values('tank.u')[-1], np.exp(-1.0), rtol=1e-3)
assert np.allclose(trajectory.values('load.p_i').ravel(), trajectory.values('tank.u').ravel() / 2.0)
trajectory.values('tank.u')[-1].round(4)
```

The voltage of the capacitor decays as `exp(-t / RC)`, and the current through the load is that voltage over `R`. The
systems of the field `engineering` in the catalog hold electrical and thermal components built in the same way. A port
component drives a mechanism through an `EquationLaw` whose target is a coordinate of an `EulerLagrangeInteraction`,
which adds to the equation of that coordinate, as `dc_motor` shows.

## The solve

`OrdersPass` solves the equations of a system with algebraic states inside the right side of its differential
equations, so its solvers run unchanged.

- The unknowns are the highest derivatives and the algebraic states.
- A maximum bipartite matching pairs the equations with the unknowns. The strongly connected components of their
  dependency graph, in topological order, are the blocks.
- A block linear in its unknowns is solved symbolically up to six unknowns, and by a numeric linear solve above.
- Any other block runs `scipy.optimize.root` on the NumPy backend, with the Jacobian Sage derives, at every evaluation,
  from the values of the last evaluation. When that fails it starts again one unit away, and then raises `SolverError`.
- The laws that read an algebraic state read its solution. A function argument, an event guard, or a reset that reads
  an algebraic state of a nonlinear block raises `ValidationError`, because these need a symbolic value.

The solve has no index reduction, so the structural check below reports a system of a higher index.

## Issues of connected systems

`PassCompiler` reports these issues with the other issues of a system, before compilation.

- `port-connector`: the ports of one connection have different connectors, or a connection names a port that no single
  element declares. Connect declared ports of one connector.
- `port-connections`: a port appears in more than one connection. List the ports of the node in one connection.
- `port-states`: an element type declares a port without an algebraic state for one of its variables. Declare
  `<port>_<variable>` with `StateSpec(algebraic=True)`.
- `structural-singularity`: the matching leaves equations or unknowns unmatched, as for two capacitors in parallel. The
  issue names the constrained states. Add a compliant element, such as a resistance, between them.
- `algebraic-law`: a law other than an `EquationLaw` targets an algebraic state. State its value with an `EquationLaw`.

A connection under discrete time gives the `law-time` issue of every `EquationLaw`.
