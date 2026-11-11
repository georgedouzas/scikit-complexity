# Ports and connections

This page assembles a system from components that connect through ports, the way engineers draw circuits and
plants. A port is the point where a component meets its neighbours, such as the pin of a resistor. Each component
states equations and leaves open which variable each equation determines, so one resistor gives a current under a
voltage source and a voltage under a current source.

Here two reservoirs of a chemical share one node. Each is an [`Element`](../core_components/elements.md#element) whose
[`EquationLaw`](../core_components/laws.md#equationlaw) ties its flow to the gap between its potential and its level,
and a [`ConnectionInteraction`](../core_components/interactions.md#connectioninteraction) joins them in a
[`CompositeSystem`](../core_components/systems.md#compositesystem).

```python
from typing import Annotated, ClassVar

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
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


system = CompositeSystem(
    elements=[ReservoirElement('first', level=1.0), ReservoirElement('second', level=3.0)],
    interactions=[ConnectionInteraction('node', members=['first', 'second'], ports=['a', 'a'])],
)
trajectory = simulate(system, until=1.0, dt=0.5)
assert trajectory.get_values('first.a_mu')[-1] == 2.0
assert trajectory.get_values('second.a_n')[-1] == -1.0
trajectory.get_values('first.a_mu')[-1], trajectory.get_values('second.a_n')[-1]
```

The block returns `(2.0, -1.0)`. The node makes the two potentials equal and the two flows sum to zero, so the
potential settles at 2.0, halfway between the levels 1.0 and 3.0. A flow of 1.0 leaves the second reservoir and enters
the first.

## Algebraic states

This section shows how to declare a state that the equations fix at every time instead of a law that integrates
it. `StateSpec(algebraic=True)` declares such an algebraic state, as the potentials and flows of the reservoirs above.

- An algebraic state has order 1 and no lags.
- Its initial value is the starting guess of a nonlinear solve.
- A law other than an `EquationLaw` that targets it raises `AlgebraicStateError`.
- The trajectory records it at every frame, so observables and visuals read it like any other state.
- The solvers integrate only the other states, so the state vector they advance keeps it at its initial value.
- A law given as a function that reads values by name sees that initial value. A symbolic law reads the solved value.

## Ports and connections

This section declares the ports of an element and joins them at nodes. A connector names the two variables of a
port: a potential, which is equal at a node, such as a voltage, and a flow, which sums to zero there, such as a
current. `ConnectorSpec(name, potential, flow)` defines one, and the core holds no named connector of its own. An
element class declares its ports in the class variable `ports`, a mapping from the name of a port to its connector. It
takes one algebraic state per variable of a port, named `<port>_<variable>`. A flow is positive into the element.

`ConnectionInteraction(label, members=..., ports=...)` is one node.

- It binds the elements of the node through a variadic role, and names the port of each in the same order.
- It returns the equations that make the potentials equal and the flows sum to zero.
- A node with many ports is one connection.
- A port that no connection names has a zero flow.

A circuit takes one connector and one element class per component. Here a capacitor of 0.5 farad, charged to 1
volt, discharges through a resistor of 2 ohm.

```python
from typing import Annotated, ClassVar

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
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


circuit = CompositeSystem(
    elements=[CapacitorElement('tank', u=1.0, C=0.5), ResistorElement('load', R=2.0), GroundElement('ground')],
    interactions=[
        ConnectionInteraction('top', members=['tank', 'load'], ports=['p', 'p']),
        ConnectionInteraction('bottom', members=['tank', 'load', 'ground'], ports=['n', 'n', 'p']),
    ],
)
trajectory = simulate(circuit, until=1.0, dt=0.1)
assert np.isclose(trajectory.get_values('tank.u')[-1], np.exp(-1.0), rtol=1e-3)
assert np.allclose(trajectory.get_values('load.p_i').ravel(), trajectory.get_values('tank.u').ravel() / 2.0)
trajectory.get_values('tank.u')[-1].round(4)
```

The voltage of the capacitor is 0.3679 after one second. It decays as `exp(-t / RC)` with `RC = 1`, and the current
through the load is that voltage over `R`. The ready-made systems of `skcomplex.systems.engineering.circuits` build
their electrical and thermal components in the same way. A port component also drives a mechanism. Its
`EquationLaw` then targets a coordinate of an
[`EulerLagrangeInteraction`](../core_components/interactions.md#eulerlagrangeinteraction) and adds to the equation of
that coordinate, as `DcMotorSystem` of `skcomplex.systems.engineering.mechatronics` shows.

## The solve

This section explains how the engine finds the algebraic states at each step. The compiler solves the equations of
the system inside the right side of its differential equations, so the solvers run unchanged.

- The unknowns are the highest derivatives and the algebraic states.
- A maximum bipartite matching pairs the equations with the unknowns. The strongly connected components of their
  dependency graph, in topological order, are the blocks.
- A block linear in its unknowns is solved symbolically up to six unknowns, and by a numeric linear solve above.
- Any other block runs `scipy.optimize.root` on the NumPy backend, with the Jacobian Sage derives, at every evaluation,
  from the values of the last evaluation. When that fails it starts again one unit away, and then raises `SolverError`.
- The laws that read an algebraic state read its solution. A function argument, an event guard, or a reset that reads
  an algebraic state of a nonlinear block raises `ValidationError`, because these need a symbolic value.

The index of a system counts how often its equations must be differentiated before they fix every unknown. The
solve has no index reduction, so the structural check below reports a system of a higher index.

## Issues of connected systems

This section lists the faults of a connected system and their fixes. `simulate` reports these issues with the other
issues of a system, before it compiles anything.

- `port-connector`: the ports of one connection have different connectors, or a connection names a port that no single
  element declares. Connect declared ports of one connector.
- `port-connections`: a port appears in more than one connection. List the ports of the node in one connection.
- `port-states`: an element type declares a port without an algebraic state for one of its variables. Declare
  `<port>_<variable>` with `StateSpec(algebraic=True)`.
- `structural-singularity`: the matching leaves equations or unknowns unmatched, as for two capacitors in parallel. The
  issue names the constrained states. Add a compliant element, such as a resistance, between them.
- `algebraic-law`: a law other than an `EquationLaw` targets an algebraic state. State its value with an `EquationLaw`.

A connection under discrete time gives the `law-time` issue of every `EquationLaw`, because an equation law requires
continuous or static time.
