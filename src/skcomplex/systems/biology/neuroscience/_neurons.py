"""Neuroscience systems: spiking neurons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.excitable import SpikeResetInteraction

PEAK = 30.0


class IzhikevichElement(Element):
    """Izhikevich neuron, a quadratic potential `v` and a linear recovery `u`.

    Args:
        label:
            The label of the neuron.

        v:
            The initial potential.

        u:
            The initial recovery.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import IzhikevichElement
        >>> IzhikevichElement('neuron').u
        -13.0
    """

    def __init__(self, label: str, *, v: float = -65.0, u: float = -13.0) -> None:
        super().__init__(label)
        self.v = v
        self.u = u


class IzhikevichInteraction(Interaction):
    """Izhikevich's quadratic potential and linear recovery of a neuron.

    The potential follows `v' = 0.04 v² + 5 v + 140 - u + I` and the recovery `u' = a (b v - u)`. A spike reset fires
    the neuron.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the neuron, with the potential `v` and the recovery `u`.

        a:
            The time scale of the recovery.

        b:
            The sensitivity of the recovery to the potential.

        I:
            The input current.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import IzhikevichInteraction
        >>> IzhikevichInteraction('membrane', target='neuron').b
        0.2
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, a: float = 0.02, b: float = 0.2, I: float = 10.0) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b
        self.I = I

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the quadratic potential and the linear recovery.

        Args:
            context:
                The namespace of the neuron, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s = context
        v, u = s.target.v, s.target.u
        self.laws_ = [RateLaw(v, 0.04 * v**2 + 5 * v + 140 - u + s.I), RateLaw(u, s.a * (s.b * v - u))]


class LeakyIntegrateAndFireElement(Element):
    """Leaky integrate-and-fire neuron with the potential `v`.

    Args:
        label:
            The label of the neuron.

        v:
            The initial potential.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import LeakyIntegrateAndFireElement
        >>> LeakyIntegrateAndFireElement('neuron').v
        -65.0
    """

    def __init__(self, label: str, *, v: float = -65.0) -> None:
        super().__init__(label)
        self.v = v


class LeakyIntegrateAndFireInteraction(Interaction):
    """Leak of the potential of a neuron towards `rest + resistance I` over the time `tau`.

    The potential follows `tau v' = -(v - rest) + resistance I`. A spike reset fires the neuron. Units are mV, ms, MΩ,
    and nA.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the neuron, with the potential `v`.

        rest:
            The resting potential.

        resistance:
            The membrane resistance.

        tau:
            The membrane time constant.

        I:
            The input current.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import LeakyIntegrateAndFireInteraction
        >>> LeakyIntegrateAndFireInteraction('membrane', target='neuron').tau
        10.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        rest: float = -65.0,
        resistance: Annotated[float, RealSet.open(0, oo)] = 10.0,
        tau: Annotated[float, RealSet.open(0, oo)] = 10.0,
        I: float = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.rest = rest
        self.resistance = resistance
        self.tau = tau
        self.I = I

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the leaky integration of the input current.

        Args:
            context:
                The namespace of the neuron, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s = context
        v = s.target.v
        self.laws_ = [RateLaw(v, (-(v - s.rest) + s.resistance * s.I) / s.tau)]


class IntegrateAndFireSystem(BaseSystem):
    """The leaky integrate-and-fire neuron, `tau v' = -(v - rest) + R I`.

    It spikes and resets at a threshold.

    With `R I` above `threshold - rest` it fires periodically with period `tau ln(R I / (R I - threshold + rest))` when
    `reset = rest`. The system has the membrane potential `neuron.v` and its spikes recorded as `spike` events.

    Args:
        v:
            The initial membrane potential in mV.

        rest:
            The resting potential in mV.

        threshold:
            The potential at which the neuron spikes in mV.

        reset:
            The potential after a spike in mV.

        resistance:
            The membrane resistance in MΩ.

        tau:
            The membrane time constant in ms.

        I:
            The input current in nA.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import IntegrateAndFireSystem
        >>> trajectory = IntegrateAndFireSystem().prepare().simulate(until=50.0, dt=0.1)
        >>> round(trajectory.events[0].time, 2)
        13.86
    """

    field = 'biology/neuroscience'
    topics = ('neuron', 'spiking', 'events')
    references = ('Lapicque 1907', 'Gerstner et al. 2014')
    until = 100.0
    dt = 0.1

    def __init__(
        self,
        v: float = -65.0,
        *,
        rest: float = -65.0,
        threshold: float = -50.0,
        reset: float = -65.0,
        resistance: float = 10.0,
        tau: float = 10.0,
        I: float = 2.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.v = v
        self.rest = rest
        self.threshold = threshold
        self.reset = reset
        self.resistance = resistance
        self.tau = tau
        self.I = I

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the leaky integrate-and-fire neuron.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        neuron = LeakyIntegrateAndFireElement('neuron', v=self.v)
        membrane = LeakyIntegrateAndFireInteraction(
            'membrane',
            target='neuron',
            rest=self.rest,
            resistance=self.resistance,
            tau=self.tau,
            I=self.I,
        )
        spike = SpikeResetInteraction('spike', target='neuron', threshold=self.threshold, after=self.reset)
        self.system_ = System(self.label, elements=[neuron], interactions=[membrane, spike])


class IzhikevichSystem(BaseSystem):
    """The Izhikevich neuron.

    Its four parameters give regular spiking, bursting, chattering, and more.

    The potential follows `v' = 0.04 v² + 5 v + 140 - u + I` and the recovery `u' = a (b v - u)`. When `v` reaches 30 mV
    the neuron spikes, `v` resets to `c`, and `u` grows by `d`. The defaults give regular spiking. The system has the
    potential `neuron.v`, the recovery `neuron.u`, and its spikes as `spike` events.

    Args:
        v:
            The initial membrane potential in mV.

        u:
            The initial recovery variable, or `None` for `b v`.

        a:
            The time scale of the recovery.

        b:
            The sensitivity of the recovery to the potential.

        c:
            The potential after a spike.

        d:
            The increase of the recovery after a spike.

        I:
            The input current.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import IzhikevichSystem
        >>> IzhikevichSystem(d=2.0).prepare().system_.interactions[1].kick
        2.0
    """

    field = 'biology/neuroscience'
    topics = ('neuron', 'spiking', 'events', 'bursting')
    references = ('Izhikevich 2003',)
    until = 200.0
    dt = 0.1

    def __init__(
        self,
        v: float = -65.0,
        u: float | None = None,
        *,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        I: float = 10.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.v = v
        self.u = u
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.I = I

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Izhikevich neuron.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        neuron = IzhikevichElement('neuron', v=self.v, u=self.b * self.v if self.u is None else self.u)
        membrane = IzhikevichInteraction('membrane', target='neuron', a=self.a, b=self.b, I=self.I)
        spike = SpikeResetInteraction('spike', target='neuron', threshold=PEAK, after=self.c, kicked='u', kick=self.d)
        self.system_ = System(self.label, elements=[neuron], interactions=[membrane, spike])
