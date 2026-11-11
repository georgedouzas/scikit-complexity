"""Energy harvesters: circuits that turn vibrations into electric energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..common.mechanics import SpringDamperInteraction


class PiezoHarvesterElement(Element):
    """Two-mass suspension of a car whose spring presses a piezoelectric bar.

    The road shakes the suspension. The element holds the tyre between the unsprung mass and the road, and the system
    joins the two masses by a `SpringDamperInteraction` of the stiffness `k2` and the damping `b`.
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y1: Annotated[object, StateSpec(order=2)] = 0.0,
        y1_d1: float = 0.0,
        y2: Annotated[object, StateSpec(order=2)] = 0.0,
        y2_d1: float = 0.0,
        m1: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 40.0,
        m2: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 400.0,
        k1: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 200000.0,
        k2: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20000.0,
        b: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1500.0,
        A: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        omega: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 9.42,
        n: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        d: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 4e-10,
        C: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1e-08,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y1 = y1
        self.y1_d1 = y1_d1
        self.y2 = y2
        self.y2_d1 = y2_d1
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.A = A
        self.omega = omega
        self.n = n
        self.d = d
        self.C = C

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the acceleration of the unsprung mass by the tyre on the road.

        It leaves the suspension apart.
        """
        return [RateLaw(s.y1, -s.k1 * (s.y1 - s.A * sin(s.omega * s.t)) / s.m1)]


def _get_piezo_voltage(s: SimpleNamespace) -> Expression:
    """Return the voltage of the piezoelectric bar, `n d k2 (y2 - y1) / C`."""
    harvester = s.harvester
    return harvester.n * harvester.d * harvester.k2 * (harvester.y2 - harvester.y1) / harvester.C


class VibrationEnergyHarvesterSystem(BaseSystem):
    """A dual-mass piezoelectric harvester in a car suspension.

    The roughness of the road drives it. The unsprung and sprung masses follow
    `m1 y1'' + b (y1' - y2') + k2 (y1 - y2) + k1 (y1 - z(t)) = 0` and `m2 y2'' + b (y2' - y1') + k2 (y2 - y1) = 0`, with
    the road profile `z(t) = A sin(ω t)`. A lever with arm ratio `n` presses a piezoelectric bar of constant `d` and
    capacitance `C`, whose charge is `n d k2 (y2 - y1)` and whose voltage is the observable `voltage`. The damper `b`
    stands for the energy the bar absorbs. The system has the displacements `harvester.y1` and `harvester.y2`, and the
    observable `voltage`.

    Args:
        m1:
            The unsprung mass.

        m2:
            The sprung mass.

        k1:
            The stiffness of the tyre.

        k2:
            The stiffness of the suspension spring.

        b:
            The damping coefficient of the harvester.

        A:
            The amplitude of the road profile.

        omega:
            The angular frequency of the road profile.

        n:
            The ratio of the arms of the lever.

        d:
            The piezoelectric constant of the bar.

        C:
            The capacitance of the bar.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.engineering.circuits import VibrationEnergyHarvesterSystem
        >>> VibrationEnergyHarvesterSystem(b=1000.0).prepare().system_.elements[0].b
        1000.0
    """

    field = 'engineering/circuits'
    topics = ('energy harvesting', 'piezoelectric', 'vibration', 'vehicles', 'ode')
    references = ('Trzaska 2023',)
    until = 10.0
    dt = 0.01
    observables = (ObservableSpec('voltage', _get_piezo_voltage),)

    def __init__(
        self,
        *,
        m1: float = 40.0,
        m2: float = 400.0,
        k1: float = 2.0e5,
        k2: float = 2.0e4,
        b: float = 1500.0,
        A: float = 0.01,
        omega: float = 9.42,
        n: float = 10.0,
        d: float = 4.0e-10,
        C: float = 1.0e-8,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.A = A
        self.omega = omega
        self.n = n
        self.d = d
        self.C = C

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        harvester = PiezoHarvesterElement(
            'harvester',
            m1=self.m1,
            m2=self.m2,
            k1=self.k1,
            k2=self.k2,
            b=self.b,
            A=self.A,
            omega=self.omega,
            n=self.n,
            d=self.d,
            C=self.C,
        )
        suspension = SpringDamperInteraction(
            'suspension',
            target='harvester',
            states=('y1', 'y2'),
            stiffness='k2',
            damping='b',
            divisor={'y1': 'm1', 'y2': 'm2'},
        )
        return CompositeSystem(self.label, elements=[harvester], interactions=[suspension])
