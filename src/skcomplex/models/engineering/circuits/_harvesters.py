"""Energy harvesters: circuits that turn vibrations into electric energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model


def _piezo_voltage(s: SimpleNamespace) -> Expression:
    """Return the voltage of the piezoelectric bar, `n d k2 (y2 - y1) / C`."""
    harvester = s.harvester
    return harvester.n * harvester.d * harvester.k2 * (harvester.y2 - harvester.y1) / harvester.C


class PiezoHarvesterElement(Element):
    """Piezo harvester element."""

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
        """Return the laws of the piezo harvester element."""
        return [
            RateLaw(
                s.y1,
                (
                    -s.b * (derivative(s.y1, 1) - derivative(s.y2, 1))
                    - s.k2 * (s.y1 - s.y2)
                    - s.k1 * (s.y1 - s.A * sin(s.omega * s.t))
                )
                / s.m1,
            ),
            RateLaw(s.y2, (-s.b * (derivative(s.y2, 1) - derivative(s.y1, 1)) - s.k2 * (s.y2 - s.y1)) / s.m2),
        ]


@register_model(
    field='engineering/circuits',
    tags=('energy harvesting', 'piezoelectric', 'vibration', 'vehicles', 'ode'),
    references=('Trzaska 2023',),
    defaults={'until': 10.0, 'dt': 0.01},
    observables=lambda system: [ObservableSpec('voltage', _piezo_voltage)],
)
def vibration_energy_harvester(
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
) -> System:
    """Return a dual-mass piezoelectric harvester in a car suspension, driven by the roughness of the road.

    The unsprung and sprung masses follow `m1 y1'' + b (y1' - y2') + k2 (y1 - y2) + k1 (y1 - z(t)) = 0` and
    `m2 y2'' + b (y2' - y1') + k2 (y2 - y1) = 0`, with the road profile `z(t) = A sin(ω t)`. A lever with arm ratio
    `n` presses a piezoelectric bar of constant `d` and capacitance `C`, whose charge is `n d k2 (y2 - y1)` and whose
    voltage is the observable `voltage`. The damper `b` stands for the energy the bar absorbs.

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

    Returns:
        system:
            The model, with the displacements `harvester.y1` and `harvester.y2`, and the observable `voltage`.

    Examples:
        >>> from skcomplex.models.engineering.circuits import vibration_energy_harvester
        >>> vibration_energy_harvester(b=1000.0).elements[0].b
        1000.0
    """
    harvester = PiezoHarvesterElement('harvester', m1=m1, m2=m2, k1=k1, k2=k2, b=b, A=A, omega=omega, n=n, d=d, C=C)
    return System(elements=[harvester])
