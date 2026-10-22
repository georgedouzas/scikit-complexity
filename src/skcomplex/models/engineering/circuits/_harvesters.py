"""Energy harvesters: circuits that turn vibrations into electric energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import sin

from ....core.base import Domain, Expression, parameter, state
from ....core.components import Element
from ....core.laws import Rate, Symbols, derivative
from ....core.runtime import register_model
from ....core.system import Observable, System


def _piezo_voltage(s: Symbols) -> Expression:
    """Return the voltage of the piezoelectric bar, `n d k2 (y2 - y1) / C`."""
    harvester = s.harvester
    return harvester.n * harvester.d * harvester.k2 * (harvester.y2 - harvester.y1) / harvester.C


PiezoHarvester = Element.define(
    'PiezoHarvester',
    states={'y1': state(default=0.0, order=2), 'y2': state(default=0.0, order=2)},
    parameters={
        'm1': parameter(default=40.0, domain=Domain.POSITIVE),
        'm2': parameter(default=400.0, domain=Domain.POSITIVE),
        'k1': parameter(default=2.0e5, domain=Domain.NONNEGATIVE),
        'k2': parameter(default=2.0e4, domain=Domain.NONNEGATIVE),
        'b': parameter(default=1500.0, domain=Domain.NONNEGATIVE),
        'A': parameter(default=0.01, domain=Domain.NONNEGATIVE),
        'omega': parameter(default=9.42, domain=Domain.NONNEGATIVE),
        'n': parameter(default=10.0, domain=Domain.POSITIVE),
        'd': parameter(default=4.0e-10, domain=Domain.POSITIVE),
        'C': parameter(default=1.0e-8, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Rate(
            s.y1,
            (
                -s.b * (derivative(s.y1, 1) - derivative(s.y2, 1))
                - s.k2 * (s.y1 - s.y2)
                - s.k1 * (s.y1 - s.A * sin(s.omega * s.t))
            )
            / s.m1,
        ),
        Rate(s.y2, (-s.b * (derivative(s.y2, 1) - derivative(s.y1, 1)) - s.k2 * (s.y2 - s.y1)) / s.m2),
    ],
)


@register_model(
    field='engineering/circuits',
    tags=('energy harvesting', 'piezoelectric', 'vibration', 'vehicles', 'ode'),
    references=('Trzaska 2023',),
    defaults={'until': 10.0, 'dt': 0.01},
    observables=lambda system: [Observable('voltage', _piezo_voltage)],
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
    harvester = PiezoHarvester('harvester', m1=m1, m2=m2, k1=k1, k2=k2, b=b, A=A, omega=omega, n=n, d=d, C=C)
    return System(components=[harvester])
