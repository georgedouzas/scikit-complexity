"""Driven flows: a Rössler flow whose control parameter follows an Ornstein-Uhlenbeck process."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter
from ....core.components import Element
from ....core.laws import Noise, Rate
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama
from ....core.system import System

DrivenRossler = Element.define(
    'DrivenRossler',
    states={'x': 1.0, 'y': 1.0, 'z': 1.0, 'eta': 0.0},
    parameters={
        'a': 0.2,
        'b': 0.4,
        'c': 5.743,
        'tau': parameter(default=50.0, domain=Domain.POSITIVE),
        'rho': parameter(default=0.5, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(s.x, -s.y - s.z),
        Rate(s.y, s.x + s.a * s.y + s.eta),
        Rate(s.z, s.b + s.z * (s.x - s.c)),
        Rate(s.eta, -s.eta / s.tau),
        Noise(s.eta, s.rho / s.tau),
    ],
)


@register_model(
    field='mathematics/dynamical_systems',
    tags=('Rössler', 'Ornstein-Uhlenbeck', 'sde', 'bifurcation diagram', 'chaos', 'nonstationary'),
    references=('Cestnik and Martens 2025', 'Rössler 1976', 'Uhlenbeck and Ornstein 1930'),
    defaults={'until': 500.0, 'dt': 0.1, 'solver': EulerMaruyama(dt=0.01)},
)
def ou_driven_rossler(
    x: float = 1.0,
    y: float = 1.0,
    z: float = 1.0,
    *,
    a: float = 0.2,
    b: float = 0.4,
    c: float = 5.743,
    tau: float = 50.0,
    rho: float = 0.5,
    eta: float = 0.0,
) -> System:
    """Return the Rössler flow driven through its bifurcations by a slow Ornstein-Uhlenbeck control parameter.

    The flow is `x' = -y - z`, `y' = x + a y + eta`, `z' = b + z (x - c)`, Eq. 13 of Cestnik and Martens, and the
    control follows `eta' = (-eta + rho xi) / tau` with white noise `xi`. The control is an Ornstein-Uhlenbeck
    process with the correlation time `tau` and the stationary variance `rho^2 / (2 tau)`. A control that varies
    slowly against the period of the flow sweeps it through the period doublings and chaotic bands of its
    bifurcation diagram.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The threshold of the third coordinate's growth.

        tau:
            The correlation time of the control.

        rho:
            The amplitude of the noise of the control.

        eta:
            The initial control.

    Returns:
        system:
            The model, with the states `rossler.x`, `rossler.y`, `rossler.z`, and the control `rossler.eta`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import ou_driven_rossler
        >>> ou_driven_rossler(tau=10.0).elements[0].tau
        10.0
    """
    rossler = DrivenRossler('rossler', x=x, y=y, z=z, eta=eta, a=a, b=b, c=c, tau=tau, rho=rho)
    return System(components=[rossler])
