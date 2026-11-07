"""Driven flows: a Rössler flow whose control parameter follows an Ornstein-Uhlenbeck process."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import RateInteraction, RelaxationInteraction
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..common.flows._rossler import RosslerElement


class DrivenRosslerElement(RosslerElement):
    """Rössler flow with a control `eta`, the state that the drive adds to the rate of `y` and that relaxes to 0.

    The element holds the flow and the states and parameters of its control. The interactions of the system add the
    control to the rate of `y` and make it an Ornstein-Uhlenbeck process of correlation time `tau` and noise `rho`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        eta:
            The initial control.

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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import DrivenRosslerElement
        >>> DrivenRosslerElement('rossler', tau=10.0).tau
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: float = 1.0,
        y: float = 1.0,
        z: float = 1.0,
        eta: Annotated[float, StateSpec()] = 0.0,
        a: float = 0.2,
        b: float = 0.4,
        c: float = 5.743,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 50.0,
        rho: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual, x=x, y=y, z=z, a=a, b=b, c=c)
        self.eta = eta
        self.tau = tau
        self.rho = rho


@register_system(
    field='mathematics/dynamical_systems',
    tags=('Rössler', 'Ornstein-Uhlenbeck', 'sde', 'bifurcation diagram', 'chaos', 'nonstationary'),
    references=('Cestnik and Martens 2025', 'Rössler 1976', 'Uhlenbeck and Ornstein 1930'),
    defaults={'until': 500.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
            The system, with the states `rossler.x`, `rossler.y`, `rossler.z`, and the control `rossler.eta`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ou_driven_rossler
        >>> ou_driven_rossler(tau=10.0).elements[0].tau
        10.0
    """
    rossler = DrivenRosslerElement('rossler', x=x, y=y, z=z, eta=eta, a=a, b=b, c=c, tau=tau, rho=rho)
    drive = RateInteraction('drive', target='rossler', state='y', rate=lambda s: s.target.eta)
    control = RelaxationInteraction(
        'control',
        target='rossler',
        state='eta',
        time=lambda s: s.target.tau,
        noise=lambda s: s.target.rho / s.target.tau,
    )
    return System(elements=[rossler], interactions=[drive, control])
