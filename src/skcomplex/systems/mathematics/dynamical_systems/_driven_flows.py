"""Driven flows.

The control parameter of a Rössler flow follows an Ornstein-Uhlenbeck process.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ..common.flows._rossler import RosslerInteraction


class OUDriveInteraction(Interaction):
    """The drive of a flow by a control `eta` that follows an Ornstein-Uhlenbeck process.

    The control adds to the rate of `y` and follows `eta' = (-eta + rho xi) / tau` with white noise `xi`, so it relaxes
    to 0 over the correlation time `tau`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose coordinate `y` the control drives and that holds the control `eta`.

        tau:
            The correlation time of the control.

        rho:
            The amplitude of the noise of the control.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import OUDriveInteraction
        >>> OUDriveInteraction('control', target='rossler', tau=10.0).tau
        10.0
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(self, label: str, *, target: Role = 'rossler', tau: float = 50.0, rho: float = 0.5) -> None:
        super().__init__(label)
        self.target = target
        self.tau = tau
        self.rho = rho

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drive of `y` and the rate and noise of the control as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            y, eta = context.target.y, context.target.eta
            self.laws_: list[BaseComponent] = [
                RateLaw(y, eta),
                RateLaw(eta, -eta / context.tau),
                NoiseLaw(eta, context.rho / context.tau),
            ]


class OUDrivenRosslerSystem(BaseSystem):
    """The Rössler flow driven through its bifurcations.

    A slow Ornstein-Uhlenbeck control parameter drives the flow.

    The flow is `x' = -y - z`, `y' = x + a y + eta`, `z' = b + z (x - c)`, Eq. 13 of Cestnik and Martens, and the
    control follows `eta' = (-eta + rho xi) / tau` with white noise `xi`. The control is an Ornstein-Uhlenbeck process
    with the correlation time `tau` and the stationary variance `rho^2 / (2 tau)`. A control that varies slowly against
    the period of the flow sweeps it through the period doublings and chaotic bands of its bifurcation diagram. The
    system has the states `rossler.x`, `rossler.y`, `rossler.z`, and the control `rossler.eta`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import OUDrivenRosslerSystem
        >>> OUDrivenRosslerSystem(tau=10.0).prepare().system_.interactions[1].tau
        10.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('Rössler', 'Ornstein-Uhlenbeck', 'sde', 'bifurcation diagram', 'chaos', 'nonstationary')
    references = ('Cestnik and Martens 2025', 'Rössler 1976', 'Uhlenbeck and Ornstein 1930')
    until = 500.0
    dt = 0.1

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.a = a
        self.b = b
        self.c = c
        self.tau = tau
        self.rho = rho
        self.eta = eta

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its drive."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('rossler', values={'x': self.x, 'y': self.y, 'z': self.z, 'eta': self.eta})],
            interactions=[
                RosslerInteraction('flow', target='rossler', a=self.a, b=self.b, c=self.c),
                OUDriveInteraction('control', target='rossler', tau=self.tau, rho=self.rho),
            ],
            solver=EulerMaruyamaSolver(dt=0.01),
        )
