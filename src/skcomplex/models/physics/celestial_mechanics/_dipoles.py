"""Charged particle models: a charge in the magnetic dipole field of a planet."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import derivative
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model


class DipoleChargeElement(Element):
    """Dipole charge element."""

    visual = VisualSpec('bodies', {'positions': ('x', 'y', 'z'), 'trail': None, 'marks': [(0.0, 0.0, 0.0)]})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
        y: Annotated[object, StateSpec(order=2)] = 0.0,
        y_d1: float = 0.0,
        z: Annotated[object, StateSpec(order=2)] = 0.0,
        z_d1: float = 0.0,
        alpha: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.y = y
        self.y_d1 = y_d1
        self.z = z
        self.z_d1 = z_d1
        self.alpha = alpha

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the dipole charge element."""
        return [
            RateLaw(
                s.x,
                3
                * s.alpha
                * s.z
                / (s.x**2 + s.y**2 + s.z**2) ** (5 / 2)
                * (derivative(s.y, 1) * s.z - derivative(s.z, 1) * s.y)
                - s.alpha * derivative(s.y, 1) / (s.x**2 + s.y**2 + s.z**2) ** (3 / 2),
            ),
            RateLaw(
                s.y,
                -3
                * s.alpha
                * s.z
                / (s.x**2 + s.y**2 + s.z**2) ** (5 / 2)
                * (derivative(s.x, 1) * s.z - derivative(s.z, 1) * s.x)
                + s.alpha * derivative(s.x, 1) / (s.x**2 + s.y**2 + s.z**2) ** (3 / 2),
            ),
            RateLaw(
                s.z,
                3
                * s.alpha
                * s.z
                / (s.x**2 + s.y**2 + s.z**2) ** (5 / 2)
                * (derivative(s.x, 1) * s.y - derivative(s.y, 1) * s.x),
            ),
        ]


@register_model(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'second order', 'magnetism', 'chaos', 'hamiltonian'),
    references=('Störmer 1955', 'Dilão 2023'),
    defaults={'until': 20.0, 'dt': 0.02},
)
def stormer_problem(
    x: float = 1.0,
    y: float = 0.0,
    z: float = 0.0,
    *,
    vx: float = 0.0,
    vy: float = 0.1,
    vz: float = 0.1,
    alpha: float = 1.0,
) -> System:
    """Return a charged particle moving in the magnetic dipole field of a planet, the Störmer problem.

    The Lorentz force of the dipole gives `x'' = 3 alpha z (y' z - z' y) / r⁵ - alpha y' / r³`,
    `y'' = -3 alpha z (x' z - z' x) / r⁵ + alpha x' / r³`, and `z'' = 3 alpha z (x' y - y' x) / r⁵`, with
    `alpha = q M / m` for a dipole of moment `M` along the `z` axis. The magnetic force does no work, so the speed is
    conserved, and the symmetry about the axis conserves `x y' - y x' + alpha (x² + y²) / r³`. Bounded orbits near the
    planet trace the radiation belts, and many of them are chaotic.

    Args:
        x:
            The initial first coordinate, in units of the planet's radius.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate, along the dipole.

        vx:
            The initial first velocity.

        vy:
            The initial second velocity.

        vz:
            The initial third velocity.

        alpha:
            The product of the charge and the dipole moment, divided by the mass.

    Returns:
        system:
            The model, with the coordinates `charge.x`, `charge.y`, and `charge.z`, all of order 2.

    Examples:
        >>> from skcomplex.models.physics.celestial_mechanics import stormer_problem
        >>> stormer_problem(alpha=2.0).elements[0].alpha
        2.0
    """
    return System(elements=[DipoleChargeElement('charge', x=x, y=y, z=z, x_d1=vx, y_d1=vy, z_d1=vz, alpha=alpha)])
