"""Charged particle models: a charge in the magnetic dipole field of a planet."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, sqrt

from ....core.elements import Element
from ....core.interactions import EulerLagrangeInteraction
from ....core.laws.utils import derivative
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


def _get_dipole_lagrangian(s: SimpleNamespace) -> Expression:
    """Return the Lagrangian `|x'|² / 2 + alpha (x y' - y x') / r³` of a charge in a magnetic dipole field."""
    charge = s.target
    velocity = [derivative(item, 1) for item in (charge.x, charge.y, charge.z)]
    radius = sqrt(charge.x**2 + charge.y**2 + charge.z**2)
    kinetic = sum(item**2 for item in velocity) / 2
    return kinetic + charge.alpha * (charge.x * velocity[1] - charge.y * velocity[0]) / radius**3


class DipoleChargeElement(Element):
    """Charge in the magnetic dipole field of a planet, at the position `(x, y, z)` with coordinates of order 2.

    The parameter `alpha` is the product of the charge and the dipole moment, divided by the mass.

    Args:
        label:
            The label of the charge.

        visual:
            How the charge is drawn, or `None` for its class visual.

        x:
            The initial first coordinate.

        x_d1:
            The initial first velocity.

        y:
            The initial second coordinate.

        y_d1:
            The initial second velocity.

        z:
            The initial third coordinate, along the dipole.

        z_d1:
            The initial third velocity.

        alpha:
            The product of the charge and the dipole moment, divided by the mass.

    Examples:
        >>> from skcomplex.models.physics.celestial_mechanics import DipoleChargeElement
        >>> DipoleChargeElement('charge', alpha=2.0).alpha
        2.0
    """

    visual = VisualSpec('bodies', {'positions': ('x', 'y', 'z'), 'trail': None, 'marks': [(0.0, 0.0, 0.0)]})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=2)] = 1.0,
        x_d1: float = 0.0,
        y: Annotated[float, StateSpec(order=2)] = 0.0,
        y_d1: float = 0.0,
        z: Annotated[float, StateSpec(order=2)] = 0.0,
        z_d1: float = 0.0,
        alpha: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.x_d1 = x_d1
        self.y = y
        self.y_d1 = y_d1
        self.z = z
        self.z_d1 = z_d1
        self.alpha = alpha


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

    The charge follows the Euler-Lagrange equations of `|x'|² / 2 + alpha (x y' - y x') / r³`, the kinetic energy and
    the coupling of the velocity to the vector potential of the dipole, whose Lorentz force gives
    `x'' = 3 alpha z (y' z - z' y) / r⁵ - alpha y' / r³`, `y'' = -3 alpha z (x' z - z' x) / r⁵ + alpha x' / r³`,
    and `z'' = 3 alpha z (x' y - y' x) / r⁵`, with `alpha = q M / m` for a dipole of moment `M` along the `z` axis.
    The magnetic force does no work, so the speed is conserved, and the symmetry about the axis conserves
    `x y' - y x' + alpha (x² + y²) / r³`. Bounded orbits near the planet trace the radiation belts, and many of them are
    chaotic.

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
    return System(
        elements=[DipoleChargeElement('charge', x=x, y=y, z=z, x_d1=vx, y_d1=vy, z_d1=vz, alpha=alpha)],
        interactions=[
            EulerLagrangeInteraction(
                'lorentz',
                target='charge',
                states=('x', 'y', 'z'),
                lagrangian=_get_dipole_lagrangian,
            ),
        ],
    )
