"""Charged particle systems.

The module holds a charge in the magnetic dipole field of a planet.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Mapping, Sequence
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, sqrt

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.mechanics import LagrangianInteraction


class DipoleChargeElement(Element):
    """Charge in the magnetic dipole field of a planet.

    The charge is at the position `(x, y, z)`, and its velocities are `x'`, `y'`, and `z'`.

    The parameter `alpha` is the product of the charge and the dipole moment, divided by the mass.

    Args:
        label:
            The label of the charge.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate, along the dipole.

        alpha:
            The product of the charge and the dipole moment, divided by the mass.

        values:
            Further values by name, such as initial velocities named with a prime, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import DipoleChargeElement
        >>> DipoleChargeElement('charge', alpha=2.0).alpha
        2.0
    """

    visual = VisualSpec('bodies', {'positions': ('x', 'y', 'z'), 'trail': None, 'marks': [(0.0, 0.0, 0.0)]})

    def __init__(
        self,
        label: str,
        *,
        x: float = 1.0,
        y: float = 0.0,
        z: float = 0.0,
        alpha: float = 1.0,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.x = x
        self.y = y
        self.z = z
        self.alpha = alpha


class DipoleLorentzInteraction(LagrangianInteraction):
    """Motion of a charge in the magnetic dipole field of a planet.

    The motion is the Euler-Lagrange equations of `|x'|² / 2 + alpha (x y' - y x') / r³`, the kinetic energy and the
    coupling of the velocity to the vector potential of a dipole along the `z` axis, with the value `alpha` of the
    target. It is a proton trapped in the radiation belts, a cosmic ray deflected by the Earth, or an electron in a
    magnetic bottle.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the charge.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import DipoleLorentzInteraction
        >>> DipoleLorentzInteraction('lorentz', target='charge').target
        'charge'
    """

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _get_coordinates(self, context: SimpleNamespace) -> list[Expression]:
        """Return the coordinates of the charge."""
        return [context.target.x, context.target.y, context.target.z]

    def _get_lagrangian(
        self,
        context: SimpleNamespace,
        positions: list[Expression],
        velocities: list[Expression],
    ) -> Expression:
        """Return the Lagrangian `|x'|² / 2 + alpha (x y' - y x') / r³` of the charge.

        Args:
            context:
                The namespace of the charge and the time `t`.

            positions:
                The symbols of the coordinates.

            velocities:
                The symbols of the velocities.

        Returns:
            lagrangian:
                The Lagrangian.
        """
        x, y, z = positions
        radius = sqrt(x**2 + y**2 + z**2)
        kinetic = sum(item**2 for item in velocities) / 2
        return kinetic + context.target.alpha * (x * velocities[1] - y * velocities[0]) / radius**3


class StormerProblemSystem(BaseSystem):
    """A charged particle moving in the magnetic dipole field of a planet.

    This is the Störmer problem.

    The charge follows the Euler-Lagrange equations of `|x'|² / 2 + alpha (x y' - y x') / r³`, the kinetic energy and
    the coupling of the velocity to the vector potential of the dipole, whose Lorentz force gives `x'' = 3 alpha z (y' z
    - z' y) / r⁵ - alpha y' / r³`, `y'' = -3 alpha z (x' z - z' x) / r⁵ + alpha x' / r³`, and `z'' = 3 alpha z (x' y -
    y' x) / r⁵`, with `alpha = q M / m` for a dipole of moment `M` along the `z` axis. The magnetic force does no work,
    so the speed is conserved, and the symmetry about the axis conserves `x y' - y x' + alpha (x² + y²) / r³`. Bounded
    orbits near the planet trace the radiation belts, and many of them are chaotic. The system has the coordinates
    `charge.x`, `charge.y`, and `charge.z`, and the velocities `charge.x'`, `charge.y'`, and `charge.z'`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import StormerProblemSystem
        >>> StormerProblemSystem(alpha=2.0).prepare().system_.elements[0].alpha
        2.0
    """

    field = 'physics/celestial_mechanics'
    topics = ('mechanics', 'ode', 'second order', 'magnetism', 'chaos', 'hamiltonian')
    references = ('Störmer 1955', 'Dilão 2023')
    until = 20.0
    dt = 0.02

    def __init__(
        self,
        x: float = 1.0,
        y: float = 0.0,
        z: float = 0.0,
        *,
        vx: float = 0.0,
        vy: float = 0.1,
        vz: float = 0.1,
        alpha: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.alpha = alpha

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a charged particle moving in the magnetic dipole field of a planet.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        charge = DipoleChargeElement(
            'charge',
            x=self.x,
            y=self.y,
            z=self.z,
            alpha=self.alpha,
            values={"x'": self.vx, "y'": self.vy, "z'": self.vz},
        )
        self.system_ = System(
            self.label,
            elements=[charge],
            interactions=[DipoleLorentzInteraction('lorentz', target='charge')],
        )
