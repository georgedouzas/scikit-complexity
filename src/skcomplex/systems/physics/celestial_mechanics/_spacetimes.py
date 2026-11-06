"""Spacetimes of a central mass: the Schwarzschild metric of a static one and the Kerr metric of a spinning one."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import QQ, SR, Expression, cos, matrix, sin
from sage.structure.element import Matrix

from ....core.components.spaces import PseudoRiemannianSpace
from ....core.errors import ValidationError


class SchwarzschildSpace(PseudoRiemannianSpace):
    """Spacetime outside a static, spherical mass `M`, the Schwarzschild metric in units with `G = c = 1`.

    The line element in the coordinates `(t, r, theta, phi)` is
    `ds² = -(1 - 2M/r) dt² + dr² / (1 - 2M/r) + r² (dθ² + sin²θ dφ²)`, with the horizon at `r = 2M`. The mass enters
    the metric as an exact rational, which keeps the symbolic derivation fast.

    Args:
        mass:
            The mass `M`.

        curvature:
            Whether `prepare` derives the curvature tensors besides the Christoffel symbols.

        label:
            The label of the space in its system.

    Raises:
        ValidationError:
            The mass is not positive.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import SchwarzschildSpace
        >>> SchwarzschildSpace(mass=1.0, curvature=False).prepare().christoffel_[1][0][0]
        (r - 2)/r^3
    """

    def __init__(self, mass: float = 1.0, *, curvature: bool = False, label: str = 'space') -> None:
        super().__init__(curvature=curvature, label=label)
        self.mass = mass

    def _get_coordinates(self) -> tuple[Expression, ...]:
        """Return the coordinates `(t, r, theta, phi)`."""
        return tuple(SR.var(name) for name in ('t', 'r', 'theta', 'phi'))

    def _get_metric(self) -> Matrix:
        """Return the Schwarzschild metric of the mass.

        Returns:
            metric:
                The metric in the coordinates `(t, r, theta, phi)`.

        Raises:
            ValidationError:
                The mass is not positive.
        """
        if not self.mass > 0:
            error_msg = f'The mass of {type(self).__name__} is {self.mass!r}. Give a positive mass.'
            raise ValidationError(error_msg)
        _, r, theta, _ = self._get_coordinates()
        lapse = 1 - 2 * QQ(self.mass) / r
        return matrix(SR, 4, 4, {(0, 0): -lapse, (1, 1): 1 / lapse, (2, 2): r**2, (3, 3): r**2 * sin(theta) ** 2})


class KerrSpace(PseudoRiemannianSpace):
    """Spacetime outside a spinning mass `M` of angular momentum `M a`, the Kerr metric in Boyer-Lindquist coordinates.

    With `Σ = r² + a² cos²θ` and `Δ = r² - 2 M r + a²`, the line element in `(t, r, theta, phi)` is
    `ds² = -(1 - 2Mr/Σ) dt² - (4 M a r sin²θ / Σ) dt dφ + Σ/Δ dr² + Σ dθ² + (r² + a² + 2 M a² r sin²θ / Σ) sin²θ dφ²`,
    in units with `G = c = 1`. The term in `dt dφ` drags the frames of free particles around the spin axis. The mass and
    the spin enter the metric as exact rationals, which keeps the symbolic derivation fast.

    Args:
        mass:
            The mass `M`.

        spin:
            The spin parameter `a`, the angular momentum per unit mass, between `-M` and `M`.

        curvature:
            Whether `prepare` derives the curvature tensors besides the Christoffel symbols, which takes long.

        label:
            The label of the space in its system.

    Raises:
        ValidationError:
            The mass is not positive, or the spin exceeds the mass.

    Examples:
        >>> from skcomplex.systems.physics.celestial_mechanics import KerrSpace
        >>> KerrSpace(mass=1.0, spin=0.5).prepare().metric_[0, 3]
        -4*r*sin(theta)^2/(4*r^2 + cos(theta)^2)
    """

    def __init__(
        self,
        mass: float = 1.0,
        spin: float = 0.5,
        *,
        curvature: bool = False,
        label: str = 'space',
    ) -> None:
        super().__init__(curvature=curvature, label=label)
        self.mass = mass
        self.spin = spin

    def _get_coordinates(self) -> tuple[Expression, ...]:
        """Return the coordinates `(t, r, theta, phi)`."""
        return tuple(SR.var(name) for name in ('t', 'r', 'theta', 'phi'))

    def _get_metric(self) -> Matrix:
        """Return the Kerr metric of the mass and the spin.

        Returns:
            metric:
                The metric in the coordinates `(t, r, theta, phi)`.

        Raises:
            ValidationError:
                The mass is not positive, or the spin exceeds the mass.
        """
        if not self.mass > 0 or not abs(self.spin) <= self.mass:
            error_msg = (
                f'{type(self).__name__} has the mass {self.mass!r} and the spin {self.spin!r}. Give a positive mass '
                'and a spin no larger than it.'
            )
            raise ValidationError(error_msg)
        _, r, theta, _ = self._get_coordinates()
        mass, spin = QQ(self.mass), QQ(self.spin)
        sigma = r**2 + spin**2 * cos(theta) ** 2
        delta = r**2 - 2 * mass * r + spin**2
        drag = -2 * mass * spin * r * sin(theta) ** 2 / sigma
        return matrix(
            SR,
            4,
            4,
            {
                (0, 0): -(1 - 2 * mass * r / sigma),
                (0, 3): drag,
                (3, 0): drag,
                (1, 1): sigma / delta,
                (2, 2): sigma,
                (3, 3): (r**2 + spin**2 + 2 * mass * spin**2 * r * sin(theta) ** 2 / sigma) * sin(theta) ** 2,
            },
        )
