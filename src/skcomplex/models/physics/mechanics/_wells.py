"""Potential well models: a particle in the one-dimensional well `k x² e^-x`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, Expression, parameter, role
from ....core.model import EuclideanSpace, Particle, Potential, Symbols, System
from ....core.runtime import register_model


class ExponentialWell(Potential):
    """Potential `k x² e^-x` on a particle in one dimension, a well at `x = 0` behind a barrier at `x = 2`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        k:
            The strength of the potential.

    Examples:
        >>> from skcomplex.models.physics.mechanics import ExponentialWell
        >>> ExponentialWell('well', target='particle', k=2.0).k
        2.0
    """

    target: str = role()
    k: float = parameter(default=2.0, domain=Domain.POSITIVE)

    def energy(self, s: Symbols) -> Expression:
        """Return `k x² e^-x`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        position = s.target.x[0]
        return s.k * position**2 * exp(-position)


@register_model(
    field='physics/mechanics',
    tags=('mechanics', 'ode', 'potential', 'hamiltonian'),
    references=('Starkovich 2021',),
    defaults={'until': 20.0, 'dt': 0.05},
)
def particle_in_a_well(x: float = 1.0, v: float = 0.5, *, k: float = 2.0, m: float = 1.0) -> System:
    """Return a particle moving without friction in the potential `U(x) = k x² e^-x`.

    The potential has a stable equilibrium at `x = 0` and an unstable one at `x = 2`, the top of a barrier of height
    `4 k e^-2`. A particle in the well with less energy than the barrier oscillates, and one with more escapes to
    large `x`.

    Args:
        x:
            The initial position.

        v:
            The initial velocity.

        k:
            The strength of the potential.

        m:
            The mass of the particle.

    Returns:
        system:
            The model, with the particle `particle` and its position `particle.x` in one dimension.

    Examples:
        >>> from skcomplex.models.physics.mechanics import particle_in_a_well
        >>> particle_in_a_well(k=3.0).interactions[0].k
        3.0
    """
    return System(
        components=[Particle('particle', x=x, v=v, m=m)],
        interactions=[ExponentialWell('well', target='particle', k=k)],
        space=EuclideanSpace(1),
    )
