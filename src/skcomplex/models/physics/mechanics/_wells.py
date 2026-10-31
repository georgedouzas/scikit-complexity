"""Potential well models: a particle in the one-dimensional well `k x² e^-x`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.interactions import PotentialInteraction
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec
from ..._catalog import register_model
from ..common.particles import MassPointElement


class ExponentialWellInteraction(PotentialInteraction):
    """Potential `k x² e^-x` on a particle in one dimension, a well at `x = 0` behind a barrier at `x = 2`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        k:
            The strength of the potential.

    Examples:
        >>> from skcomplex.models.physics.mechanics import ExponentialWellInteraction
        >>> ExponentialWellInteraction('well', target='particle', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, velocity='x_d1')
        self.target = target
        self.k = k

    def _get_energy(self, s: SimpleNamespace) -> Expression:
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
        elements=[MassPointElement('particle', x=x, x_d1=v, m=m)],
        interactions=[ExponentialWellInteraction('well', target='particle', k=k)],
        space=ContinuousSpace(n_dim=1),
    )
