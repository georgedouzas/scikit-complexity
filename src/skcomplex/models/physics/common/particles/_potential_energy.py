"""Define potential energies of particles, whose forces are minus the gradients of the energy of their positions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from .....core.interactions import GradientInteraction
from .....specs import RoleSpec


class PotentialEnergyInteraction(GradientInteraction):
    """Potential energy of the positions of particles, whose force on every particle changes its velocity.

    It is the gradient of the energy with the mass as divisor: the force `-dU/dx` on every particle changes its
    velocity state by the force divided by its mass. A subclass binds one role per particle and returns the energy
    from `_get_potential`, or a user passes the energy as `potential`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or particle group.

        potential:
            A function of the namespace that returns the energy, or `None` for a subclass that returns it.

        states:
            The name of the vector state that holds the position of every particle.

        velocity:
            The name of the vector state that holds the velocity of every particle, which the force changes.

        divisor:
            The name of the state or parameter that holds the mass of every particle.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var, vector
        >>> from skcomplex.models.physics.common.particles import PotentialEnergyInteraction
        >>> ball = SimpleNamespace(x=vector([var('x')]), v=vector([var('v')]), m=2)
        >>> trap = PotentialEnergyInteraction('trap', target='ball', potential=lambda s: s.target.x[0] ** 2)
        >>> [law.value for law in trap.check().prepare().run(SimpleNamespace(target=ball)).laws_]
        [(-x)]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        potential: Callable[[SimpleNamespace], object] | None = None,
        states: str | Sequence[str] = 'x',
        velocity: str = 'v',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=target, states=states, potential=potential, divisor=divisor)
        self.velocity = velocity

    def _get_rate_state(self, member: SimpleNamespace, name: str) -> object:
        """Return the velocity of a particle, the state whose rate the force gives.

        Args:
            member:
                The namespace of the particle.

            name:
                The name of the position state.

        Returns:
            velocity:
                The velocity state of the particle.
        """
        return getattr(member, self.velocity)
