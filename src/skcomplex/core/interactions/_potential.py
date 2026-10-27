"""Define potential interactions, whose forces are minus the gradients of a potential energy of positions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod

from sage.all__sagemath_symbolics import SR, Expression, vector

from ..laws import RateLaw
from ..specs import RoleSpec
from ._interaction import Interaction


class PotentialInteraction(Interaction):
    """Base class of interactions between particles defined by a potential energy of their positions.

    A subclass takes one role per particle and returns the energy from `_get_energy`. The force on each particle is
    minus the gradient of the energy with respect to its position `x`, derived symbolically, and changes its velocity
    `v` by the force divided by its mass `m`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

    Examples:
        >>> from typing import Annotated
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var, vector
        >>> from skcomplex.core.interactions import PotentialInteraction
        >>> from skcomplex.core.specs import ParameterSpec, RoleSpec
        >>> class SpringInteraction(PotentialInteraction):
        ...     def __init__(self, label, a: Annotated[str, RoleSpec()], b: Annotated[str, RoleSpec()]):
        ...         super().__init__(label)
        ...         self.a = a
        ...         self.b = b
        ...
        ...     def _get_energy(self, namespace):
        ...         gap = namespace.a.x - namespace.b.x
        ...         return gap.dot_product(gap) / 2
        >>> a = SimpleNamespace(x=vector([var('xa')]), v=vector([var('va')]), m=1)
        >>> b = SimpleNamespace(x=vector([var('xb')]), v=vector([var('vb')]), m=1)
        >>> spring = SpringInteraction('spring', 'left', 'right').check().prepare()
        >>> [law.value for law in spring.run(SimpleNamespace(a=a, b=b))]
        [(-xa + xb), (xa - xb)]
    """

    @abstractmethod
    def _get_energy(self, namespace: object) -> Expression:
        """Return the potential energy as a symbolic expression of the positions of the particles of the roles."""

    def _get_laws(self, namespace: object) -> list[object]:
        """Return the velocity change of every particle from the gradient of the energy.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with one namespace per role.

        Returns:
            laws:
                The rate laws of the velocities of the particles.
        """
        energy = SR(self._get_energy(namespace))
        laws: list[object] = []
        for name, spec in self._get_specs().items():
            if isinstance(spec, RoleSpec):
                particle = getattr(namespace, name)
                gradient = vector(SR, [energy.diff(component) for component in particle.x])
                laws.append(RateLaw(particle.v, -gradient / particle.m))
        return laws
