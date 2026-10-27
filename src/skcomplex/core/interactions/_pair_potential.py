"""Define pair potential interactions, whose forces derive from a potential energy of the distance of a pair."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, oo, sqrt
from sage.sets.real_set import RealSet

from ..laws import RateLaw
from ..specs import ParameterSpec, RoleSpec
from ._pair import PairInteraction


class PairPotentialInteraction(PairInteraction):
    """Base class of pair interactions between particles defined by a potential energy of their distance.

    A subclass returns the energy of one pair at a distance from `_get_pair_energy`. The force on each particle of a
    pair is minus the derivative of the energy along their separation, derived symbolically, and changes its velocity
    `v` by the force divided by its mass `m`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        first:
            The path of the group whose members are the first of every pair.

        second:
            The path of the group whose members are the second of every pair, which may be the first group.

        cutoff:
            The distance beyond which pairs do not interact.

        position:
            The name of the vector state that holds the positions of the members.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var, vector
        >>> from skcomplex.core.interactions import PairPotentialInteraction
        >>> class HarmonicInteraction(PairPotentialInteraction):
        ...     def _get_pair_energy(self, namespace, r):
        ...         return r**2 / 2
        >>> first = SimpleNamespace(x=vector([var('x1')]), v=vector([var('v1')]), m=1)
        >>> second = SimpleNamespace(x=vector([var('x2')]), v=vector([var('v2')]), m=1)
        >>> harmonic = HarmonicInteraction('bond', 'atoms', 'atoms').check().prepare()
        >>> [law.value for law in harmonic.run(SimpleNamespace(first=first, second=second))]
        [(-x1 + x2), (x1 - x2)]
    """

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        position: str = 'x',
        *,
        period: float | None = None,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position)
        self.period = period

    @abstractmethod
    def _get_pair_energy(self, namespace: object, r: Expression) -> Expression:
        """Return the potential energy of one pair at the distance `r` as a symbolic expression."""

    def _get_laws(self, namespace: object) -> list[object]:
        """Return the velocity changes of both particles of a pair from the derivative of the pair energy.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `first` and `second` for the pair.

        Returns:
            laws:
                The rate laws of the velocities of the two particles.
        """
        first, second = namespace.first, namespace.second  # type: ignore[attr-defined]
        distance = SR.var('r', domain='positive')
        separation = getattr(first, self.position) - getattr(second, self.position)
        r = sqrt(separation.dot_product(separation))
        slope = SR(self._get_pair_energy(namespace, distance)).diff(distance).subs({distance: r})
        force = -slope * separation / r
        return [RateLaw(first.v, force / first.m), RateLaw(second.v, -force / second.m)]
