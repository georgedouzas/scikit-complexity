"""Define pair force interactions, whose forces derive from a given energy function of the distance of a pair."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ._pair_potential import PairPotentialInteraction


class PairForceInteraction(PairPotentialInteraction):
    """Pair interaction whose force is minus the derivative of a given energy function of the distance of a pair.

    The energy is a callable `energy(namespace, r)` that returns the energy of one pair at the distance `r` as a
    symbolic expression of `r` and of the parameters in the namespace, such as `namespace.cutoff`. Pairs farther apart
    than the cutoff do not interact.

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

        velocity:
            The name of the vector state that holds the velocities of the members.

        mass:
            The name of the state or parameter that holds the masses of the members.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        energy:
            The energy of one pair, a callable of the namespace of the symbols and of the distance.

    Raises:
        ValidationError:
            The energy is not callable, raised by `check`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var, vector
        >>> from skcomplex.core.interactions import PairForceInteraction
        >>> first = SimpleNamespace(x=vector([var('x1')]), v=vector([var('v1')]), m=1)
        >>> second = SimpleNamespace(x=vector([var('x2')]), v=vector([var('v2')]), m=1)
        >>> bond = PairForceInteraction('bond', 'atoms', 'atoms', energy=lambda namespace, r: r**2 / 2)
        >>> [law.value for law in bond.check().prepare().run(SimpleNamespace(first=first, second=second)).laws_]
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
        velocity: str = 'v',
        mass: str = 'm',
        period: float | None = None,
        energy: Callable[[SimpleNamespace, Expression], Expression],
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            velocity=velocity,
            mass=mass,
            period=period,
        )
        self.energy = energy

    def _check(self) -> None:
        """Raise an error for declared states or for an energy that is not callable.

        Raises:
            ValidationError:
                The type declares states, or the energy is not callable.
        """
        super()._check()
        if not callable(self.energy):
            error_msg = (
                f'The energy of {type(self).__name__} {self.label!r} is {self.energy!r}. Give a callable '
                '`energy(namespace, r)` that returns the energy of one pair at the distance `r`.'
            )
            raise ValidationError(error_msg)

    def _get_pair_energy(self, namespace: SimpleNamespace, r: Expression) -> Expression:
        """Return the energy of one pair at the distance `r` from the energy function.

        Args:
            namespace:
                The namespace of the symbols the energy can use.

            r:
                The distance between the two members of the pair.

        Returns:
            energy:
                The pair energy as a symbolic expression.
        """
        return self.energy(namespace, r)
