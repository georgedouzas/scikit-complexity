"""Define pair coupling, which adds a function of every pair within a cutoff, or minus a gradient, to both members."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, cast

from sage.all__sagemath_symbolics import SR, oo, sqrt
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, RateLaw
from ..types import NamespaceFunction, PairPotential, Target
from ._pair import PairInteraction


class PairCouplingInteraction(PairInteraction):
    """Coupling of every pair of members closer than a cutoff, through a function of the pair or a pair potential.

    Every member of a pair adds a value to the rate of its state `write`, which gives the highest derivative of a state
    of any order. With `function`, the value is `function(namespace)`, where `namespace.first` is the member that adds
    it and `namespace.second` the other member. With `potential`, a function `potential(namespace, r)` of the distance
    `r` between the values of the vector state `position` of the pair, the value is minus the gradient of the potential
    with respect to the member's own `position`. A `divisor` names a state or parameter of the members that divides
    the value. With several states `write`, the function returns one value per state, in their order, so one pair
    couples several states jointly.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        first:
            The path of the group whose members are the first of every pair.

        second:
            The path of the group whose members are the second of every pair, which may be the first group.

        cutoff:
            The largest distance between the members of a pair.

        position:
            The name of the vector state whose distances select the pairs, and which the potential is a function of.

        write:
            The name of the state whose rate the coupling adds to, or a sequence of names, one per value of the
            function.

        function:
            The value a member adds, or a sequence of values, one per state `write`, a callable of the namespace of the
            pair, or `None` when the potential is given.

        potential:
            The pair potential, a callable of the namespace of the pair and the distance, or `None` when the function
            is given.

        divisor:
            The name of the state or parameter of the members that divides the value, or `None` for none.

        period:
            The side of a periodic box, for minimum image distances, or `None` for the period of the space.

    Raises:
        ValidationError:
            Not exactly one of the function and the potential is a callable, or the potential has several states to
            write, raised by `check`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var, vector
        >>> from skcomplex.core.interactions import PairCouplingInteraction
        >>> first = SimpleNamespace(x=vector([var('x1')]), m=var('m1'))
        >>> second = SimpleNamespace(x=vector([var('x2')]), m=var('m2'))
        >>> spring = PairCouplingInteraction('spring', 'a', 'b', potential=lambda s, r: r**2 / 2, divisor='m')
        >>> [law.value for law in spring.check().prepare().run(SimpleNamespace(first=first, second=second)).laws_]
        [(-(x1 - x2)/m1), ((x1 - x2)/m2)]
    """

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        position: str = 'x',
        *,
        write: str | Sequence[str] = 'x',
        function: NamespaceFunction | None = None,
        potential: PairPotential | None = None,
        divisor: str | None = None,
        period: float | None = None,
    ) -> None:
        super().__init__(label, first, second, cutoff=cutoff, position=position)
        self.write = write
        self.function = function
        self.potential = potential
        self.divisor = divisor
        self.period = period

    def _check(self) -> None:
        """Raise an error unless exactly one of the function and the potential is a callable.

        Raises:
            ValidationError:
                The function and the potential are both given or both missing, the given one is not callable, or the
                potential has several states to write.
        """
        super()._check()
        given = [item for item in (self.function, self.potential) if item is not None]
        if len(given) != 1 or not callable(given[0]):
            error_msg = (
                f'{type(self).__name__} {self.label!r} has the function {self.function!r} and the potential '
                f'{self.potential!r}. Give exactly one of them as a callable.'
            )
            raise ValidationError(error_msg)
        if self.potential is not None and not isinstance(self.write, str):
            error_msg = f'{type(self).__name__} {self.label!r} writes {self.write!r}, but a potential writes one state.'
            raise ValidationError(error_msg)

    def _get_gradient_value(self, namespace: SimpleNamespace) -> Target:
        """Return minus the gradient of the pair potential with respect to the position of the first member."""
        assert self.potential is not None
        distance = SR.var('r', domain='positive')
        separation = getattr(namespace.first, self.position) - getattr(namespace.second, self.position)
        r = sqrt(separation.dot_product(separation))
        slope = SR(self.potential(namespace, distance)).diff(distance).subs({distance: r})
        return -slope * separation / r

    def _get_values(self, namespace: SimpleNamespace) -> list[Target]:
        """Return the values the first member of a pair adds to the rates of its states, in the order of `write`."""
        if self.function is None:
            values = [self._get_gradient_value(namespace)]
        elif isinstance(self.write, str):
            values = [self.function(namespace)]
        else:
            values = list(cast(Sequence[Target], self.function(namespace)))
        if self.divisor is None:
            return values
        return [value / getattr(namespace.first, self.divisor) for value in values]

    def _get_names(self) -> list[str]:
        """Return the names of the states the coupling writes."""
        return [self.write] if isinstance(self.write, str) else list(self.write)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the written state of both members of a pair.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `first` and `second` for the pair.

        Returns:
            laws:
                The rate laws of the written states of the two members.
        """
        swapped = SimpleNamespace(**{**vars(namespace), 'first': namespace.second, 'second': namespace.first})
        names = self._get_names()
        laws: list[BaseLaw] = []
        for member, pair in ((namespace.first, namespace), (namespace.second, swapped)):
            values = self._get_values(pair)
            laws += [RateLaw(getattr(member, name), value) for name, value in zip(names, values, strict=True)]
        return laws
