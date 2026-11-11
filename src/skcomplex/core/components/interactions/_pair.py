"""Define the interaction whose laws run for every pair of members within a cutoff."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

from sage.all__sagemath_symbolics import SR, oo, sqrt
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ...types import ElementKind
from ..laws import BaseLaw, RateLaw
from ..spaces import BaseSpace
from ..types import NamespaceFunction, PairPotential, Target
from ._base import BaseInteraction


class PairInteraction(BaseInteraction):
    """An interaction whose laws run for every pair of members within a cutoff.

    The pairs are those whose vector states `position` lie closer than `cutoff`. The laws are written for one pair and
    read `namespace.first`, the member that receives the value, `namespace.second`, the other member, and the
    parameters. Each member adds a value to the rate of its state `write`, which is the highest derivative of a state of
    any order:

    - With `function`, the value is `function(namespace)`, or one value per state when `write` names several.
    - With `potential`, a function `potential(namespace, r)` of the distance `r`, the value is minus the gradient of the
      potential with respect to the member's own position.

    A `divisor` divides the value, such as a mass. With neither a function nor a potential, a subclass writes its own
    laws in `_get_laws`, and a subclass that names the members otherwise returns their names from `_get_role`.

    Unlike `NeighborInteraction`, whose links are fixed edges, it finds its pairs by distance at every evaluation.
    Choose it when members move in continuous space and act on those close by.

    It appears in many fields. Atoms of a gas repel under the Lennard-Jones potential, stars attract by gravity within a
    cutoff, and pedestrians keep their distance in a crowd.

    Args:
        label:
            The name of the interaction, unique among the components of its system, such as `'repulsion'`.

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
            pair, or `None`.

        potential:
            The pair potential, a callable of the namespace of the pair and the distance, or `None`.

        divisor:
            The name of the state or parameter of the members that divides the value, or `None` for none.

        period:
            The side of a periodic box, for minimum image distances, or `None` for the period of the space.

    Raises:
        ValidationError:
            The function and the potential are both given, the given one is not a callable, or the potential has
            several states to write, raised by `prepare`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var, vector
        >>> from skcomplex.core.components.interactions import PairInteraction
        >>> first = SimpleNamespace(x=vector([var('x1')]), m=var('m1'))
        >>> second = SimpleNamespace(x=vector([var('x2')]), m=var('m2'))
        >>> spring = PairInteraction('spring', 'a', 'b', potential=lambda s, r: r**2 / 2, divisor='m')
        >>> [law.value for law in spring.prepare().run(SimpleNamespace(first=first, second=second)).laws_]
        [(-(x1 - x2)/m1), ((x1 - x2)/m2)]
        >>> PairInteraction('contact', 'a', 'b').prepare().run(SimpleNamespace(first=first, second=second)).laws_
        []
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RateLaw,)

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('position', 'write', 'divisor'))],
        second: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('position', 'write', 'divisor'))],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        position: str = 'x',
        *,
        write: str | Sequence[str] = 'x',
        function: NamespaceFunction | None = None,
        potential: PairPotential | None = None,
        divisor: str | None = None,
        period: float | None = None,
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.cutoff = cutoff
        self.position = position
        self.write = write
        self.function = function
        self.potential = potential
        self.divisor = divisor
        self.period = period

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error unless at most one of the function and the potential is given.

        The one given must be a callable.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The function and the potential are both given, the given one is not callable, or the potential has
                several states to write.
        """
        super()._prepare(context)
        given = [item for item in (self.function, self.potential) if item is not None]
        if len(given) > 1 or not all(callable(item) for item in given):
            error_msg = (
                f'{type(self).__name__} {self.label!r} has the function {self.function!r} and the potential '
                f'{self.potential!r}. Give at most one of them, as a callable.'
            )
            raise ValidationError(error_msg)
        if self.potential is not None and not isinstance(self.write, str):
            error_msg = f'{type(self).__name__} {self.label!r} writes {self.write!r}, but a potential writes one state.'
            raise ValidationError(error_msg)

    def _get_gradient_value(self, namespace: SimpleNamespace) -> Target:
        """Return minus the gradient of the pair potential.

        The gradient is with respect to the position of the first member.
        """
        assert self.potential is not None
        distance = SR.var('r', domain='positive')
        first, second = self._get_member(namespace, 'first'), self._get_member(namespace, 'second')
        separation = getattr(first, self.position) - getattr(second, self.position)
        r = sqrt(separation.dot_product(separation))
        slope = SR(self.potential(namespace, distance)).diff(distance).subs({distance: r})
        return -slope * separation / r

    def _get_values(self, namespace: SimpleNamespace) -> list[Target]:
        """Return the values the first member of a pair adds to the rates of its states.

        They are in the order of `write`.
        """
        if self.function is None:
            values = [self._get_gradient_value(namespace)]
        elif isinstance(self.write, str):
            values = [self.function(namespace)]
        else:
            values = list(cast(Sequence[Target], self.function(namespace)))
        if self.divisor is None:
            return values
        return [value / getattr(self._get_member(namespace, 'first'), self.divisor) for value in values]

    def _get_names(self) -> list[str]:
        """Return the names of the states the coupling writes."""
        return [self.write] if isinstance(self.write, str) else list(self.write)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the written states of both members of a pair.

        There are none without a function and a potential.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `first` and `second` for the pair.

        Returns:
            laws:
                The rate laws of the written states of the two members.
        """
        if self.function is None and self.potential is None:
            return []
        first, second = self._get_member(namespace, 'first'), self._get_member(namespace, 'second')
        swapped = SimpleNamespace(
            **{**vars(namespace), self._get_role('first'): second, self._get_role('second'): first},
        )
        names = self._get_names()
        laws: list[BaseLaw] = []
        for member, pair in ((first, namespace), (second, swapped)):
            values = self._get_values(pair)
            laws += [RateLaw(getattr(member, name), value) for name, value in zip(names, values, strict=True)]
        return laws
