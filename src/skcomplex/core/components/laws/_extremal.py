"""Define extremal laws, which update the site with the smallest or largest value of a state and its neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from typing import ClassVar

from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import BaseNeighborhood, DiscreteTime, LatticeSpace, NetworkSpace
from ._base import BaseLaw

EXTREMA = ('min', 'max')


class ExtremalLaw(BaseLaw):
    """Rule applied to the site with the smallest or largest value of a key state and to its neighbours.

    Every application finds the extremal site, with ties broken by the random stream of the simulation, gathers the
    site and its neighbours without duplicates, the site first, and calls `f(*values, random, **parameters)` with one
    array per target and a uniform random array of the same length. Every application is recorded as an event.

    Args:
        targets:
            The symbol of a lattice state, or a tuple of them that the rule updates together.

        over:
            The neighbourhood of the extremal site that the rule also updates.

        rule:
            The function `f(*values, random, **parameters)`.

        key:
            The lattice state whose extremum is searched, or `None` for the first target.

        extremum:
            `'min'` for the smallest value or `'max'` for the largest.

        repeat:
            The number of applications per step.

        label:
            The name of the events, or the empty string for the path of the element.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import ExtremalLaw
        >>> from skcomplex.core.components.spaces import BoxNeighborhood
        >>> law = ExtremalLaw(var('fitness'), BoxNeighborhood(), lambda values, *, random: random)
        >>> law.extremum, law.repeat
        ('min', 1)
    """

    law_tag: ClassVar[str] = 'update'

    draws = True

    def __init__(
        self,
        targets: Expression | tuple[Expression, ...],
        over: BaseNeighborhood,
        rule: Callable[..., object],
        *,
        key: Expression | None = None,
        extremum: str = 'min',
        repeat: int = 1,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.targets = targets
        self.over = over
        self.rule = rule
        self.key = key
        self.extremum = extremum
        self.repeat = repeat

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice states the rule updates."""
        return self.targets if isinstance(self.targets, tuple) else (self.targets,)

    def _get_key(self) -> object:
        """Return the lattice state whose extremum is searched."""
        return self._get_targets()[0] if self.key is None else self.key

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: discrete time on a lattice or a network."""
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace, NetworkSpace))

    def _check(self) -> None:
        """Raise an error for an unknown extremum or fewer than one application per step."""
        if self.extremum not in EXTREMA:
            error_msg = f'Unknown extremum {self.extremum!r}. Use one of {", ".join(EXTREMA)}.'
            raise ValidationError(error_msg)
        if int(self.repeat) < 1:
            error_msg = f'An extremal law applies at least once per step, but its repeat is {self.repeat}.'
            raise ValidationError(error_msg)
