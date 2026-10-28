"""Define local rule laws, the next values of lattice cells from the ordered values of their neighbourhoods."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from sage.all__sagemath_symbolics import Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import BaseNeighborhood, CayleySpace, DiscreteTime, LatticeSpace, NetworkSpace
from ._base import BaseLaw

SCHEDULES = ('synchronous', 'random-sequential', 'random-site', 'checkerboard', 'colored')


class LocalRuleLaw(BaseLaw):
    """Next value of lattice cells computed from the ordered values of their neighbourhoods.

    A function rule receives one array per neighbour of every read state, in neighbourhood order, with one value per
    site, and keyword arguments for the parameters of the element that it names, plus `random`, an array of uniform
    numbers per site, when the rule is stochastic, and `degree`, the number of neighbours of every site besides the
    site itself, when it names it. It returns the next values of the target, or a tuple with one array per target.

    Args:
        targets:
            The symbol of a lattice state, or a tuple of them that the rule updates together.

        over:
            The neighbourhood whose values the rule reads: a lattice neighbourhood of offsets, an adjacency on a
            network, or a memory of group elements on a group space.

        rule:
            A function, a `Table`, a `Totalistic` rule, an `OuterTotalistic` rule, or an object whose `table()` returns
            a `Table`, such as a rule code.

        reads:
            The lattice states whose neighbourhoods a function rule reads, or `None` for the targets.

        schedule:
            `'synchronous'` to update every site from the previous configuration, `'random-sequential'` to update the
            sites one at a time in a random order, `'random-site'` to update as many randomly chosen sites,
            `'checkerboard'` to update the two sublattices of a square lattice in turn, or `'colored'` to update the
            classes of a proper coloring of the neighbour graph in turn.

        stochastic:
            Whether a function rule receives the uniform random array `random`.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import LocalRuleLaw
        >>> from skcomplex.specs import TotalisticSpec
        >>> from skcomplex.core.spaces import BoxNeighborhood
        >>> LocalRuleLaw(var('cell'), BoxNeighborhood(), TotalisticSpec({2: 1})).check().prepare().targets_
        (cell,)
    """

    def __init__(
        self,
        targets: Expression | tuple[Expression, ...],
        over: BaseNeighborhood,
        rule: object,
        *,
        reads: tuple[Expression, ...] | None = None,
        schedule: str = 'synchronous',
        stochastic: bool = False,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.targets = targets
        self.over = over
        self.rule = rule
        self.reads = reads
        self.schedule = schedule
        self.stochastic = stochastic

    def _get_targets(self) -> tuple[object, ...]:
        """Return the lattice states the rule updates."""
        return self.targets if isinstance(self.targets, tuple) else (self.targets,)

    def _get_requirements(self) -> RequirementSpec:
        """Return what the rule requires of a system: discrete time on a lattice, a network, or a Cayley graph."""
        return RequirementSpec(time=(DiscreteTime,), space=(LatticeSpace, NetworkSpace, CayleySpace))

    def _check(self) -> Self:
        """Raise an error for an unknown schedule.

        Returns:
            law:
                The law itself.
        """
        if self.schedule not in SCHEDULES:
            error_msg = f'Unknown schedule {self.schedule!r}. Use one of {", ".join(SCHEDULES)}.'
            raise ValidationError(error_msg)
        return self
