"""Define local rules.

A local rule sets the next values of cells from the ordered values of their neighbourhoods.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from ...errors import ValidationError
from ...specs import RoleSpec
from ...types import ElementKind
from ..laws import BaseLaw, LocalRuleLaw
from ..spaces import BaseNeighborhood, BaseSpace
from ._interaction import Interaction


class LocalRuleInteraction(Interaction):
    """Local rule of a cellular automaton.

    The rule sets the next value of every cell from its ordered neighbourhood. The rule is a table of the next value of
    every configuration of the neighbourhood, `TableSpec`, a rule of the sum of the neighbourhood, `TotalisticSpec`, a
    rule of the cell and the sum of its neighbours, `OuterTotalisticSpec`, an object whose `get_table()` returns a
    `TableSpec`, or a function. A function receives one array per neighbour of every read state, in neighbourhood order,
    with one value per cell, the uniform array `random` when the rule is stochastic, and the parameters of the bound
    group it names, and returns the next values of the states, one array per state. A state of several components per
    cell, such as a truncated power series of `PowerSeriesSpec`, reaches a function as one row of components per cell. A
    `StateTranslationNeighborhood` reads a group space at the product of every site and the group element that the first
    read state holds there.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group on the cells of a lattice, the nodes of a network, or a Cayley graph.

        state:
            The name of the state the rule sets, or a tuple of the names of the states it sets together.

        rule:
            The table, the totalistic rule, or the function.

        neighborhood:
            The ordered neighbourhood of every cell, such as `BallNeighborhood(metric='manhattan', center='first')`.

        reads:
            The names of the states whose neighbourhoods a function reads, or `None` for the states it sets.

        schedule:
            `'synchronous'`, `'random-sequential'`, `'random-site'`, `'checkerboard'`, or `'colored'`.

        stochastic:
            Whether a function receives the uniform random array `random`.

    Raises:
        ValidationError:
            The state or the read states are not names, or the neighbourhood is not a neighbourhood.

    Examples:
        >>> from skcomplex.core.components.interactions import LocalRuleInteraction
        >>> from skcomplex.core.components.spaces import BallNeighborhood
        >>> from skcomplex.core.specs import TotalisticSpec
        >>> parity = TotalisticSpec({1: 1, 3: 1})
        >>> around = BallNeighborhood(metric='manhattan', n_dim=1, center='first')
        >>> rule = LocalRuleInteraction('parity', target='cells', state='cell', rule=parity, neighborhood=around)
        >>> rule.schedule
        'synchronous'
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (LocalRuleLaw,)

    def _get_draws(self) -> bool:
        """Return whether the rule draws.

        It draws when it is stochastic or updates sites in a random order or at random.
        """
        return super()._get_draws() or self.stochastic or self.schedule == 'random-site'

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state', 'reads'))],
        state: str | tuple[str, ...],
        rule: object,
        neighborhood: BaseNeighborhood,
        reads: Sequence[str] | None = None,
        schedule: str = 'synchronous',
        stochastic: bool = False,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.rule = rule
        self.neighborhood = neighborhood
        self.reads = reads
        self.schedule = schedule
        self.stochastic = stochastic

    def _get_names(self) -> tuple[str, ...]:
        """Return the names of the states the rule sets."""
        return (self.state,) if isinstance(self.state, str) else tuple(self.state)

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for states that are not names, or a bad neighbourhood.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The state or the read states are not names, or the neighbourhood is not a neighbourhood.
        """
        super()._prepare(context)
        names = [*self._get_names(), *(self.reads or ())]
        if not names or not all(isinstance(name, str) and name.isidentifier() for name in names):
            error_msg = f'The local rule {self.label!r} sets {self.state!r} and reads {self.reads!r}. Give state names.'
            raise ValidationError(error_msg)
        if not isinstance(self.neighborhood, BaseNeighborhood):
            error_msg = f'The neighbourhood of the local rule {self.label!r} is {self.neighborhood!r}. Give one.'
            raise ValidationError(error_msg)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the local rule law of the states of the target.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The local rule law.
        """
        targets = tuple(getattr(self._get_member(namespace, 'target'), name) for name in self._get_names())
        reads = (
            None
            if self.reads is None
            else tuple(getattr(self._get_member(namespace, 'target'), name) for name in self.reads)
        )
        return [
            LocalRuleLaw(
                targets[0] if isinstance(self.state, str) else targets,
                self.neighborhood,
                self.rule,
                reads=reads,
                schedule=self.schedule,
                stochastic=self.stochastic,
                label=self.label,
            ),
        ]
