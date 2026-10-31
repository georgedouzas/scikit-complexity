"""Define imitation of a discrete state of the neighbours, by a random neighbour, the majority, or the best."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, JumpLaw
from ..laws.utils import neighbors, where
from ..spaces import AdjacencyNeighborhood
from ._neighbor import NeighborInteraction

RULES = ('random', 'majority', 'best')


class ImitationInteraction(NeighborInteraction):
    """Imitation of a discrete state of the neighbours of every member, at a rate.

    Every member imitates at the rate `rate`. By the rule `'random'` it copies a neighbour chosen at random, by
    `'majority'` it takes the option that more of its neighbours hold than any other, and by `'best'` it copies the
    neighbour whose payoff is largest, when that payoff is larger than its own. A member without such a neighbour keeps
    its state. The rules `'majority'` and `'best'` read the neighbourhood of the space, the graph of a network or the
    neighbourhood of a lattice. The laws are jumps between every pair of options, so the state changes by random events
    in continuous time.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        nodes:
            The path of the group on the nodes or sites.

        state:
            The name of the discrete state of the members.

        payoff:
            The name of the state that holds the payoff of the members, which the rule `'best'` reads.

        options:
            The values the state takes.

        rate:
            The rate at which every member imitates.

        rule:
            `'random'`, `'majority'`, or `'best'`.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

    Raises:
        ValidationError:
            The rule is unknown, or the options are not distinct.

    Examples:
        >>> from skcomplex.core.interactions import ImitationInteraction
        >>> ImitationInteraction('imitation', nodes='voters', rule='majority').check().options
        (0.0, 1.0)
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        state: str = 'state',
        payoff: str = 'payoff',
        options: Sequence[float] = (0.0, 1.0),
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        rule: str = 'random',
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.nodes = nodes
        self.state = state
        self.payoff = payoff
        self.options = options
        self.rate = rate
        self.rule = rule

    def _check(self) -> None:
        """Raise an error for an unknown rule or options that are not distinct.

        Raises:
            ValidationError:
                The rule is unknown, or the options are not distinct.
        """
        super()._check()
        if self.rule not in RULES:
            error_msg = f'The rule of imitation is {self.rule!r}. Use one of {", ".join(RULES)}.'
            raise ValidationError(error_msg)
        if len(set(self.options)) < max(len(self.options), 2):
            error_msg = f'The options of imitation are {self.options!r}. Give at least two distinct values.'
            raise ValidationError(error_msg)

    def _switches(self, value: Expression, rate: Expression, chosen: Expression) -> list[BaseLaw]:
        """Return the jumps of a state from every option to the chosen option, at a rate when it is chosen."""
        return [
            JumpLaw(rate * where(value == first, 1, 0) * where(chosen == second, 1, 0), {value: second - first})
            for first, second in itertools.permutations(self.options, 2)
        ]

    def _copies(self, member: SimpleNamespace, other: SimpleNamespace, rate: Expression) -> list[BaseLaw]:
        """Return the copies of the state of the other end of an edge by a member, chosen among its neighbours."""
        value = getattr(member, self.state)
        return self._switches(value, rate / member.degree, getattr(other, self.state))

    def _count(self, other: SimpleNamespace, option: float) -> Expression:
        """Return the number of the neighbours of a member that hold an option."""
        over = None if self.layer is None else AdjacencyNeighborhood(layer=self.layer)
        return neighbors(where(getattr(other, self.state) == option, 1, 0), over=over)

    def _majority(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jumps of the target to the option that more of its neighbours hold than any other."""
        value = getattr(namespace.target, self.state)
        counts = {option: self._count(namespace.source, option) for option in self.options}
        laws = []
        for first, second in itertools.permutations(self.options, 2):
            leads = [where(counts[second] > count, 1, 0) for option, count in counts.items() if option != second]
            rate = namespace.rate * where(value == first, 1, 0)
            for lead in leads:
                rate = rate * lead
            laws.append(JumpLaw(rate, {value: second - first}))
        return laws

    def _best(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jumps of the target to the option of its neighbour of largest payoff, when it pays more."""
        source, target = namespace.source, namespace.target
        over = None if self.layer is None else AdjacencyNeighborhood(layer=self.layer)
        payoff = getattr(source, self.payoff)
        best = neighbors(getattr(source, self.state), over=over, reduce='argmax', weight=payoff)
        largest = neighbors(payoff, over=over, reduce='max')
        rate = namespace.rate * where(largest > getattr(target, self.payoff), 1, 0)
        return self._switches(getattr(target, self.state), rate, best)

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jumps of the state by the rule of the imitation.

        Args:
            namespace:
                The namespace of an edge.

        Returns:
            laws:
                The jumps of the state between every pair of options.
        """
        if self.rule == 'majority':
            return self._majority(namespace)
        if self.rule == 'best':
            return self._best(namespace)
        source, target = namespace.source, namespace.target
        return [*self._copies(target, source, namespace.rate), *self._copies(source, target, namespace.rate)]
