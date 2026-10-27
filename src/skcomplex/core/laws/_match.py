"""Define match laws, the pairings of the members of agent groups that the rules of a step read."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Self

import numpy as np

from ..errors import ValidationError
from ..spaces import DiscreteTime
from ..specs import RequirementSpec
from ._base import BaseLaw
from ._rule import RuleLaw
from .utils import random_pairing

type Pairing = Callable[[Sequence[object], Sequence[object], np.random.Generator, int], list[tuple[object, object]]]


class MatchLaw(BaseLaw):
    """Pairing of the members of one agent group, or of two, every step, which the rules of the step read.

    The members are paired uniformly at random by default. Any other pairing, such as stable matching, an order book,
    or an auction, is a function given as `pairing`. Rules read the pairs of the step with `matches(model, label)`.

    Args:
        left:
            The path of the first agent group.

        right:
            The path of the second agent group, or `None` for pairs within the first.

        pairing:
            A function `pairing(left, right, rng, capacity)` of the agents of both groups that returns their pairs, or
            `None` for uniform random pairing.

        capacity:
            The largest number of pairs of every member.

        phase:
            The phase of the step the pairing runs in, before the rules of the phase that read it.

        label:
            The name of the pairs.

    Examples:
        >>> from skcomplex.core.laws import MatchLaw
        >>> MatchLaw('buyers', 'sellers', label='trades').check().capacity
        1
    """

    def __init__(
        self,
        left: str,
        right: str | None = None,
        pairing: Pairing | None = None,
        capacity: int = 1,
        phase: str | None = None,
        *,
        label: str = 'match',
    ) -> None:
        super().__init__(label)
        self.left = left
        self.right = right
        self.pairing = pairing
        self.capacity = capacity
        self.phase = phase

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since a pairing stores pairs for the rules of the step."""
        return ()

    def _get_requirements(self) -> RequirementSpec:
        """Return what the pairing requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _get_rule(self) -> RuleLaw:
        """Return the rule the agent stepper runs, which stores the pairs of the step in the model."""
        return RuleLaw(self.run, phase=self.phase)

    def _check(self) -> Self:
        """Raise an error for a capacity that is not a positive integer.

        Returns:
            law:
                The law itself.
        """
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity < 1:
            error_msg = f'The match {self.label!r} has capacity {self.capacity!r}. Give a positive integer.'
            raise ValidationError(error_msg)
        return self

    def _run(self, inputs: object) -> object:
        """Store the pairs of the step in the storage of the model under `match:<label>`.

        Args:
            inputs:
                The model view the rules receive, with `agents(path)`, `storage`, and `rng`.

        Returns:
            pairs:
                The pairs of agents.
        """
        model = inputs
        left = model.agents(self.left)  # type: ignore[attr-defined]
        right = left if self.right is None else model.agents(self.right)  # type: ignore[attr-defined]
        pairs = (self.pairing or random_pairing)(left, right, model.rng, self.capacity)  # type: ignore[attr-defined]
        model.storage[f'match:{self.label}'] = pairs  # type: ignore[attr-defined]
        return pairs
