"""Define match laws, the pairings of the members of agent groups that the rules of a step read."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...errors import ValidationError
from ...specs import RequirementSpec
from ..times import DiscreteTime
from ..types import ModelViewProtocol, Pairing
from ._base import BaseLaw
from ._rule import RuleLaw
from .utils import random_pairing


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

    def _check(self) -> None:
        """Raise an error for a capacity that is not a positive integer."""
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity < 1:
            error_msg = f'The match {self.label!r} has capacity {self.capacity!r}. Give a positive integer.'
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Store the pairs of the step in the storage of the model under `match:<label>`.

        Args:
            inputs:
                The model view the rules receive, with `agents(path)`, `storage`, and `rng`.

        Raises:
            ValidationError:
                The input is not a model view.
        """
        if not isinstance(inputs, ModelViewProtocol):
            error_msg = f'The match {self.label!r} runs on the model view of a step, but got {inputs!r}.'
            raise ValidationError(error_msg)
        left = inputs.agents(self.left)
        right = left if self.right is None else inputs.agents(self.right)
        pairs = (self.pairing or random_pairing)(left, right, inputs.rng, self.capacity)
        inputs.storage[f'match:{self.label}'] = pairs
        self.value_ = pairs
