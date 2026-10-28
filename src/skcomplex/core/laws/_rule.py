"""Define rule laws, behaviour written as code that agents and models run every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import Self

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import DiscreteTime
from ._base import BaseLaw


class RuleLaw(BaseLaw):
    """Behaviour written as code, run once per step for every member of an agent group, or once per step for a model.

    An element whose laws include rules is an agent, alone or in a group. A rule of an element receives the agent and
    the model, `function(agent, model)`. A rule of an interaction receives the model alone, `function(model)`. A system
    with rules is agent-based and needs discrete time.

    Args:
        function:
            The function that runs the rule.

        phase:
            The name of the phase of the step the rule runs in, or `None` for the default phase.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from skcomplex.core.laws import RuleLaw
        >>> moved = []
        >>> RuleLaw(lambda agent, model: moved.append(agent)).check().run(('walker', None))
        >>> moved
        ['walker']
    """

    def __init__(self, function: Callable[..., object], phase: str | None = None, *, label: str = '') -> None:
        super().__init__(label)
        self.function = function
        self.phase = phase

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since a rule changes the agents it receives in place."""
        return ()

    def _get_requirements(self) -> RequirementSpec:
        """Return what the rule requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _check(self) -> Self:
        """Raise an error for a function that cannot be called.

        Returns:
            law:
                The law itself.
        """
        if not callable(self.function):
            error_msg = f'The function of a rule must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
        return self

    def _run(self, inputs: object) -> object:
        """Run the rule on its arguments, the agent and the model, or the model alone.

        Args:
            inputs:
                The arguments of the function, a sequence such as `(agent, model)`.

        Returns:
            result:
                What the function returns, usually `None`.
        """
        arguments = tuple(inputs) if isinstance(inputs, Sequence) else (inputs,)
        return self.function(*arguments)
