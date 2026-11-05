"""Define rule laws, behaviour written as code that agents and models run every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import DiscreteTime
from ._base import BaseLaw


class RuleLaw(BaseLaw):
    """Behaviour written as code, run once per step for every member of an agent group, or once per step for a model.

    An element whose laws include rules is an agent, alone or in a group. A rule of an element receives the agent and
    the model, `function(agent, model)`. A rule of an interaction receives the model alone, `function(model)`, unless it
    names the role whose `members` run it: every member of the groups of that role then runs it in its own activation,
    after the rules of its element and the member rules of the interactions before it, as `function(agent, model)` with
    the model view of the interaction. A system with rules is agent-based and needs discrete time.

    Args:
        function:
            The function that runs the rule.

        phase:
            The name of the phase of the step the rule runs in, or `None` for the default phase.

        members:
            The role of the interaction whose members run the rule, or `None` for a rule of the model or of an element.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from skcomplex.core.laws import RuleLaw
        >>> moved = []
        >>> RuleLaw(lambda agent, model: moved.append(agent)).check().prepare().run(('walker', None)).value_
        >>> moved
        ['walker']
    """

    def __init__(
        self,
        function: Callable[..., object],
        phase: str | None = None,
        *,
        members: str | None = None,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.function = function
        self.phase = phase
        self.members = members

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since a rule changes the agents it receives in place."""
        return ()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the rule requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _check(self) -> None:
        """Raise an error for a function that cannot be called, or members that name no role."""
        if not callable(self.function):
            error_msg = f'The function of a rule must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
        if self.members is not None and not (isinstance(self.members, str) and self.members):
            error_msg = f'The members of a rule name a role, but they are {self.members!r}.'
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Run the rule on its arguments, the agent and the model, or the model alone.

        Args:
            inputs:
                The arguments of the function, a sequence such as `(agent, model)`.
        """
        arguments = tuple(inputs) if isinstance(inputs, Sequence) else (inputs,)
        self.value_ = self.function(*arguments)
