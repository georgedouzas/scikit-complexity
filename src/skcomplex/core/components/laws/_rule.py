"""Define the law of behaviour written as Python code and run once per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from typing import ClassVar

from ...errors import ValidationError
from ...specs import RequirementSpec
from ..spaces import DiscreteTime
from ._base import BaseLaw


class RuleLaw(BaseLaw):
    """A law of behaviour written as Python code and run once per step.

    A rule runs for every member of an agent group, or once for a system. A rule of an element receives the agent and
    the system, `function(agent, system)`, and an element whose laws include rules is an agent. A rule of an interaction
    receives the system alone, `function(system)`, unless it names the role whose `members` run it. Every member of that
    role then runs it in its own activation, after the rules of its element, as `function(agent, system)`. A system with
    rules is agent-based and needs discrete time.

    Unlike `UpdateLaw`, which gives the next value as an expression the compiler lowers, it runs arbitrary Python on
    each agent. Choose it when behaviour needs branches, memory, or search that an expression cannot hold.

    It appears in many fields. Sheep and wolves move and eat on a grid, traders follow strategies in a market, and
    households decide where to live in a model of segregation.

    Args:
        function:
            The function that runs the rule.

        phase:
            The name of the phase of the step the rule runs in, or `None` for the default phase.

        members:
            The role of the interaction whose members run the rule, or `None` for a rule of the system or of an element.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from skcomplex.core.components.laws import RuleLaw
        >>> moved = []
        >>> RuleLaw(lambda agent, system: moved.append(agent)).prepare().run(('walker', None)).value_
        >>> moved
        ['walker']
    """

    law_tag: ClassVar[str] = 'update'

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

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for an uncallable function, or members that name no role.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
        if not callable(self.function):
            error_msg = f'The function of a rule must be callable, but it is {self.function!r}.'
            raise ValidationError(error_msg)
        if self.members is not None and not (isinstance(self.members, str) and self.members):
            error_msg = f'The members of a rule name a role, but they are {self.members!r}.'
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Run the rule on its arguments, the agent and the system, or the system alone.

        Args:
            inputs:
                The arguments of the function, a sequence such as `(agent, system)`.
        """
        arguments = tuple(inputs) if isinstance(inputs, Sequence) else (inputs,)
        self.value_ = self.function(*arguments)
