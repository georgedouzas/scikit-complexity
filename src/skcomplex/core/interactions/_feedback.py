"""Define feedback interactions, single actors whose rules read observables and set the parameters of other parts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import ClassVar

from ...errors import ValidationError
from ..base import BasePart
from ..laws import BaseLaw, RuleLaw
from ..types import ModelViewProtocol
from ._interaction import Interaction


class FeedbackInteraction[ModelT: ModelViewProtocol](Interaction):
    """Single actor that reads the model once per round of its phase and sets parameters from what it reads.

    Its rule runs once per round of its phase, after the agents of the phase. It reads observables with
    `model.observable(name)`, reporters with `model.history(name)`, and agents with `model.agents(name)`, and sets
    parameters of any group or interaction with `model.set_parameter(name, value)`. A change is visible to every rule
    at once and recorded in the trajectory like any parameter. Subclasses declare their own parameters, such as a
    target, and override `act`; otherwise `rule(model)` acts.

    Args:
        label:
            The label of the feedback.

        rule:
            The function of the model view that acts, used when a subclass does not override `act`.

        phase:
            The phase the feedback acts in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.core.interactions import FeedbackInteraction
        >>> FeedbackInteraction('controller', rule=lambda model: None).check().phase is None
        True
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[ModelT], None] | None = None,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.rule = rule
        self.phase = phase

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the rule of the feedback in its phase.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the interaction.
        """
        return [RuleLaw(self.act, phase=self.phase)]

    def act(self, model: ModelT) -> None:
        """Read the model and set parameters.

        Args:
            model:
                The model view scoped to the feedback.

        Raises:
            ValidationError:
                The feedback has neither a rule nor its own `act`.
        """
        if self.rule is None:
            error_msg = f'The feedback {self.label!r} has no rule. Pass `rule` or override `act`.'
            raise ValidationError(error_msg)
        self.rule(model)
