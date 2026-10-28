"""Define the linear state feedback that sets a control input of a plant from its states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, BoundaryLaw
from ....core.specs import RoleSpec
from ._utils import _limited, _read


class StateFeedbackInteraction(Interaction):
    """Linear state feedback `u = -K x` that sets a control input of a plant from its states.

    The control input is an adaptive parameter of the plant, which the controller sets after every solver step, so it
    holds within a step like a sampled controller. A gain from `LinearizationAnalysis` of the plant makes the
    controller a linear-quadratic regulator.

    Args:
        label:
            The label of the controller.

        control:
            The name of the adaptive parameter of the plant that the controller sets.

        states:
            The names of the fed back states of the plant, such as `('x', 'x_d1')` for a position and its velocity.

        gain:
            The gain of every state, in the order of `states`.

        limits:
            The lower and upper limit of the control input, or `None` for no limits.

        **values:
            The values of the declared variables, by name:

            plant:
                The path of the element that holds the states and the control input.

    Examples:
        >>> from skcomplex.models.engineering.control import StateFeedbackInteraction
        >>> StateFeedbackInteraction(
        ...     'regulator', plant='mass', control='force', states=('x', 'x_d1'), gain=(1.0, 1.7)
        ... ).gain
        (1.0, 1.7)
    """

    def __init__(
        self,
        label: str,
        *,
        plant: Annotated[str, RoleSpec()],
        control: str = 'u',
        states: Sequence[str] = (),
        gain: Sequence[float] = (),
        limits: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(label)
        self.plant = plant
        self.control = control
        self.states = states
        self.gain = gain
        self.limits = limits

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the control input the feedback sets.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        feedback = -sum(
            (float(gain) * _read(s.plant, name) for name, gain in zip(self.states, self.gain, strict=True)),
            0,
        )
        return [BoundaryLaw(_read(s.plant, self.control), _limited(feedback, self.limits))]
