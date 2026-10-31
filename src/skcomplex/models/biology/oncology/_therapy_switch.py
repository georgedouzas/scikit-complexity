"""Define therapy switches, crossings that set a therapy on or off when a signal of the tumour crosses a threshold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import CrossingInteraction
from ....specs import ParameterSpec, RoleSpec


class TherapySwitchInteraction(CrossingInteraction):
    """Switch that sets the therapy state of a tumour to a value when a signal crosses a threshold in a direction.

    A pair of switches, one that sets the therapy to 1 when the signal rises through a high threshold and one that
    sets it to 0 when the signal falls through a low one, is the relay with hysteresis of intermittent therapy. Every
    switch is an event labelled by the interaction.

    Args:
        label:
            The label of the interaction and of its events.

        target:
            The path of the tumour.

        signal:
            A function of the namespace that returns the signal, such as the PSA of the tumour.

        threshold:
            The value of the signal at the switch.

        value:
            The value of the therapy state after the switch.

        state:
            The name of the therapy state.

        direction:
            The direction of the crossing, 1 for a rising and -1 for a falling signal.

    Raises:
        ValidationError:
            The direction is not -1, 0, or 1, or the signal is not a function.

    Examples:
        >>> from skcomplex.models.biology.oncology import TherapySwitchInteraction
        >>> switch = TherapySwitchInteraction('on', target='tumour', signal=lambda s: s.target.AD, threshold=15.0)
        >>> switch.check().threshold
        15.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        signal: Callable[[SimpleNamespace], object],
        threshold: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        value: float = 1.0,
        state: str = 'u',
        direction: int = 1,
    ) -> None:
        super().__init__(
            label,
            target=target,
            guard=self._get_guard,
            reset={state: self._get_value},
            direction=direction,
        )
        self.signal = signal
        self.threshold = threshold
        self.value = value
        self.state = state

    def _get_guard(self, s: SimpleNamespace) -> Expression:
        """Return the signal above the threshold."""
        return self.signal(s) - s.threshold

    def _get_value(self, s: SimpleNamespace) -> float:
        """Return the value of the therapy state after the switch."""
        return self.value
