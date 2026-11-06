"""Synchronous adoption of the threshold system, where every agent adopts for good from its social circle."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.interactions import TransitionInteraction
from ....core.components.types import SystemViewProtocol
from ....core.errors import ValidationError
from ....core.specs import RoleSpec
from ._adopter_view import AdopterView

MODES = ('infection', 'influence')


class AdoptionInteraction(TransitionInteraction):
    """Synchronous adoption of the threshold system, where every agent adopts for good from its social circle.

    In the infection mode an agent adopts once anyone in its social circle has adopted. In the influence mode it
    adopts once the percentage of adopters in its circle exceeds its threshold, or in society when its circle is empty.
    Every agent decides from the adoptions at the start of the step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of adopters.

        mode:
            `'infection'` or `'influence'`.

    Raises:
        ValidationError:
            The mode is unknown.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import AdoptionInteraction
        >>> AdoptionInteraction('adoption', target='people', mode='influence').mode
        'influence'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        mode: str = 'infection',
    ) -> None:
        super().__init__(
            label,
            target=target,
            state='adopted',
            source=0.0,
            destination=1.0,
            condition=self._adopts,
            form='agent',
        )
        self.mode = mode

    def _check(self) -> None:
        """Raise an error for an unknown mode.

        Raises:
            ValidationError:
                The mode is unknown.
        """
        super()._check()
        if self.mode not in MODES:
            error_msg = f'Unknown adoption mode {self.mode!r}. Use one of {", ".join(MODES)}.'
            raise ValidationError(error_msg)

    def _adopts(self, agent: AdopterView, system: SystemViewProtocol) -> bool:
        """Return whether an agent adopts from the adopters in its social circle, or in society without a circle."""
        circle = [item for item in agent.neighbors(agent.reach) if isinstance(item, AdopterView)]
        adopters = sum(item.adopted for item in circle)
        if self.mode == 'infection':
            return adopters > 0
        if circle:
            return 100 * adopters / len(circle) > agent.threshold
        society = [item for item in system.agents('target') if isinstance(item, AdopterView)]
        return 100 * sum(item.adopted for item in society) / len(society) > agent.threshold
