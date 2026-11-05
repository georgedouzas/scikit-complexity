"""Angular diffusion of self-propelled particles, tumbling events over a periodic orientation axis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.interactions import EncounterInteraction
from ....specs import RoleSpec


class AngularDiffusionInteraction(EncounterInteraction):
    """Self-propelled particles change their angle at a rate by a wrapped normal noise, a linear redistribution.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        candidate:
            The path of the field on a phase space with a periodic orientation axis.

        angles:
            The number of cells of the orientation axis on `[0, 2 pi)`.

        tumbling:
            The rate of the events.

        noise:
            The standard deviation of the noise of an event.

        state:
            The name of the state of the density.

    Examples:
        >>> from skcomplex.models.physics.active_matter import AngularDiffusionInteraction
        >>> AngularDiffusionInteraction('tumble', candidate='flock', angles=8, tumbling=1.0, noise=0.5).field
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        angles: int = 32,
        tumbling: float = 1.0,
        noise: float = 0.5,
        state: str = 'f',
    ) -> None:
        super().__init__(label, candidate=candidate, rate=self._rate, kernel=self._kernel, state=state)
        self.angles = angles
        self.tumbling = tumbling
        self.noise = noise

    def _rate(self, _: SimpleNamespace) -> float:
        """Return the rate of the events."""
        return self.tumbling

    def _kernel(self, _: SimpleNamespace) -> npt.NDArray[np.float64]:
        """Return the probability of every angle after an event at every angle."""
        centres = 2 * np.pi * (np.arange(self.angles) + 0.5) / self.angles
        gaps = np.angle(np.exp(1j * np.subtract.outer(centres, centres)))
        weights = np.exp(-(gaps**2) / (2 * self.noise**2))
        return weights / weights.sum(axis=1, keepdims=True)
