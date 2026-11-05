"""Binary alignment of self-propelled particles over a periodic orientation axis, by the mean angle of a pair."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.interactions import EncounterInteraction
from ....specs import RoleSpec

ANTIPODAL = 1e-9


class AngularAlignmentInteraction(EncounterInteraction):
    """Pairs of self-propelled particles leave a collision at their mean angle plus a wrapped normal noise.

    A pair of angles meets at the rate `4 d0 v0 |sin((a - b) / 2)|`, the frequency of Bertin, Droz, and Grégoire, with
    `d0` the interaction range and `v0` the speed. A pair of opposite angles leaves at either of its two mean angles
    with equal probability.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        candidate:
            The path of the field on a phase space with a periodic orientation axis.

        field:
            The path of the field the candidate meets, usually the candidate itself.

        angles:
            The number of cells of the orientation axis on `[0, 2 pi)`.

        frequency:
            The product `d0 v0` of the interaction range and the speed.

        noise:
            The standard deviation of the noise of the outcome.

        state:
            The name of the state of the density.

    Examples:
        >>> from skcomplex.models.physics.active_matter import AngularAlignmentInteraction
        >>> align = AngularAlignmentInteraction('align', candidate='flock', field='flock', frequency=0.5, noise=0.5)
        >>> align.outcome
        'conservative'
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        angles: int = 32,
        frequency: float = 1.0,
        noise: float = 0.5,
        state: str = 'f',
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._rate, kernel=self._kernel, state=state)
        self.angles = angles
        self.frequency = frequency
        self.noise = noise

    def _centres(self) -> npt.NDArray[np.float64]:
        """Return the angle of every cell."""
        return 2 * np.pi * (np.arange(self.angles) + 0.5) / self.angles

    def _rate(self, _: SimpleNamespace) -> npt.NDArray[np.float64]:
        """Return the rate of every pair of angles, times the width of the cell of the partner."""
        centres = self._centres()
        return 8 * np.pi / self.angles * self.frequency * np.abs(np.sin(np.subtract.outer(centres, centres) / 2))

    def _noise_weights(
        self,
        centres: npt.NDArray[np.float64],
        means: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the wrapped normal weight of every angle around the mean angle of every pair."""
        gaps = np.angle(np.exp(1j * (centres[None, None, :] - means[:, :, None])))
        return np.exp(-(gaps**2) / (2 * self.noise**2))

    def _kernel(self, _: SimpleNamespace) -> npt.NDArray[np.float64]:
        """Return the probability of every outcome angle of every pair of angles, around the mean angle of the pair."""
        centres = self._centres()
        cosines, sines = np.cos(centres), np.sin(centres)
        along, across = np.add.outer(cosines, cosines), np.add.outer(sines, sines)
        means = np.arctan2(across, along)
        opposite = np.hypot(along, across) < ANTIPODAL
        means = np.where(opposite, centres[:, None] + np.pi / 2, means)
        weights = self._noise_weights(centres, means)
        weights = np.where(opposite[:, :, None], (weights + self._noise_weights(centres, means + np.pi)) / 2, weights)
        return weights / weights.sum(axis=2, keepdims=True)
