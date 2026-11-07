"""Kinetic theory of flocking of Bertin, Droz, and Grégoire over position and orientation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, cos, sin

from ....core.components.elements import DensityElement, FieldElement
from ....core.components.interactions import AdvectionInteraction, EncounterInteraction
from ....core.components.laws.utils import coordinate
from ....core.components.spaces import LatticeSpace, PhaseSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.specs import AxisSpec, RoleSpec
from ..._catalog import register_system

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
        >>> from skcomplex.systems.physics.active_matter import AngularAlignmentInteraction
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
        >>> from skcomplex.systems.physics.active_matter import AngularDiffusionInteraction
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


@register_system(
    field='physics/active_matter',
    tags=('field', 'kinetic', 'phase space', 'flocking', 'pde'),
    references=('Bertin, Droz, and Grégoire 2006', 'Bertin, Droz, and Grégoire 2009'),
    defaults={'until': 0.5, 'dt': 0.1, 'solver': RK4Solver(dt=0.05)},
)
def bertin_droz_gregoire(
    positions: int = 4,
    angles: int = 16,
    *,
    density: float = 1.0,
    speed: float = 1.0,
    interaction_range: float = 0.5,
    tumbling: float = 1.0,
    noise: float = 0.5,
    perturbation: float = 0.01,
) -> System:
    """Return the kinetic theory of flocking, whose polar order grows above a transition density.

    Particles stream along their orientation, tumble at a rate by a wrapped normal noise, and align in binary collisions
    at their mean angle with the same noise. A homogeneous disordered state is unstable above the density
    `pi tumbling (1 - exp(-noise^2 / 2)) / (8 interaction_range speed (exp(-noise^2 / 2) - 2 / 3))`.

    Args:
        positions:
            The number of positions along each of the two periodic position axes of unit spacing.

        angles:
            The number of cells of the orientation axis.

        density:
            The density of the homogeneous state.

        speed:
            The speed of the particles.

        interaction_range:
            The interaction range.

        tumbling:
            The rate of the tumbling events.

        noise:
            The standard deviation of the noise of tumbling and alignment.

        perturbation:
            The relative amplitude of the initial polar perturbation.

    Returns:
        system:
            The system, with the density `flock.f`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import bertin_droz_gregoire
        >>> bertin_droz_gregoire(2, 8).elements[0].prepare().size_
        32
    """
    axis = AxisSpec('theta', angles, 0.0, 2 * math.pi, periodic=True)
    space = PhaseSpace(LatticeSpace((positions, positions)), [axis]).prepare()
    theta = space.coordinates_['theta']
    profile = density * (1.0 + perturbation * np.cos(theta)) / (2 * math.pi)
    values = np.broadcast_to(profile, space.shape_).copy()
    flock = FieldElement(
        'flock',
        element=DensityElement,
        space=space,
        values={'f': values},
        discretization='finite-volume',
    )

    def drift(_: SimpleNamespace) -> list[Expression]:
        return [speed * cos(coordinate('theta')), speed * sin(coordinate('theta')), 0]

    return System(
        elements=[flock],
        interactions=[
            AdvectionInteraction('stream', target='flock', state='f', drift=drift),
            AngularDiffusionInteraction('tumble', candidate='flock', angles=angles, tumbling=tumbling, noise=noise),
            AngularAlignmentInteraction(
                'align',
                candidate='flock',
                field='flock',
                angles=angles,
                frequency=interaction_range * speed,
                noise=noise,
            ),
        ],
    )
