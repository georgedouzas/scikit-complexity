"""Opinion dynamics systems: the kinetic theory of opinion formation.

The opinions lie on a discrete grid.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement
from ....core.components.laws.utils import moment
from ....core.components.spaces import ContinuousTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec
from ...general.kinetics import EncounterInteraction
from ..common.kinetics import ActivityDensityElement

OPINION_REFERENCES = ('Bellomo et al. 2017', 'Toscani 2006', 'Deffuant et al. 2000')


def opinion_grid(levels: int) -> npt.NDArray[np.float64]:
    """Return the opinion of every level, evenly spaced from -1 to 1.

    Args:
        levels:
            The number of opinion levels.

    Returns:
        opinions:
            The opinion of every level.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import opinion_grid
        >>> opinion_grid(5).tolist()
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    """
    return np.linspace(-1.0, 1.0, levels)


def persuasion_table(levels: int, confidence: int, persuasion: object, repulsion: object) -> npt.NDArray[np.object_]:
    """Return the transition table of encounters between opinion levels, `A[h, k, i]`.

    A candidate at level `h` that meets a field particle at level `k` within `confidence` levels moves one level
    toward it with probability `persuasion`. One that meets a field particle further away moves one level away from it
    with probability `repulsion`, unless it holds an extreme opinion already.

    Args:
        levels:
            The number of opinion levels.

        confidence:
            The largest distance in levels at which encounters persuade.

        persuasion:
            The probability of moving toward a close opinion, a number or a parameter.

        repulsion:
            The probability of moving away from a distant opinion, a number or a parameter.

    Returns:
        table:
            The transition probabilities, which sum to one over the outcomes.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import persuasion_table
        >>> persuasion_table(3, 1, 0.4, 0.1)[0, 1].astype(float).tolist()
        [0.6, 0.4, 0.0]
    """
    table = np.zeros((levels, levels, levels), dtype=object)
    for candidate in range(levels):
        for field in range(levels):
            toward = int(np.sign(field - candidate))
            outcome, probability = (
                (candidate + toward, persuasion)
                if abs(field - candidate) <= confidence
                else (candidate - toward, repulsion)
            )
            if toward and 0 <= outcome < levels:
                table[candidate, field, outcome] += probability
            table[candidate, field, candidate] += 1 - table[candidate, field].sum()
    return table


def mean_opinion(s: SimpleNamespace, opinions: npt.NDArray[np.float64]) -> Expression:
    """Return the mean opinion of a distribution over opinion levels.

    Args:
        s:
            The namespace of one member of the distribution.

        opinions:
            The opinion of every level.

    Returns:
        mean:
            The normalized first moment.
    """
    return moment(s.x, 1, opinions, normalized=True)


def opinion_variance(s: SimpleNamespace, opinions: npt.NDArray[np.float64]) -> Expression:
    """Return the variance of the opinions of a distribution.

    It grows as opinions polarise.

    Args:
        s:
            The namespace of one member of the distribution.

        opinions:
            The opinion of every level.

    Returns:
        variance:
            The central second moment.
    """
    return moment(s.x, 2, opinions, central=True)


class OpinionEncountersInteraction(EncounterInteraction):
    """Conservative encounters between opinion levels.

    They persuade close opinions and repel distant ones.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the distribution over opinion levels whose members move.

        field:
            The path of the distribution over opinion levels they meet, usually the same distribution.

        levels:
            The number of opinion levels.

        confidence:
            The largest distance in levels at which encounters persuade.

        eta:
            The encounter rate.

        persuasion:
            The probability of moving toward a close opinion.

        repulsion:
            The probability of moving away from a distant opinion.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import OpinionEncountersInteraction
        >>> OpinionEncountersInteraction('encounters', candidate='opinions', field='opinions').confidence
        2
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        levels: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 7,
        confidence: Annotated[int, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        persuasion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        repulsion: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.levels = levels
        self.confidence = confidence
        self.eta = eta
        self.persuasion = persuasion
        self.repulsion = repulsion

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the encounter rate."""
        return s.eta

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table of the persuasion and repulsion of the opinion levels."""
        return persuasion_table(int(self.levels), int(self.confidence), s.persuasion, s.repulsion)


def _levels_of(system: CompositeSystem) -> int:
    """Return the number of opinion levels of a system, the size of its distribution."""
    return int(next(item.size or 0 for item in system.elements if item.label == 'opinions'))


def opinion_observables(levels: int) -> list[ObservableSpec]:
    """Return the mean opinion and the variance of the opinions of a distribution.

    The distribution is `opinions`.

    Args:
        levels:
            The number of opinion levels.

    Returns:
        observables:
            The observables, each reading the moments of the distribution.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import opinion_observables
        >>> [item.label for item in opinion_observables(7)]
        ['mean_opinion', 'opinion_variance']
    """
    opinions = opinion_grid(levels)
    return [
        ObservableSpec('mean_opinion', partial(mean_opinion, opinions=opinions), group='opinions'),
        ObservableSpec('opinion_variance', partial(opinion_variance, opinions=opinions), group='opinions'),
    ]


class KineticOpinionSystem(BaseSystem):
    """The kinetic system of opinion formation over a discrete grid of opinions.

    The population is a distribution over opinion levels from -1 to 1, a site group on a lattice of the levels, which
    starts almost uniform with a slight lean toward positive opinions. Encounters between levels move an opinion one
    level toward a close opinion and one level away from a distant one, and keep the total density. With a wide
    confidence the opinions reach a consensus around the mean, and with a narrow confidence and repulsion they
    polarise into the two extremes.

    The system has the distribution `opinions` and the encounters `encounters`. Its standard observables are the mean
    opinion and the variance of the opinions.

    Args:
        levels:
            The number of opinion levels.

        confidence:
            The largest distance in levels at which encounters persuade.

        persuasion:
            The probability of moving toward a close opinion.

        repulsion:
            The probability of moving away from a distant opinion.

        eta:
            The encounter rate.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import KineticOpinionSystem
        >>> KineticOpinionSystem(levels=5).prepare().system_.elements[0].size
        5
    """

    field = 'social/opinion_dynamics'
    topics = ('opinion', 'kinetic', 'distribution', 'polarization')
    references = OPINION_REFERENCES
    until = 60
    dt = 0.5

    def __init__(
        self,
        levels: int = 7,
        *,
        confidence: int = 2,
        persuasion: float = 0.3,
        repulsion: float = 0.1,
        eta: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.levels = levels
        self.confidence = confidence
        self.persuasion = persuasion
        self.repulsion = repulsion
        self.eta = eta

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        initial = 1.0 + 0.1 * opinion_grid(self.levels)
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'opinions',
                    element=ActivityDensityElement,
                    size=self.levels,
                    on='sites',
                    values={'x': initial / initial.sum()},
                ),
            ],
            interactions=[
                OpinionEncountersInteraction(
                    'encounters',
                    candidate='opinions',
                    field='opinions',
                    levels=self.levels,
                    confidence=self.confidence,
                    eta=self.eta,
                    persuasion=self.persuasion,
                    repulsion=self.repulsion,
                ),
            ],
            space=LatticeSpace(self.levels, boundary='fixed'),
            time=ContinuousTime(),
        )

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        system = self._get_pass_parameters()
        settings['observables'] = opinion_observables(_levels_of(system))
        return settings
