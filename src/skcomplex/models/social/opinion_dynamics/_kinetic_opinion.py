"""Opinion dynamics models: the kinetic theory of opinion formation over a discrete opinion grid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, Expression, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import Encounter, Law, Symbols, moment
from ....core.runtime import register_model
from ....core.space import ContinuousTime, LatticeSpace
from ....core.system import Observable, System

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
        >>> from skcomplex.models.social.opinion_dynamics import opinion_grid
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
        >>> from skcomplex.models.social.opinion_dynamics import persuasion_table
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


def mean_opinion(s: Symbols, opinions: npt.NDArray[np.float64]) -> Expression:
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
    return moment(s.f, 1, opinions, normalized=True)


def opinion_variance(s: Symbols, opinions: npt.NDArray[np.float64]) -> Expression:
    """Return the variance of the opinions of a distribution, which grows as opinions polarise.

    Args:
        s:
            The namespace of one member of the distribution.

        opinions:
            The opinion of every level.

    Returns:
        variance:
            The central second moment.
    """
    return moment(s.f, 2, opinions, central=True)


class OpinionLevel(Element):
    """Density of the population at one opinion level, a member of a distribution over opinions.

    Args:
        label:
            The label of the distribution.

        f:
            The initial density.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import OpinionLevel
        >>> OpinionLevel.group('opinions', 7, f=1 / 7).size
        7
    """

    f: float = state(default=0.0, domain=Domain.NONNEGATIVE)


class OpinionEncounters(Interaction):
    """Conservative encounters between opinion levels, which persuade close opinions and repel distant ones.

    Args:
        label:
            The label of the interaction.

        population:
            The path of the distribution over opinion levels.

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
        >>> from skcomplex.models.social.opinion_dynamics import OpinionEncounters
        >>> OpinionEncounters('encounters', population='opinions').confidence
        2
    """

    population: str = role()
    levels: int = parameter(default=7, domain=Domain.POSITIVE)
    confidence: int = parameter(default=2, domain=Domain.NONNEGATIVE)
    eta: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    persuasion: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    repulsion: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the encounter of the population with itself.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        table = persuasion_table(int(self.levels), int(self.confidence), s.persuasion, s.repulsion)
        return [Encounter(s.population.f, s.population.f, s.eta, table)]


def _levels_of(system: System) -> int:
    """Return the number of opinion levels of a model, the size of its distribution."""
    return int(next(item.size or 0 for item in system.components if item.label == 'opinions'))


def opinion_observables(levels: int) -> list[Observable]:
    """Return the mean opinion and the variance of the opinions of a distribution `opinions`.

    Args:
        levels:
            The number of opinion levels.

    Returns:
        observables:
            The observables, each reading the moments of the distribution.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import opinion_observables
        >>> [item.label for item in opinion_observables(7)]
        ['mean_opinion', 'opinion_variance']
    """
    opinions = opinion_grid(levels)
    return [
        Observable('mean_opinion', partial(mean_opinion, opinions=opinions), group='opinions'),
        Observable('opinion_variance', partial(opinion_variance, opinions=opinions), group='opinions'),
    ]


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'kinetic', 'distribution', 'polarization'),
    references=OPINION_REFERENCES,
    defaults={'until': 60, 'dt': 0.5},
    observables=lambda system: opinion_observables(_levels_of(system)),
)
def kinetic_opinion(
    levels: int = 7,
    *,
    confidence: int = 2,
    persuasion: float = 0.3,
    repulsion: float = 0.1,
    eta: float = 1.0,
) -> System:
    """Return the kinetic model of opinion formation over a discrete grid of opinions.

    The population is a distribution over opinion levels from -1 to 1, a site group on a lattice of the levels, which
    starts almost uniform with a slight lean toward positive opinions. Encounters between levels move an opinion one
    level toward a close opinion and one level away from a distant one, and keep the total density. With a wide
    confidence the opinions reach a consensus around the mean, and with a narrow confidence and repulsion they
    polarise into the two extremes.

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

    Returns:
        system:
            The model, with the distribution `opinions` and the encounters `encounters`. Its standard observables are
            the mean opinion and the variance of the opinions.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import kinetic_opinion
        >>> kinetic_opinion(levels=5).components[0].size
        5
    """
    initial = 1.0 + 0.1 * opinion_grid(levels)
    return System(
        components=[
            Group('opinions', element=OpinionLevel, size=levels, on='sites', values={'f': initial / initial.sum()}),
        ],
        interactions=[
            OpinionEncounters(
                'encounters',
                population='opinions',
                levels=levels,
                confidence=confidence,
                eta=eta,
                persuasion=persuasion,
                repulsion=repulsion,
            ),
        ],
        space=LatticeSpace(levels, boundary='fixed'),
        time=ContinuousTime(),
    )
