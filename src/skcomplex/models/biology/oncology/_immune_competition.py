"""Oncology models: the kinetic theory of active particles for the competition between tumour and immune cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, Expression, parameter, role, state
from ....core.components import Element, Group
from ....core.interactions import Interaction
from ....core.laws import Destruction, Encounter, Law, Proliferation, Symbols, moment
from ....core.runtime import register_model
from ....core.space import ContinuousTime, LatticeSpace
from ....core.system import Observable, System

IMMUNE_REFERENCES = ('Bellomo et al. 2017', 'Bellomo and Delitala 2008', 'Bellomo, Bellouquid, and Delitala 2004')


def activity_grid(classes: int) -> npt.NDArray[np.float64]:
    """Return the activity of every class, evenly spaced from 0 to 1.

    Args:
        classes:
            The number of activity classes.

    Returns:
        activities:
            The activity of every class.

    Examples:
        >>> from skcomplex.models.biology.oncology import activity_grid
        >>> activity_grid(5).tolist()
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    return np.linspace(0.0, 1.0, classes)


def step_table(classes: int, step: int, fields: int = 1) -> npt.NDArray[np.float64]:
    """Return the transition table that moves a candidate one class up or down, staying at the end classes.

    Args:
        classes:
            The number of activity classes of the candidate.

        step:
            `1` to move up and `-1` to move down.

        fields:
            The number of activity classes of the field.

    Returns:
        table:
            The transition table `A[h, k, i]`.

    Examples:
        >>> from skcomplex.models.biology.oncology import step_table
        >>> step_table(3, -1)[:, 0].tolist()
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    """
    table = np.zeros((classes, fields, classes))
    for candidate in range(classes):
        table[candidate, :, min(max(candidate + step, 0), classes - 1)] = 1.0
    return table


def cell_density(s: Symbols, activities: npt.NDArray[np.float64]) -> Expression:
    """Return the density of a population of cells, the order-0 moment of its distribution.

    Args:
        s:
            The namespace of one member of the distribution.

        activities:
            The activity of every class.

    Returns:
        density:
            The order-0 moment.
    """
    return moment(s.f, 0, activities)


def mean_activity(s: Symbols, activities: npt.NDArray[np.float64]) -> Expression:
    """Return the mean activity of a population of cells.

    Args:
        s:
            The namespace of one member of the distribution.

        activities:
            The activity of every class.

    Returns:
        mean:
            The normalized first moment.
    """
    return moment(s.f, 1, activities, normalized=True)


class Cells(Element):
    """Density of a population of cells at one activity class, a member of a distribution over activities.

    Args:
        label:
            The label of the distribution.

        f:
            The initial density.

    Examples:
        >>> from skcomplex.models.biology.oncology import Cells
        >>> Cells.group('tumour', 5, f=0.02).size
        5
    """

    f: float = state(default=0.0, domain=Domain.NONNEGATIVE)


class TumourGrowth(Interaction):
    """Proliferation of tumour cells in the host environment, and their progression to higher activity classes.

    Args:
        label:
            The label of the interaction.

        tumour:
            The path of the distribution of tumour cells over their progression.

        host:
            The path of the group of one member whose fixed density is the host environment.

        classes:
            The number of progression classes.

        growth:
            The proliferation rate.

        progression:
            The rate at which tumour cells move one progression class up.

    Examples:
        >>> from skcomplex.models.biology.oncology import TumourGrowth
        >>> TumourGrowth('growth', tumour='tumour', host='host').growth
        0.3
    """

    tumour: str = role()
    host: str = role()
    classes: int = parameter(default=5, domain=Domain.POSITIVE)
    growth: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    progression: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the proliferation and the progression.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            Proliferation(s.tumour.f, s.host.f, s.growth),
            Encounter(s.tumour.f, s.host.f, s.progression, step_table(int(self.classes), 1)),
        ]


class ImmuneCompetition(Interaction):
    """Encounters of tumour and immune cells: active immune cells destroy tumour cells, which inhibit immune cells.

    A tumour cell meets an immune cell of activation `v_k` and is destroyed at the rate `kill v_k`. An immune cell
    meets a tumour cell of progression `u_k` and loses one activation class at the rate `inhibition u_k`.

    Args:
        label:
            The label of the interaction.

        tumour:
            The path of the distribution of tumour cells over their progression.

        immune:
            The path of the distribution of immune cells over their activation.

        classes:
            The number of activity classes of both distributions.

        kill:
            The rate at which fully active immune cells destroy tumour cells.

        inhibition:
            The rate at which the most progressed tumour cells inhibit immune cells.

    Examples:
        >>> from skcomplex.models.biology.oncology import ImmuneCompetition
        >>> ImmuneCompetition('competition', tumour='tumour', immune='immune').kill
        1.0
    """

    tumour: str = role()
    immune: str = role()
    classes: int = parameter(default=5, domain=Domain.POSITIVE)
    kill: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    inhibition: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the destruction of tumour cells and the inhibition of immune cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        classes = int(self.classes)
        activities = activity_grid(classes)
        kill = np.tile(activities, (classes, 1))
        return [
            Destruction(s.tumour.f, s.immune.f, kill * s.kill),
            Encounter(s.immune.f, s.tumour.f, kill * s.inhibition, step_table(classes, -1, classes)),
        ]


def _classes_of(system: System) -> int:
    """Return the number of activity classes of a model, the size of its tumour distribution."""
    return int(next(item.size or 0 for item in system.components if item.label == 'tumour'))


def immune_observables(classes: int) -> list[Observable]:
    """Return the tumour density, the mean tumour progression, and the mean immune activation of the model.

    Args:
        classes:
            The number of activity classes.

    Returns:
        observables:
            The observables, each reading the moments of a distribution.

    Examples:
        >>> from skcomplex.models.biology.oncology import immune_observables
        >>> [item.label for item in immune_observables(5)]
        ['tumour_density', 'tumour_progression', 'immune_activation']
    """
    activities = activity_grid(classes)
    return [
        Observable('tumour_density', partial(cell_density, activities=activities), group='tumour'),
        Observable('tumour_progression', partial(mean_activity, activities=activities), group='tumour'),
        Observable('immune_activation', partial(mean_activity, activities=activities), group='immune'),
    ]


@register_model(
    field='biology/oncology',
    tags=('tumour', 'immune', 'kinetic', 'distribution'),
    references=IMMUNE_REFERENCES,
    defaults={'until': 40, 'dt': 0.5},
    observables=lambda system: immune_observables(_classes_of(system)),
)
def immune_competition(
    classes: int = 5,
    *,
    activation: float = 0.75,
    tumour: float = 0.05,
    immune: float = 1.0,
    growth: float = 0.3,
    progression: float = 0.05,
    kill: float = 1.0,
    inhibition: float = 0.2,
) -> System:
    """Return the kinetic model of the competition between tumour cells and immune cells.

    Tumour cells and immune cells are distributions over activity classes, the progression of tumour cells and the
    activation of immune cells, held by site groups on a lattice of the classes. Tumour cells proliferate in the host
    and progress, active immune cells destroy them, and tumour cells inhibit the activation of immune cells. The
    tumour starts in the lowest progression class and the immune cells in the class nearest `activation`. While
    `kill activation immune` exceeds `growth`, the immune response wins and the tumour density decays. Below that
    threshold the tumour grows, inhibits the immune cells further, and escapes.

    Args:
        classes:
            The number of activity classes of both distributions.

        activation:
            The initial activation of the immune cells, between 0 and 1.

        tumour:
            The initial density of tumour cells.

        immune:
            The density of immune cells.

        growth:
            The proliferation rate of tumour cells.

        progression:
            The rate at which tumour cells progress one class.

        kill:
            The rate at which fully active immune cells destroy tumour cells.

        inhibition:
            The rate at which the most progressed tumour cells inhibit immune cells.

    Returns:
        system:
            The model, with the distributions `tumour` and `immune`, the host `host`, and the interactions `growth`
            and `competition`. Its standard observables are the tumour density, the mean tumour progression, and the
            mean immune activation.

    Examples:
        >>> from skcomplex.models.biology.oncology import immune_competition
        >>> [item.label for item in immune_competition().components]
        ['tumour', 'immune', 'host']
    """
    activities = activity_grid(classes)
    tumour_density = np.zeros(classes)
    tumour_density[0] = tumour
    immune_density = np.zeros(classes)
    immune_density[int(np.argmin(np.abs(activities - activation)))] = immune
    return System(
        components=[
            Group('tumour', element=Cells, size=classes, on='sites', values={'f': tumour_density}),
            Group('immune', element=Cells, size=classes, on='sites', values={'f': immune_density}),
            Cells.group('host', 1, f=1.0),
        ],
        interactions=[
            TumourGrowth(
                'growth',
                tumour='tumour',
                host='host',
                classes=classes,
                growth=growth,
                progression=progression,
            ),
            ImmuneCompetition(
                'competition',
                tumour='tumour',
                immune='immune',
                classes=classes,
                kill=kill,
                inhibition=inhibition,
            ),
        ],
        space=LatticeSpace(classes, boundary='fixed'),
        time=ContinuousTime(),
    )
