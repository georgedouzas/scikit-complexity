"""Oncology systems: the kinetic theory of active particles for the competition between tumour and immune cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import DensityElement, GroupElement
from ....core.components.interactions import EncounterInteraction
from ....core.components.laws.utils import moment
from ....core.components.spaces import ContinuousTime, LatticeSpace
from ....core.components.system import BaseSystem, RunSettings, System
from ....core.specs import ObservableSpec

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
        >>> from skcomplex.systems.biology.oncology import activity_grid
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
        >>> from skcomplex.systems.biology.oncology import step_table
        >>> step_table(3, -1)[:, 0].tolist()
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    """
    table = np.zeros((classes, fields, classes))
    for candidate in range(classes):
        table[candidate, :, min(max(candidate + step, 0), classes - 1)] = 1.0
    return table


def cell_density(s: SimpleNamespace, activities: npt.NDArray[np.float64]) -> Expression:
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


def mean_activity(s: SimpleNamespace, activities: npt.NDArray[np.float64]) -> Expression:
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


def _classes_of(system: System) -> int:
    """Return the number of activity classes of a system, the size of its tumour distribution."""
    return int(next(item.size or 0 for item in system.elements if item.label == 'tumour'))


def immune_observables(classes: int) -> list[ObservableSpec]:
    """Return the tumour density, the mean tumour progression, and the mean immune activation of the system.

    Args:
        classes:
            The number of activity classes.

    Returns:
        observables:
            The observables, each reading the moments of a distribution.

    Examples:
        >>> from skcomplex.systems.biology.oncology import immune_observables
        >>> [item.label for item in immune_observables(5)]
        ['tumour_density', 'tumour_progression', 'immune_activation']
    """
    activities = activity_grid(classes)
    return [
        ObservableSpec('tumour_density', partial(cell_density, activities=activities), group='tumour'),
        ObservableSpec('tumour_progression', partial(mean_activity, activities=activities), group='tumour'),
        ObservableSpec('immune_activation', partial(mean_activity, activities=activities), group='immune'),
    ]


class ImmuneCompetitionSystem(BaseSystem):
    """The kinetic system of the competition between tumour cells and immune cells.

    Tumour cells and immune cells are distributions over activity classes, the progression of tumour cells and the
    activation of immune cells, held by site groups on a lattice of the classes. Tumour cells proliferate in the host
    and progress, active immune cells destroy them, and tumour cells inhibit the activation of immune cells. The tumour
    starts in the lowest progression class and the immune cells in the class nearest `activation`. While `kill
    activation immune` exceeds `growth`, the immune response wins and the tumour density decays. Below that threshold
    the tumour grows, inhibits the immune cells further, and escapes. The system has the distributions `tumour` and
    `immune`, the host `host`, and the interactions `growth` and `competition`. Its standard observables are the tumour
    density, the mean tumour progression, and the mean immune activation.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import ImmuneCompetitionSystem
        >>> [item.label for item in ImmuneCompetitionSystem().prepare().system_.elements]
        ['tumour', 'immune', 'host']
    """

    field = 'biology/oncology'
    topics = ('tumour', 'immune', 'kinetic', 'distribution')
    references = IMMUNE_REFERENCES
    until = 40
    dt = 0.5

    def __init__(
        self,
        classes: int = 5,
        *,
        activation: float = 0.75,
        tumour: float = 0.05,
        immune: float = 1.0,
        growth: float = 0.3,
        progression: float = 0.05,
        kill: float = 1.0,
        inhibition: float = 0.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.classes = classes
        self.activation = activation
        self.tumour = tumour
        self.immune = immune
        self.growth = growth
        self.progression = progression
        self.kill = kill
        self.inhibition = inhibition

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings, with the tumour and immune observables of the built parts.

        Returns:
            settings:
                The tuned run settings by name.
        """
        settings = super().get_run_settings()
        settings['observables'] = immune_observables(_classes_of(self._build_system()))
        return settings

    def _build_system(self) -> System:
        """Return the parts of the kinetic system of the competition between tumour cells and immune cells."""
        activities = activity_grid(self.classes)
        activation_rates = np.tile(activities, (self.classes, 1))
        tumour_density = np.zeros(self.classes)
        tumour_density[0] = self.tumour
        immune_density = np.zeros(self.classes)
        immune_density[int(np.argmin(np.abs(activities - self.activation)))] = self.immune
        return System(
            self.label,
            elements=[
                GroupElement(
                    'tumour',
                    element=DensityElement,
                    size=self.classes,
                    on='sites',
                    values={'f': tumour_density},
                ),
                GroupElement(
                    'immune',
                    element=DensityElement,
                    size=self.classes,
                    on='sites',
                    values={'f': immune_density},
                ),
                GroupElement('host', element=DensityElement, size=1, values={'f': 1.0}),
            ],
            interactions=[
                EncounterInteraction(
                    'proliferation',
                    candidate='tumour',
                    field='host',
                    rate=lambda _: self.growth,
                    outcome='proliferative',
                ),
                EncounterInteraction(
                    'progression',
                    candidate='tumour',
                    field='host',
                    rate=lambda _: self.progression,
                    kernel=lambda _: step_table(self.classes, 1),
                ),
                EncounterInteraction(
                    'destruction',
                    candidate='tumour',
                    field='immune',
                    rate=lambda _: activation_rates * self.kill,
                    outcome='destructive',
                ),
                EncounterInteraction(
                    'inhibition',
                    candidate='immune',
                    field='tumour',
                    rate=lambda _: activation_rates * self.inhibition,
                    kernel=lambda _: step_table(self.classes, -1, self.classes),
                ),
            ],
            space=LatticeSpace(self.classes, boundary='fixed'),
            time=ContinuousTime(),
        )
