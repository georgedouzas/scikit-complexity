"""Epidemiology systems: outbreaks by record dynamics.

An outbreak strikes only when a strain beats every earlier one.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import uniform, where
from ....core.components.spaces import DiscreteTime
from ....core.components.system import BaseSystem, System
from ....core.specs import RoleSpec, StateSpec, VisualSpec


class PathogenElement(Element):
    """Pathogen whose strength is drawn anew every season.

    The strength is drawn uniformly from zero to one.

    Args:
        label:
            The label of the pathogen.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        eta:
            The strength of the strain of the current season.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import PathogenElement
        >>> PathogenElement('pathogen').eta
        0.0
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        eta: Annotated[float, StateSpec(domain=RealSet.closed_open(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.eta = eta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the strength of the next season, a uniform draw.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [UpdateLaw(s.eta, uniform('eta'))]


class ResidentElement(Element):
    """Resident of a population with a fixed resistance to a disease.

    The resident is immune for good once infected.

    Args:
        label:
            The label of the resident or of the group of residents.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        R:
            The resistance, between 0 and 1, where 1 is immunity.

        infected:
            Whether the resident was infected in the last outbreak.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ResidentElement
        >>> ResidentElement('resident', R=0.3).R
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        R: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.5,
        infected: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.R = R
        self.infected = infected


class ResistanceInfectionInteraction(Interaction):
    """Infection of every resident whose resistance is below the strain strength.

    The strain is the one of the season, and an infected resident becomes immune.

    A resident with `R < eta` is infected in the season and its resistance becomes 1, so it is never infected again.

    Args:
        label:
            The label of the interaction.

        residents:
            The path of the group of residents.

        pathogen:
            The path of the pathogen.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ResistanceInfectionInteraction
        >>> ResistanceInfectionInteraction('infection', residents='people', pathogen='flu').pathogen
        'flu'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        residents: Annotated[str, RoleSpec()],
        pathogen: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.residents = residents
        self.pathogen = pathogen

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infection and the immunity of the residents below the strain.

        They are the residents whose resistance is below the strength of the strain.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        resident, strength = s.residents, s.pathogen.eta
        susceptible = strength > resident.R
        return [
            UpdateLaw(resident.R, where(susceptible, 1, resident.R)),
            UpdateLaw(resident.infected, where(susceptible, 1, 0)),
        ]


class RecordDynamicsEpidemicSystem(BaseSystem):
    """Yearly strains that infect every resident less resistant than their strength.

    The infected residents become immune.

    Every resident keeps a resistance `R` drawn uniformly from `[0, 1]`. Once a year a strain of strength `eta`, drawn
    uniformly from `[0, 1)`, infects the residents with `R < eta`, whose resistance becomes 1. In a large population an
    outbreak happens only in a year whose strain is stronger than every earlier one, a record, so outbreaks thin out as
    time goes on: the mean number of outbreaks in `t` years is close to the harmonic number `H_t`, about `ln t + 0.577`,
    a Poisson process in the logarithm of time. The system has the resistances `residents.R`, the infections of the last
    year `residents.infected`, and the strength `pathogen.eta` of the strain of the coming year.

    Args:
        n:
            The number of residents.

        seed:
            The seed of the resistances.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import RecordDynamicsEpidemicSystem
        >>> RecordDynamicsEpidemicSystem(100).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'biology/epidemiology'
    topics = ('record dynamics', 'intermittency', 'immunity', 'stochastic map')
    references = ('Jensen 2023',)
    until = 1000
    dt = 1

    def __init__(
        self,
        n: int = 10_000,
        *,
        seed: int | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.seed = seed

    def _build_system(self) -> System:
        """Return the parts of yearly strains that infect less resistant residents.

        A strain infects every resident less resistant than its strength.
        """
        resistance = np.random.default_rng(self.seed).random(self.n)
        residents = GroupElement('residents', element=ResidentElement, size=self.n, values={'R': resistance})
        return System(
            self.label,
            elements=[residents, PathogenElement('pathogen')],
            interactions=[ResistanceInfectionInteraction('infection', residents='residents', pathogen='pathogen')],
            time=DiscreteTime(),
        )
