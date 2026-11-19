"""Epidemiology systems: outbreaks by record dynamics.

An outbreak strikes only when a strain beats every earlier one.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import uniform, where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role


class PathogenElement(Element):
    """Pathogen whose strength `StrainDrawInteraction` draws anew every season.

    Args:
        label:
            The label of the pathogen.

        eta:
            The strength of the strain of the current season.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import PathogenElement
        >>> PathogenElement('pathogen').eta
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        eta: Annotated[object, RealSet.closed_open(0, 1)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.eta = eta


class StrainDrawInteraction(Interaction):
    """Draw of the strength of the strain of every season, uniform from zero to one.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the pathogen.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import StrainDrawInteraction
        >>> StrainDrawInteraction('strain', target='pathogen').target
        'pathogen'
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the strength of the next season, a uniform draw.

        Args:
            context:
                The namespace of the pathogen.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseLaw] = [UpdateLaw(context.target.eta, uniform('eta'))]


class ResidentElement(Element):
    """Resident of a population with a fixed resistance to a disease.

    The resident is immune for good once infected.

    Args:
        label:
            The label of the resident or of the group of residents.

        R:
            The resistance, between 0 and 1, where 1 is immunity.

        infected:
            Whether the resident was infected in the last outbreak.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ResidentElement
        >>> ResidentElement('resident', R=0.3).R
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        R: Annotated[float | Array, RealSet.closed(0, 1)] = 0.5,
        infected: Annotated[float | Array, FiniteEnumeratedSet([0, 1])] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
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
        residents: Role,
        pathogen: Role,
    ) -> None:
        super().__init__(label)
        self.residents = residents
        self.pathogen = pathogen

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the infection and the immunity of the residents whose resistance is below the strength of the strain.

        Args:
            context:
                The namespace of the residents and the pathogen.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        resident, strength = context.residents, context.pathogen.eta
        susceptible = strength > resident.R
        self.laws_: list[BaseLaw] = [
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import RecordDynamicsEpidemicSystem
        >>> RecordDynamicsEpidemicSystem(100).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of yearly strains that infect less resistant residents.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        resistance = np.random.default_rng(self.seed).random(self.n)
        residents = ResidentElement('residents', R=resistance, size=self.n)
        self.system_ = System(
            self.label,
            elements=[residents, PathogenElement('pathogen')],
            interactions=[
                StrainDrawInteraction('strain', target='pathogen'),
                ResistanceInfectionInteraction('infection', residents='residents', pathogen='pathogen'),
            ],
        )
