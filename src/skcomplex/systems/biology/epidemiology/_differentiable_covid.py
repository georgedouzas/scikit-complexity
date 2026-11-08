"""Differentiable agents of a COVID-19 epidemic in households, schools, and companies, after Quera-Bofarull et al."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import exp, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli, total
from ....core.components.spaces import DiscreteTime
from ....core.components.system import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec

BINARY = FiniteEnumeratedSet([0, 1])
HOUSEHOLD_ADULTS = 2
HOUSEHOLD_CHILDREN = 2


class CovidAgentElement(Element):
    """Agent of the differentiable COVID-19 model, susceptible, infected, or recovered.

    Args:
        label:
            The label of the agent or the group.

        s:
            1 for a susceptible agent.

        i:
            1 for an infected agent.

        r:
            1 for a recovered agent.

        new:
            1 for an agent infected in the last step.

        beta_household:
            The transmission of households.

        beta_school:
            The transmission of schools.

        beta_company:
            The transmission of companies.

        psi:
            The susceptibility of the agent.

        gamma:
            The probability per day that an infected agent recovers.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import CovidAgentElement
        >>> CovidAgentElement('agent', i=1).i
        1
    """

    def __init__(
        self,
        label: str,
        *,
        s: Annotated[float, StateSpec(domain=BINARY)] = 1,
        i: Annotated[float, StateSpec(domain=BINARY)] = 0,
        r: Annotated[float, StateSpec(domain=BINARY)] = 0,
        new: Annotated[float, StateSpec(domain=BINARY)] = 0,
        beta_household: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        beta_school: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        beta_company: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        psi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.2,
    ) -> None:
        super().__init__(label)
        self.s = s
        self.i = i
        self.r = r
        self.new = new
        self.beta_household = beta_household
        self.beta_school = beta_school
        self.beta_company = beta_company
        self.psi = psi
        self.gamma = gamma


class CaseCountElement(Element):
    """Daily infections of children and of adults, counted the step after they happen.

    Args:
        label:
            The label of the element.

        children:
            The children infected in the step before.

        adults:
            The adults infected in the step before.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import CaseCountElement
        >>> CaseCountElement('cases').children
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        children: Annotated[float, StateSpec()] = 0.0,
        adults: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.children = children
        self.adults = adults


class CovidTransmissionInteraction(Interaction):
    """Infection and recovery of agents by relaxed Bernoulli draws, Eq. 6 of Quera-Bofarull et al. (2023).

    A susceptible agent is infected with the probability `1 - exp(-psi sum_L beta_L I_L)`, where `I_L` is the share of
    infected agents among the other members of its place of kind `L`, household, school, or company, read through a
    weighted sum with the weights `1 / (n_L - 1)` of co-members. An infected agent recovers with the probability
    `gamma`. Both are Bernoulli draws with a temperature, which every backend but JAX draws exactly, and JAX relaxes so
    runs have gradients in the transmissions. The interaction declares its draws reparameterized.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of agents.

        contacts:
            The weights of the co-members of every agent at its places, one matrix per kind of place.

        temperature:
            The temperature of the relaxed draws on JAX.

        hard:
            Whether JAX runs draw exact agents with the gradients of the relaxation.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.biology.epidemiology import CovidTransmissionInteraction
        >>> CovidTransmissionInteraction('spread', target='people', contacts=(np.eye(2),), temperature=0.1).temperature
        0.1
    """

    law_classes = (UpdateLaw,)
    draws = True
    reparameterized = True

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        contacts: tuple[scipy.sparse.csr_array | npt.NDArray[np.float64], ...],
        temperature: float = 0.1,
        hard: bool = False,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.contacts = contacts
        self.temperature = temperature
        self.hard = hard

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the updates of the states of every agent."""
        s = namespace.target
        betas = (s.beta_household, s.beta_school, s.beta_company)
        pressure = sum(
            beta * total(s.other.i, weights=weights) for beta, weights in zip(betas, self.contacts, strict=True)
        )
        infected = bernoulli('infect', 1 - exp(-s.psi * pressure), temperature=self.temperature, hard=self.hard)
        recovered = bernoulli('recover', s.gamma, temperature=self.temperature, hard=self.hard)
        return [
            UpdateLaw(s.s, s.s - s.s * infected),
            UpdateLaw(s.i, s.i + s.s * infected - s.i * recovered),
            UpdateLaw(s.r, s.r + s.i * recovered),
            UpdateLaw(s.new, s.s * infected),
        ]


class CaseCountInteraction(Interaction):
    """Count of the agents infected in the last step, among children and among adults.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the counts.

        source:
            The path of the group of agents.

        children:
            1 for every agent that is a child, 0 for an adult.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.biology.epidemiology import CaseCountInteraction
        >>> CaseCountInteraction('count', target='cases', source='people', children=np.ones(2)).source
        'people'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        children: npt.NDArray[np.float64],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.children = children

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the counts of the new infections of children and of adults."""
        cases, people = namespace.target, namespace.source
        children = np.asarray(self.children, dtype=float)
        return [
            UpdateLaw(cases.children, total(people.new, over=people, weights=children)),
            UpdateLaw(cases.adults, total(people.new, over=people, weights=1.0 - children)),
        ]


class DifferentiableCovidAbmSystem(BaseSystem):
    """Differentiable agents of a COVID-19 epidemic, after Quera-Bofarull et al. (2023).

    The model of Quera-Bofarull et al. (2023, arXiv 2305.15340), a small version of GradABM-JUNE: a seeded synthetic
    population of households of two adults and two children, where children go to schools and adults to companies.
    Agents are susceptible, infected, or recovered, and are infected and recover by the relaxed Bernoulli draws of
    `CovidTransmissionInteraction`, Eq. 6, with a transmission `beta_L` per kind of place. NumPy runs draw exact agents
    and JAX runs relax the draws, so the system holds the tag `differentiable` and variational calibration fits the
    three transmissions through its JAX run. Time runs in days. The system has the group `people`, the element `cases`
    with the infections of children and adults of the day before, and the observable `daily_infections`.

    Args:
        n_households:
            The number of households.

        n_schools:
            The number of schools.

        n_companies:
            The number of companies.

        beta_household:
            The transmission of households.

        beta_school:
            The transmission of schools.

        beta_company:
            The transmission of companies.

        gamma:
            The probability per day that an infected agent recovers.

        initial_infected:
            The number of agents infected at the start.

        temperature:
            The temperature of the relaxed draws on JAX.

        hard:
            Whether JAX runs draw exact agents with the gradients of the relaxation.

        random_state:
            The seed or generator of the population and of the initial infections.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import DifferentiableCovidAbmSystem
        >>> DifferentiableCovidAbmSystem(n_households=10).get_tags().differentiable
        True
    """

    field = 'biology/epidemiology'
    topics = ('agents', 'covid-19', 'differentiable', 'calibration', 'households')
    references = ('Quera-Bofarull et al. 2023', 'Aylett-Bullock et al. 2021', 'Chopra et al. 2023')
    until = 30
    observables = (ObservableSpec('daily_infections', group='people', variable='new', reduce='sum'),)

    def __init__(
        self,
        n_households: int = 100,
        *,
        n_schools: int = 4,
        n_companies: int = 10,
        beta_household: float = 0.9,
        beta_school: float = 0.6,
        beta_company: float = 0.3,
        gamma: float = 0.2,
        initial_infected: int = 10,
        temperature: float = 0.1,
        hard: bool = False,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_households = n_households
        self.n_schools = n_schools
        self.n_companies = n_companies
        self.beta_household = beta_household
        self.beta_school = beta_school
        self.beta_company = beta_company
        self.gamma = gamma
        self.initial_infected = initial_infected
        self.temperature = temperature
        self.hard = hard
        self.random_state = random_state

    @staticmethod
    def _contacts(places: npt.NDArray[np.intp]) -> scipy.sparse.csr_array:
        """Return the weights `1 / (n - 1)` of the other members of the place of every agent, 0 for no place."""
        members = places >= 0
        same = (places[:, None] == places[None, :]) & members[:, None] & members[None, :]
        np.fill_diagonal(same, False)
        sizes = same.sum(axis=1, keepdims=True)
        return scipy.sparse.csr_array(np.where(same, 1.0 / np.maximum(sizes, 1), 0.0))

    def _build_system(self) -> System:
        """Return the agents of the synthetic population with their places and the case counts."""
        rng = np.random.default_rng(self.random_state)
        size = HOUSEHOLD_ADULTS + HOUSEHOLD_CHILDREN
        n_agents = self.n_households * size
        children = np.tile(np.arange(size) >= HOUSEHOLD_ADULTS, self.n_households)
        households = np.repeat(np.arange(self.n_households), size)
        schools = np.where(children, rng.integers(self.n_schools, size=n_agents), -1)
        companies = np.where(children, -1, rng.integers(self.n_companies, size=n_agents))
        infected = np.zeros(n_agents)
        infected[rng.choice(n_agents, size=self.initial_infected, replace=False)] = 1.0
        people = GroupElement(
            'people',
            element=CovidAgentElement,
            size=n_agents,
            values={
                's': 1.0 - infected,
                'i': infected,
                'beta_household': self.beta_household,
                'beta_school': self.beta_school,
                'beta_company': self.beta_company,
                'gamma': self.gamma,
            },
        )
        contacts = tuple(self._contacts(places) for places in (households, schools, companies))
        spread = CovidTransmissionInteraction(
            'spread',
            target='people',
            contacts=contacts,
            temperature=self.temperature,
            hard=self.hard,
        )
        count = CaseCountInteraction('count', target='cases', source='people', children=children.astype(float))
        return System(
            self.label,
            elements=[people, CaseCountElement('cases')],
            interactions=[spread, count],
            time=DiscreteTime(),
        )
