"""Bounded confidence models: compromise between agents whose opinions are close."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.model import AgentType, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import ModelView


class Discussant(AgentType):
    """Agent of the Deffuant model, with a continuous opinion and a confidence bound.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        opinion:
            The initial opinion, between 0 and 1.

        confidence:
            The largest opinion difference at which two agents still compromise.

        convergence:
            The fraction of the opinion difference by which each agent of a compromising pair moves.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Discussant
        >>> Discussant.group('people', 10, opinion=0.5)
        Group('people', element=Discussant, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)})
    opinion: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    confidence: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    convergence: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)


class Encounters(Interaction):
    """Random binary encounters of the Deffuant model, as many per step as there are agents.

    In every encounter two random agents whose opinions differ by less than the confidence bound move towards each
    other by the convergence fraction of their difference.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of discussants.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Encounters
        >>> Encounters('meetings', target='people').target
        'people'
    """

    target: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the encounter rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.meet)]

    def meet(self, model: ModelView) -> None:
        """Run one random encounter per agent.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target)
        if len(agents) < 2:  # noqa: PLR2004
            return
        confidence, convergence = agents[0].confidence, agents[0].convergence
        opinions = np.array([agent.opinion for agent in agents], dtype=float)
        pairs = model.rng.integers(len(agents), size=(len(agents), 2))
        for first, second in pairs:
            difference = opinions[second] - opinions[first]
            if first != second and abs(difference) < confidence:
                opinions[first] += convergence * difference
                opinions[second] -= convergence * difference
        for agent, opinion in zip(agents, opinions, strict=True):
            agent.opinion = float(opinion)


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'bounded confidence', 'clusters'),
    references=('Deffuant et al. 2000', 'Castellano, Fortunato, and Loreto 2009', 'Gros 2015'),
    defaults={'until': 50},
)
def deffuant(
    n_agents: int = 200,
    *,
    confidence: float = 0.2,
    convergence: float = 0.5,
    random_state: int | np.random.Generator | None = None,
) -> System:
    """Return the Deffuant bounded confidence model, whose opinions settle into about `1 / (2 * confidence)` clusters.

    Args:
        n_agents:
            The number of agents.

        confidence:
            The largest opinion difference at which two agents compromise.

        convergence:
            The fraction of the difference by which each agent of a compromising pair moves.

        random_state:
            The seed or generator of the uniform initial opinions.

    Returns:
        system:
            The model, with the group `people`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import deffuant
        >>> deffuant(20, random_state=0).elements[0].size
        20
    """
    opinions = np.random.default_rng(random_state).random(n_agents)
    people = Discussant.group('people', n_agents, opinion=opinions, confidence=confidence, convergence=convergence)
    return System(elements=[people], interactions=[Encounters('meetings', target='people')])
