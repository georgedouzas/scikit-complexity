"""Synchronous update of the social impact model, where every agent takes the opinion of the stronger impact."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.interactions import MapInteraction
from ....core.types import ModelViewProtocol
from ....specs import RoleSpec
from ._influencee_view import InfluenceeView


class SocialImpactInteraction(MapInteraction):
    """Synchronous update of the social impact model, where every agent takes the opinion of the stronger impact.

    The impact on agent `i` is `I_i = sum_j t_j (1 - o_i o_j) / d_ij^a - sum_j s_j (1 + o_i o_j) / d_ij^a`, with the
    persuasiveness `t`, the supportiveness `s`, and the distance `d` on the grid, where an agent supports itself from
    distance 1. The impact is the net pressure to change, whatever the opinion, so an agent flips its opinion when
    `I_i + h_i > 0`, with a uniform random field `h_i`. Every agent decides from the opinions at the start of the step.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of agents.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import SocialImpactInteraction
        >>> SocialImpactInteraction('impact', target='people').form
        'agent'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label, target=target, state='opinion', function=self._get_opinion, form='agent')

    def _get_impact(self, agent: InfluenceeView, model: ModelViewProtocol) -> float:
        """Return the net pressure to change on an agent from every agent of the group, itself from distance 1."""
        agents = [item for item in model.agents('target') if isinstance(item, InfluenceeView)]
        locations = np.array([item.location for item in agents], dtype=float)
        distances = np.linalg.norm(locations - np.asarray(agent.location, dtype=float), axis=-1)
        distances[[item.id == agent.id for item in agents]] = 1.0
        weights = distances**-agent.exponent
        agreement = agent.opinion * np.array([item.opinion for item in agents], dtype=float)
        persuasion = np.array([item.persuasiveness for item in agents], dtype=float)
        support = np.array([item.supportiveness for item in agents], dtype=float)
        return float((weights * persuasion * (1 - agreement)).sum() - (weights * support * (1 + agreement)).sum())

    def _get_opinion(self, agent: InfluenceeView, model: ModelViewProtocol) -> float:
        """Return the opposite opinion of an agent under a positive net pressure to change, and its own otherwise."""
        field = model.rng.uniform(-agent.noise, agent.noise)
        return -agent.opinion if self._get_impact(agent, model) + field > 0 else float(agent.opinion)
