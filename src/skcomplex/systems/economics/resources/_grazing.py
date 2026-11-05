"""Grazing of the meadow, a matching of the cows with the pastures, after which every cow eats and burns energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import MatchingInteraction
from ....core.components.laws.utils import transfer
from ....core.components.types import MatchingRule
from ....core.engine.solvers import SystemView
from ....core.errors import SimulationError
from ....core.specs import (
    MatchSpec,
    ParameterSpec,
    RandomSearchSpec,
    RoleSpec,
)
from ._cow_view import CowView
from ._herder_view import HerderView
from ._pasture_view import PastureView


class GrazingInteraction(MatchingInteraction[SystemView]):
    """Grazing of the meadow, a matching of the cows with the pastures, after which every cow eats and burns energy.

    The rule of the matching pairs the cows with the pastures, such as
    `PartnerSpec('local', 'score', score=GrazingInteraction.score, clearing='provider')`, which pairs every cow, in
    random order, with the free pasture of most grass in its cell or the cells around it, ties broken at random. A
    pasture takes one cow, its default quantity of one unit. Every matched cow moves to its pasture and eats up to its
    appetite as a transfer of grass to its energy, and its herder earns milk for what it ate. Then every cow burns
    its metabolism and counts in the herd of its herder, and a cow whose energy runs out dies.

    Args:
        label:
            The label of the grazing.

        requesters:
            The path of the group of cows.

        providers:
            The path of the group of pastures.

        rule:
            The spec of the algorithm that pairs the cows with the pastures.

        phase:
            The phase the grazing runs in.

        herders:
            The path of the group of herders.

        appetite:
            The most grass a cow eats in a step.

        metabolism:
            The energy a cow burns in a step.

        milk:
            The income of a herder per unit of grass its cows eat.

    Examples:
        >>> from skcomplex.systems.economics.resources import GrazingInteraction
        >>> GrazingInteraction('grazing', requesters='cows', providers='meadow', herders='herders').appetite
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        requesters: Annotated[str, RoleSpec()],
        providers: Annotated[str, RoleSpec()],
        rule: MatchingRule = RandomSearchSpec(),
        phase: str | None = 'grazing',
        herders: Annotated[str, RoleSpec()],
        appetite: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        metabolism: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        milk: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, requesters=requesters, providers=providers, rule=rule, phase=phase)
        self.herders = herders
        self.appetite = appetite
        self.metabolism = metabolism
        self.milk = milk

    @staticmethod
    def score(cow: object, pasture: object) -> float:
        """Return the score of a pasture for a cow, the grass on the pasture.

        Args:
            cow:
                The agent view of the cow.

            pasture:
                The agent view of the pasture.

        Returns:
            score:
                The grass on the pasture, or minus infinity for an agent that is not a pasture.
        """
        del cow
        return float(pasture.grass) if isinstance(pasture, PastureView) else -np.inf

    @staticmethod
    def _get_herder(cow: CowView) -> HerderView:
        """Return the herder that owns a cow, which its reference state `owner` names."""
        herder = cow.referenced('owner')
        if not isinstance(herder, HerderView):
            error_msg = f'Cow {cow.id} names no herder as its owner.'
            raise SimulationError(error_msg)
        return herder

    def _settle(self, match: MatchSpec, system: SystemView) -> None:
        """Move the cow of a match to its pasture, feed it, and pay its herder for what it ate.

        Args:
            match:
                The match of a cow with a pasture.

            system:
                The system view of the grazing.
        """
        cow, pasture = match.requester, match.provider
        if not isinstance(cow, CowView) or not isinstance(pasture, PastureView):
            return
        cow.move_to(pasture.location)
        eaten = min(pasture.grass, system.parameter('appetite'))
        transfer(pasture, cow, eaten, 'grass', 'energy')
        self._get_herder(cow).income += system.parameter('milk') * eaten

    def match(self, system: SystemView) -> None:
        """Pair and feed the cows, then let every cow burn its metabolism, count it in its herd, and starve.

        Args:
            system:
                The system view of the grazing.
        """
        super().match(system)
        for cow in system.agents('requesters', CowView):
            cow.energy -= system.parameter('metabolism')
            self._get_herder(cow).herd += 1
            if cow.energy < 0:
                cow.remove()
