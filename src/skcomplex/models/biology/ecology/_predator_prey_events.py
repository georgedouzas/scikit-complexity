"""Births, crowding, saturating predation, and deaths of prey and predators as events of single individuals."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import Interaction, JumpInteraction, RateInteraction
from ....core.laws import BaseLaw
from ....specs import ParameterSpec, RoleSpec


class PredatorPreyEventsInteraction(Interaction):
    """Births, crowding, saturating predation, and deaths of prey and predators as events of single individuals.

    Prey are born at the rate `r x` and die of crowding at the rate `r x^2 / K`. Predators kill prey at the Holling
    type II rate `a x y / (1 + a h x)`, are born at `e` times that rate, and die at the rate `m y`. In deterministic
    mode the rates drive the populations as differential equations.

    Args:
        label:
            The label of the interaction.

        prey:
            The path of the prey population `x`.

        predators:
            The path of the predator population `y`.

        r:
            The birth rate of the prey.

        K:
            The carrying capacity of the prey.

        a:
            The attack rate.

        h:
            The handling time.

        e:
            The number of predators born per prey eaten.

        m:
            The death rate of the predators.

        stochastic:
            Whether births, deaths, and kills are random events.

    Examples:
        >>> from skcomplex.models.biology.ecology import PredatorPreyEventsInteraction
        >>> PredatorPreyEventsInteraction('events', prey='voles', predators='weasels', K=200.0).K
        200.0
    """

    def __init__(
        self,
        label: str,
        *,
        prey: Annotated[str, RoleSpec()],
        predators: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 200.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        h: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        e: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        stochastic: bool = True,
    ) -> None:
        super().__init__(label)
        self.prey = prey
        self.predators = predators
        self.r = r
        self.K = K
        self.a = a
        self.h = h
        self.e = e
        self.m = m
        self.stochastic = stochastic

    def _get_change(self, member: SimpleNamespace, step: int, propensity: Expression) -> Sequence[BaseLaw]:
        """Return the laws that change the population of a member by a step at a propensity.

        In stochastic mode the change is a jump of one individual, and otherwise a rate of the step times the
        propensity.
        """
        change: Interaction = (
            JumpInteraction(self.label, target='member', changes={'n': step}, rate=lambda _: propensity)
            if self.stochastic
            else RateInteraction(self.label, target='member', state='n', rate=lambda _: step * propensity)
        )
        return change.prepare().run(SimpleNamespace(target=member)).laws_

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the five channels of births and deaths.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        prey, predators = s.prey, s.predators
        x, y = prey.n, predators.n
        kills = s.a * x * y / (1 + s.a * s.h * x)
        return [
            *self._get_change(prey, 1, s.r * x),
            *self._get_change(prey, -1, s.r * x**2 / s.K),
            *self._get_change(prey, -1, kills),
            *self._get_change(predators, 1, s.e * kills),
            *self._get_change(predators, -1, s.m * y),
        ]
