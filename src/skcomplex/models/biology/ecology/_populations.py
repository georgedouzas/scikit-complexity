"""Ecology models: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import delayed
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class PopulationElement(Element):
    """Population of one species.

    Args:
        label:
            The label of the population.

        n:
            The initial size.

    Examples:
        >>> from skcomplex.models.biology.ecology import PopulationElement
        >>> PopulationElement('rabbits', n=40).n
        40
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


class ExponentialGrowthInteraction(Interaction):
    """Growth at a constant per capita rate, `dn/dt = r * n`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        r:
            The per capita growth rate.

    Examples:
        >>> from skcomplex.models.biology.ecology import ExponentialGrowthInteraction
        >>> ExponentialGrowthInteraction('growth', target='rabbits', r=0.5).r
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate `r * n`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.n, s.r * s.target.n)]


class LogisticGrowthInteraction(Interaction):
    """Growth limited by a carrying capacity, `dn/dt = r * n * (1 - n / K)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        r:
            The per capita growth rate at low density.

        K:
            The carrying capacity.

    Examples:
        >>> from skcomplex.models.biology.ecology import LogisticGrowthInteraction
        >>> LogisticGrowthInteraction('growth', target='rabbits', K=200).K
        200
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r
        self.K = K

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate `r * n * (1 - n / K)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.n, s.r * s.target.n * (1 - s.target.n / s.K))]


class SeasonalLogisticGrowthInteraction(Interaction):
    """Logistic growth under a carrying capacity that varies periodically, `K(t) = K (1 + amplitude sin(2 pi t / T))`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        r:
            The per capita growth rate at low density.

        K:
            The mean carrying capacity.

        amplitude:
            The relative amplitude of the variation of the capacity, below 1.

        period:
            The period `T` of the variation.

    Examples:
        >>> from skcomplex.models.biology.ecology import SeasonalLogisticGrowthInteraction
        >>> SeasonalLogisticGrowthInteraction('growth', target='rabbits', period=12.0).period
        12.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        amplitude: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r
        self.K = K
        self.amplitude = amplitude
        self.period = period

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate `r * n * (1 - n / K(t))`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        capacity = s.K * (1 + s.amplitude * sin(2 * pi * s.t / s.period))
        return [RateLaw(s.target.n, s.r * s.target.n * (1 - s.target.n / capacity))]


class DelayedLogisticGrowthInteraction(Interaction):
    """Growth limited by the population a delay ago, `dn/dt = r * n * (1 - n(t - tau) / K)`, of Hutchinson.

    The delay stands for maturation or resource regeneration. The equilibrium `K` loses stability to oscillations when
    `r * tau` exceeds π/2.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        r:
            The per capita growth rate at low density.

        K:
            The carrying capacity.

        tau:
            The delay of the density feedback.

    Examples:
        >>> from skcomplex.models.biology.ecology import DelayedLogisticGrowthInteraction
        >>> DelayedLogisticGrowthInteraction('growth', target='rabbits', tau=2.0).tau
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec()] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r
        self.K = K
        self.tau = tau

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate `r * n * (1 - n(t - tau) / K)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.n, s.r * s.target.n * (1 - delayed(s.target.n, s.tau) / s.K))]


class MortalityInteraction(Interaction):
    """Death at a constant per capita rate, `dn/dt = -d * n`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        d:
            The per capita death rate.

    Examples:
        >>> from skcomplex.models.biology.ecology import MortalityInteraction
        >>> MortalityInteraction('death', target='foxes', d=0.4).d
        0.4
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.d = d

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate `-d * n`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.n, -s.d * s.target.n)]


class PredationInteraction(Interaction):
    """Lotka-Volterra predation, where encounters remove prey and feed predators.

    The prey loses `a * predator * prey` per unit time and the predator gains `b * predator * prey`.

    Args:
        label:
            The label of the interaction.

        predator:
            The path of the predator population.

        prey:
            The path of the prey population.

        a:
            The attack rate.

        b:
            The conversion rate of eaten prey into predators.

    Examples:
        >>> from skcomplex.models.biology.ecology import PredationInteraction
        >>> PredationInteraction('hunt', predator='foxes', prey='rabbits', a=0.4, b=0.1).prey
        'rabbits'
    """

    def __init__(
        self,
        label: str,
        *,
        predator: Annotated[str, RoleSpec()],
        prey: Annotated[str, RoleSpec()],
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label)
        self.predator = predator
        self.prey = prey
        self.a = a
        self.b = b

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates `-a * predator * prey` and `b * predator * prey`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        encounters = s.predator.n * s.prey.n
        return [RateLaw(s.prey.n, -s.a * encounters), RateLaw(s.predator.n, s.b * encounters)]
