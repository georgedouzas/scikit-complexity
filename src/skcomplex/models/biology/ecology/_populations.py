"""Ecology models: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import pi, sin

from ....core.base import Domain, parameter, role, state
from ....core.model import Element, Interaction, Law, Rate, Symbols, delayed


class Population(Element):
    """Population of one species.

    Args:
        label:
            The label of the population.

        n:
            The initial size.

    Examples:
        >>> from skcomplex.models.biology.ecology import Population
        >>> Population('rabbits', n=40).n
        40
    """

    n: float = state(default=1.0, domain=Domain.NONNEGATIVE)


class ExponentialGrowth(Interaction):
    """Growth at a constant per capita rate, `dn/dt = r * n`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        r:
            The per capita growth rate.

    Examples:
        >>> from skcomplex.models.biology.ecology import ExponentialGrowth
        >>> ExponentialGrowth('growth', target='rabbits', r=0.5).r
        0.5
    """

    target: str = role()
    r: float = parameter(default=1.0, domain=Domain.REAL)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `r * n`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.n, s.r * s.target.n)]


class LogisticGrowth(Interaction):
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
        >>> from skcomplex.models.biology.ecology import LogisticGrowth
        >>> LogisticGrowth('growth', target='rabbits', K=200).K
        200
    """

    target: str = role()
    r: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    K: float = parameter(default=100.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `r * n * (1 - n / K)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.n, s.r * s.target.n * (1 - s.target.n / s.K))]


class SeasonalLogisticGrowth(Interaction):
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
        >>> from skcomplex.models.biology.ecology import SeasonalLogisticGrowth
        >>> SeasonalLogisticGrowth('growth', target='rabbits', period=12.0).period
        12.0
    """

    target: str = role()
    r: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    K: float = parameter(default=100.0, domain=Domain.POSITIVE)
    amplitude: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    period: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `r * n * (1 - n / K(t))`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        capacity = s.K * (1 + s.amplitude * sin(2 * pi * s.t / s.period))
        return [Rate(s.target.n, s.r * s.target.n * (1 - s.target.n / capacity))]


class DelayedLogisticGrowth(Interaction):
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
        >>> from skcomplex.models.biology.ecology import DelayedLogisticGrowth
        >>> DelayedLogisticGrowth('growth', target='rabbits', tau=2.0).tau
        2.0
    """

    target: str = role()
    r: float = parameter(default=1.0, domain=Domain.REAL)
    K: float = parameter(default=100.0, domain=Domain.POSITIVE)
    tau: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `r * n * (1 - n(t - tau) / K)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.n, s.r * s.target.n * (1 - delayed(s.target.n, s.tau) / s.K))]


class Mortality(Interaction):
    """Death at a constant per capita rate, `dn/dt = -d * n`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        d:
            The per capita death rate.

    Examples:
        >>> from skcomplex.models.biology.ecology import Mortality
        >>> Mortality('death', target='foxes', d=0.4).d
        0.4
    """

    target: str = role()
    d: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `-d * n`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.n, -s.d * s.target.n)]


class Predation(Interaction):
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
        >>> from skcomplex.models.biology.ecology import Predation
        >>> Predation('hunt', predator='foxes', prey='rabbits', a=0.4, b=0.1).prey
        'rabbits'
    """

    predator: str = role()
    prey: str = role()
    a: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    b: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rates `-a * predator * prey` and `b * predator * prey`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        encounters = s.predator.n * s.prey.n
        return [Rate(s.prey.n, -s.a * encounters), Rate(s.predator.n, s.b * encounters)]
