"""Macroeconomics systems: Brock and Mirman's optimal growth.

A representative agent grows under Markov productivity.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction, TransitionInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class StochasticGrowthElement(Element):
    """Representative economy with capital and a productivity regime.

    Its technology is the Cobb-Douglas `Y = z K^alpha`. The productivity `z` is `z_low` in regime 0 and `z_high` in
    regime 1, and a Markov chain of the system moves the regime. With logarithmic utility and full depreciation, the
    optimal consumption is `C = (1 - alpha beta) Y` and the capital of the next period is the output saved,
    `K' = alpha beta z K^alpha`.

    Args:
        label:
            The label of the economy.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        K:
            The capital.

        regime:
            The productivity regime, 0 for low and 1 for high.

        alpha:
            The capital share of the technology.

        beta:
            The discount factor of the household.

        z_low:
            The productivity of regime 0.

        z_high:
            The productivity of regime 1.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import StochasticGrowthElement
        >>> StochasticGrowthElement('economy', beta=0.9).beta
        0.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        K: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 0.2,
        regime: Annotated[float, StateSpec(domain=NN)] = 0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 1 / 3,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.95,
        z_low: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.9,
        z_high: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.K = K
        self.regime = regime
        self.alpha = alpha
        self.beta = beta
        self.z_low = z_low
        self.z_high = z_high

    @staticmethod
    def get_productivity(s: SimpleNamespace) -> Expression:
        """Return the productivity of the current regime.

        Args:
            s:
                The namespace of the economy.

        Returns:
            z:
                The productivity.
        """
        return s.z_low + (s.z_high - s.z_low) * s.regime

    @staticmethod
    def get_output(s: SimpleNamespace) -> Expression:
        """Return the output `z K^alpha`.

        Args:
            s:
                The namespace of the economy.

        Returns:
            output:
                The output.
        """
        return StochasticGrowthElement.get_productivity(s) * s.K**s.alpha

    @staticmethod
    def get_consumption(s: SimpleNamespace) -> Expression:
        """Return the optimal consumption `(1 - alpha beta) Y`.

        Args:
            s:
                The namespace of the economy.

        Returns:
            consumption:
                The consumption.
        """
        return (1 - s.alpha * s.beta) * StochasticGrowthElement.get_output(s)


class BrockMirmanGrowthSystem(BaseSystem):
    """The Brock-Mirman economy.

    It is the optimal growth of a representative agent whose productivity is Markov. A representative consumer maximises
    `E_0 sum_t beta^t ln C_t` subject to `K_{t+1} = Y_t - C_t`, with full depreciation and the output
    `Y_t = z_t K_t^alpha` of a representative firm, whose productivity `z_t` follows a Markov chain. The solution is the
    linear consumption rule `C_t = (1 - alpha beta) Y_t`, `C = 0.68 Y` for `alpha = 1/3` and `beta = 0.95`, so the
    capital follows `K_{t+1} = alpha beta z_t K_t^alpha` and `ln K` is a stationary autoregression around
    `(ln(alpha beta) + E ln z) / (1 - alpha)`. This is the standard model that Caiani et al. (2016, Section 1.1.4, Eqs.
    1.3 to 1.9) compare with an agent-based economy. The source leaves the chain open, so the productivity takes two
    levels that stay with the probability `persistence`.

    The system has the capital `economy.K` and the regime `economy.regime`. The observables of the system are
    `economy.productivity`, `economy.output`, and `economy.consumption`.

    Args:
        K:
            The initial capital.

        alpha:
            The capital share of the technology.

        beta:
            The discount factor of the household.

        z_low:
            The low productivity.

        z_high:
            The high productivity.

        persistence:
            The probability that the productivity keeps its level from one period to the next.

        regime:
            The initial regime, 0 for the low and 1 for the high productivity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import BrockMirmanGrowthSystem
        >>> BrockMirmanGrowthSystem(beta=0.9).prepare().system_.elements[0].beta
        0.9
    """

    field = 'economics/macroeconomics'
    topics = ('optimal growth', 'representative agent', 'markov chain', 'productivity shocks', 'discrete map')
    references = ('Caiani et al. 2016 §1.1.4', 'Brock and Mirman 1972')
    until = 200
    observables = tuple(
        [
            ObservableSpec(name, function, element='economy')
            for name, function in (
                ('productivity', StochasticGrowthElement.get_productivity),
                ('output', StochasticGrowthElement.get_output),
                ('consumption', StochasticGrowthElement.get_consumption),
            )
        ],
    )

    def __init__(
        self,
        K: float = 0.2,
        *,
        alpha: float = 1 / 3,
        beta: float = 0.95,
        z_low: float = 0.9,
        z_high: float = 1.1,
        persistence: float = 0.9,
        regime: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.z_low = z_low
        self.z_high = z_high
        self.persistence = persistence
        self.regime = regime

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        chain = ((self.persistence, 1 - self.persistence), (1 - self.persistence, self.persistence))
        return CompositeSystem(
            self.label,
            elements=[
                StochasticGrowthElement(
                    'economy',
                    K=self.K,
                    regime=self.regime,
                    alpha=self.alpha,
                    beta=self.beta,
                    z_low=self.z_low,
                    z_high=self.z_high,
                ),
            ],
            interactions=[
                MapInteraction(
                    'saving',
                    target='economy',
                    state='K',
                    function=lambda s: s.target.alpha * s.target.beta * StochasticGrowthElement.get_output(s.target),
                ),
                TransitionInteraction('productivity', target='economy', state='regime', matrix=lambda s: chain),
            ],
        )
