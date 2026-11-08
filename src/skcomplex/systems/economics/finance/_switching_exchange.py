"""Finance systems: Kirman's exchange rate driven by traders who switch between fundamentalism and chartism."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import lag, normal, uniform, where
from ....core.components.system import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class SwitchingExchangeElement(Element):
    """Foreign exchange market whose rate deviation follows the share of fundamentalists of Kirman's recruitment chain.

    The deviation `x = s - s*` of the exchange rate from its fundamental value follows Eq. 1.36 of Caiani et al.
    (2016), `(1 + r) x_t = [1 - v n_t + g (1 - n_t)] x_t-1 - g (1 - n_t) x_t-2 + sigma eps_t`, with a standard normal
    shock `eps_t` of the supply of foreign exchange. Of the `N` traders, `k` are fundamentalists, `n = k / N`. Every
    step one trader meets another at random, after Kirman (1993): a chartist turns fundamentalist with probability
    `(1 - k / N) (epsilon + (1 - delta) k / (N - 1))` and a fundamentalist turns chartist with probability
    `(k / N) (epsilon + (1 - delta) (N - k) / (N - 1))`, by a spontaneous change `epsilon` or by recruitment.

    Args:
        label:
            The label of the market.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The deviation of the exchange rate from its fundamental value.

        k:
            The number of fundamentalists.

        n_traders:
            The number of traders `N`.

        r:
            The interest rate of the domestic currency.

        v:
            The share of the deviation that fundamentalists expect to close.

        g:
            The share of the last change that chartists expect to repeat.

        sigma:
            The standard deviation of the shocks of the deviation.

        epsilon:
            The probability of a spontaneous change of strategy.

        delta:
            One minus the probability that a trader recruits the trader it meets.

    Examples:
        >>> from skcomplex.systems.economics.finance import SwitchingExchangeElement
        >>> SwitchingExchangeElement('market', g=1.2).g
        1.2
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(lags=1)] = 0.1,
        k: Annotated[float, StateSpec(domain=NN)] = 50,
        n_traders: Annotated[float, ParameterSpec(domain=RealSet.open(1, oo))] = 100.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        v: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
        g: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.002,
        delta: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.k = k
        self.n_traders = n_traders
        self.r = r
        self.v = v
        self.g = g
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta

    @staticmethod
    def get_share(s: SimpleNamespace) -> Expression:
        """Return the share of fundamentalists `n = k / N`.

        Args:
            s:
                The namespace of the market.

        Returns:
            share:
                The share of fundamentalists.
        """
        return s.k / s.n_traders

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the step of the deviation and the recruitment step of the fundamentalists.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        n = self.get_share(s)
        chartists = 1 - n
        deviation = ((1 - s.v * n + s.g * chartists) * s.x - s.g * chartists * lag(s.x, 1)) / (1 + s.r)
        recruit = (1 - s.delta) / (s.n_traders - 1)
        up = chartists * (s.epsilon + recruit * s.k)
        down = n * (s.epsilon + recruit * (s.n_traders - s.k))
        draw = uniform('meeting', 0, 1)
        change = where(draw < up, 1, where(draw < up + down, -1, 0))
        return [
            UpdateLaw(s.x, deviation + s.sigma * normal('supply', 0, 1)),
            UpdateLaw(s.k, s.k + change),
        ]


class KirmanExchangeRateSystem(BaseSystem):
    """Kirman's foreign exchange market, whose traders switch between fundamentalism and chartism like ants.

    Fundamentalists expect the exchange rate to revert to its fundamental value, `s_t-1 + v (s* - s_t-1)`, and chartists
    expect the last change to go on, `s_t-1 + g (s_t-1 - s_t-2)`. With the mean-variance demand of both and the share
    `n_t` of fundamentalists, the market clears at the deviation of Eq. 1.36 of Caiani et al. (2016, Section 1.3.2),
    `(1 + r) x_t = [1 - v n_t + g (1 - n_t)] x_t-1 - g (1 - n_t) x_t-2`, written there with `s_t-2` for `x_t-2`, a shock
    of the supply of foreign exchange added. The share follows the recruitment chain of Kirman's ants, which the source
    refers to and Kirman (1993) gives: when spontaneous changes are rarer than recruitment, `epsilon < (1 - delta) /
    (N - 1)`, the traders spend long periods mostly of one kind and switch suddenly. A market of fundamentalists is
    stable with the eigenvalue `(1 - v) / (1 + r)` (Eq. 1.37), and a market of chartists oscillates, stable only while
    `g <= 1 + r` (Eq. 1.38), so the volatility of the rate comes and goes with the chartists. One trader meets another
    every step, so the defaults take a small market of 30 traders, whose regimes switch within a few thousand steps.

    The system has the deviation `market.x` and the number of fundamentalists `market.k`. The observable of the system
    is the share of fundamentalists `market.fundamentalists`.

    Args:
        x:
            The initial deviation of the exchange rate from its fundamental value, also its value one step earlier.

        k:
            The initial number of fundamentalists.

        n_traders:
            The number of traders.

        r:
            The interest rate of the domestic currency.

        v:
            The share of the deviation that fundamentalists expect to close.

        g:
            The share of the last change that chartists expect to repeat.

        sigma:
            The standard deviation of the shocks of the deviation.

        epsilon:
            The probability of a spontaneous change of strategy.

        delta:
            One minus the probability that a trader recruits the trader it meets.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.finance import KirmanExchangeRateSystem
        >>> KirmanExchangeRateSystem(g=1.2).prepare().system_.elements[0].g
        1.2
    """

    field = 'economics/finance'
    topics = ('exchange rate', 'fundamentalists', 'chartists', 'herding', 'ants', 'discrete map')
    references = ('Caiani et al. 2016 §1.3.2', 'Kirman 1993')
    until = 20000
    observables = (ObservableSpec('fundamentalists', SwitchingExchangeElement.get_share, element='market'),)

    def __init__(
        self,
        x: float = 0.1,
        k: int = 15,
        *,
        n_traders: int = 30,
        r: float = 0.01,
        v: float = 0.5,
        g: float = 1.0,
        sigma: float = 0.01,
        epsilon: float = 0.005,
        delta: float = 0.01,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.k = k
        self.n_traders = n_traders
        self.r = r
        self.v = v
        self.g = g
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        market = SwitchingExchangeElement(
            'market',
            x=self.x,
            k=self.k,
            n_traders=self.n_traders,
            r=self.r,
            v=self.v,
            g=self.g,
            sigma=self.sigma,
            epsilon=self.epsilon,
            delta=self.delta,
        )
        return System(self.label, elements=[market])
