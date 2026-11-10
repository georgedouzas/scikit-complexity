"""Finance systems: Kirman's exchange rate driven by switching traders.

The traders switch between fundamentalism and chartism.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.laws.utils import lag, normal, uniform, where
from ....core.specs import DeclaredVariableSpec, ObservableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


def _get_share(s: SimpleNamespace) -> Expression:
    """Return the share of fundamentalists `n = k / N` of the market."""
    return s.k / s.n_traders


class KirmanExchangeRateSystem(DynamicalSystem):
    """Kirman's foreign exchange market.

    Its traders switch between fundamentalism and chartism like ants. Fundamentalists expect the exchange rate to revert
    to its fundamental value, `s_t-1 + v (s* - s_t-1)`, and chartists expect the last change to go on,
    `s_t-1 + g (s_t-1 - s_t-2)`. With the mean-variance demand of both and the share `n_t` of fundamentalists, the
    market clears at the deviation of Eq. 1.36 of Caiani et al. (2016, Section 1.3.2),
    `(1 + r) x_t = [1 - v n_t + g (1 - n_t)] x_t-1 - g (1 - n_t) x_t-2`, written there with `s_t-2` for `x_t-2`, a shock
    of the supply of foreign exchange added. The share follows the recruitment chain of Kirman's ants, which the source
    refers to and Kirman (1993) gives: when spontaneous changes are rarer than recruitment,
    `epsilon < (1 - delta) / (N - 1)`, the traders spend long periods mostly of one kind and switch suddenly. A market
    of fundamentalists is stable with the eigenvalue `(1 - v) / (1 + r)` (Eq. 1.37), and a market of chartists
    oscillates, stable only while `g <= 1 + r` (Eq. 1.38), so the volatility of the rate comes and goes with the
    chartists. One trader meets another every step, so the defaults take a small market of 30 traders, whose regimes
    switch within a few thousand steps.

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
        >>> KirmanExchangeRateSystem(g=1.2).prepare().system_.elements[0].get_value('g')
        1.2
    """

    field = 'economics/finance'
    topics = ('exchange rate', 'fundamentalists', 'chartists', 'herding', 'ants', 'discrete map')
    references = ('Caiani et al. 2016 §1.3.2', 'Kirman 1993')
    until = 20000
    draws = True
    observables = (ObservableSpec('fundamentalists', _get_share, element='market'),)

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
        super().__init__(label=label)
        self.x = x
        self.k = k
        self.n_traders = n_traders
        self.r = r
        self.v = v
        self.g = g
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'market'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        nonnegative = RealSet.closed_open(0, oo)
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec(lags=1)),
            'k': DeclaredVariableSpec(self.k, StateSpec(domain=NN)),
            'n_traders': DeclaredVariableSpec(self.n_traders, ParameterSpec(domain=RealSet.open(1, oo))),
            'r': DeclaredVariableSpec(self.r, ParameterSpec(domain=nonnegative)),
            'v': DeclaredVariableSpec(self.v, ParameterSpec(domain=RealSet.closed(0, 1))),
            'g': DeclaredVariableSpec(self.g, ParameterSpec(domain=nonnegative)),
            'sigma': DeclaredVariableSpec(self.sigma, ParameterSpec(domain=nonnegative)),
            'epsilon': DeclaredVariableSpec(self.epsilon, ParameterSpec(domain=RealSet.closed(0, 1))),
            'delta': DeclaredVariableSpec(self.delta, ParameterSpec(domain=RealSet.closed(0, 1))),
        }

    def _get_equations(self) -> Equations:
        """Return the step of the deviation and the recruitment step of the fundamentalists."""
        return {'x': self._deviation, 'k': self._recruitment}

    @staticmethod
    def _deviation(s: SimpleNamespace) -> Expression:
        """Return the next deviation, which clears the market, with a shock of the supply."""
        market = s.target
        n = _get_share(market)
        chartists = 1 - n
        deviation = ((1 - market.v * n + market.g * chartists) * market.x - market.g * chartists * lag(market.x, 1)) / (
            1 + market.r
        )
        return deviation + market.sigma * normal('supply', 0, 1)

    @staticmethod
    def _recruitment(s: SimpleNamespace) -> Expression:
        """Return the next number of fundamentalists after one trader meets another."""
        market = s.target
        n = _get_share(market)
        recruit = (1 - market.delta) / (market.n_traders - 1)
        up = (1 - n) * (market.epsilon + recruit * market.k)
        down = n * (market.epsilon + recruit * (market.n_traders - market.k))
        draw = uniform('meeting', 0, 1)
        return market.k + where(draw < up, 1, where(draw < up + down, -1, 0))
