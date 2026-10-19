"""Macroeconomics models: growth and business cycles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, role, state
from ....core.model import Element, Interaction, Law, Rate, Symbols, System, Update, lag
from ....core.runtime import register_model


@register_model(
    field='economics/macroeconomics',
    tags=('growth', 'ode'),
    references=('Solow 1956',),
    defaults={'until': 200.0, 'dt': 1.0},
)
def solow(k: float = 1.0, *, s: float = 0.2, alpha: float = 0.3, delta: float = 0.05) -> System:
    """Return an economy whose capital per worker approaches its steady state through saving and depreciation.

    Args:
        k:
            The initial capital per worker.

        s:
            The saving rate.

        alpha:
            The capital share of output.

        delta:
            The depreciation rate.

    Returns:
        system:
            The model, with the economy `economy`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import solow
        >>> solow(s=0.3).interactions[0].s
        0.3
    """
    return System(
        elements=[Economy('economy', k=k)],
        interactions=[SolowGrowth('growth', target='economy', s=s, alpha=alpha, delta=delta)],
    )


Income = Element.define(
    'Income',
    states={'Y': state(default=110.0, lags=1, initial=(100.0,))},
    parameters={'c': 0.8, 'v': 1.5, 'G': 20.0},
    laws=lambda s: [Update(s.Y, s.c * (1 + s.v) * s.Y - s.c * s.v * lag(s.Y, 1) + s.G)],
)


@register_model(
    field='economics/macroeconomics',
    tags=('business cycle', 'discrete map', 'second order'),
    references=('Samuelson 1939',),
    defaults={'until': 40, 'dt': 1},
)
def multiplier_accelerator(
    income: float = 110.0,
    previous: float = 100.0,
    *,
    c: float = 0.8,
    v: float = 1.5,
    G: float = 20.0,
) -> System:
    """Return Samuelson's multiplier-accelerator, `Y_{t+1} = c (1 + v) Y_t - c v Y_{t-1} + G`.

    Income oscillates around `G / (1 - c)`, with growing amplitude when `c v > 1` and complex characteristic roots.

    Args:
        income:
            The income of the current period.

        previous:
            The income of the previous period.

        c:
            The marginal propensity to consume.

        v:
            The accelerator, the investment response to the change in consumption.

        G:
            The government spending.

    Returns:
        system:
            The model, with the income `economy.Y` and its previous value `economy.Y_lag1`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import multiplier_accelerator
        >>> multiplier_accelerator(v=0.5).elements[0].v
        0.5
    """
    return System(elements=[Income('economy', Y=income, Y_lag1=previous, c=c, v=v, G=G)])


class Economy(Element):
    """Economy described by its capital per worker.

    Args:
        label:
            The label of the economy.

        k:
            The initial capital per worker.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import Economy
        >>> Economy('country', k=1.0).k
        1.0
    """

    k: float = state(default=1.0, domain=Domain.NONNEGATIVE)


class SolowGrowth(Interaction):
    """Solow growth, `dk/dt = s * k**alpha - delta * k`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the economy.

        s:
            The saving rate.

        alpha:
            The output elasticity of capital.

        delta:
            The depreciation rate.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import SolowGrowth
        >>> SolowGrowth('solow', target='country', s=0.3).s
        0.3
    """

    target: str = role()
    s: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=0.3, domain=Domain.POSITIVE)
    delta: float = parameter(default=0.05, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rate `s * k**alpha - delta * k`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rate(s.target.k, s.s * s.target.k**s.alpha - s.delta * s.target.k)]
