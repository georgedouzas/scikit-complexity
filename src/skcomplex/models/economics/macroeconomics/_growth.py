"""Macroeconomics models: growth and business cycles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RateLaw, UpdateLaw
from ....core.laws.utils import lag
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model


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
        elements=[EconomyElement('economy', k=k)],
        interactions=[SolowGrowthInteraction('growth', target='economy', s=s, alpha=alpha, delta=delta)],
    )


class IncomeElement(Element):
    """Income element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Y: Annotated[object, StateSpec(lags=1, initial=(100.0,))] = 110.0,
        Y_lag1: float = 100.0,
        c: Annotated[object, ParameterSpec()] = 0.8,
        v: Annotated[object, ParameterSpec()] = 1.5,
        G: Annotated[object, ParameterSpec()] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Y = Y
        self.Y_lag1 = Y_lag1
        self.c = c
        self.v = v
        self.G = G

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the income element."""
        return [UpdateLaw(s.Y, s.c * (1 + s.v) * s.Y - s.c * s.v * lag(s.Y, 1) + s.G)]


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
    return System(elements=[IncomeElement('economy', Y=income, Y_lag1=previous, c=c, v=v, G=G)])


class EconomyElement(Element):
    """Economy described by its capital per worker.

    Args:
        label:
            The label of the economy.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        k:
            The initial capital per worker.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import EconomyElement
        >>> EconomyElement('country', k=1.0).k
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        k: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.k = k


class SolowGrowthInteraction(Interaction):
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
        >>> from skcomplex.models.economics.macroeconomics import SolowGrowthInteraction
        >>> SolowGrowthInteraction('solow', target='country', s=0.3).s
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.s = s
        self.alpha = alpha
        self.delta = delta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate `s * k**alpha - delta * k`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.k, s.s * s.target.k**s.alpha - s.delta * s.target.k)]
