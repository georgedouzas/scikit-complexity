"""Innovation systems: the diffusion of a new product through innovators and imitators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


def _adoption(s: SimpleNamespace) -> Expression:
    """Return the adoption rate `(p + q A / (P + A)) P` of the market a transfer moves people within."""
    market = s.source
    return (market.p + market.q * market.A / (market.P + market.A)) * market.P


class BassMarketElement(Element):
    """Market of a new product, split into potential adopters and adopters, who adopt by innovation and imitation.

    The market holds the stocks and the coefficients, and a transfer within it moves the adopters from `P` to `A`.

    Args:
        label:
            The label of the market.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        P:
            The potential adopters.

        A:
            The adopters.

        p:
            The coefficient of innovation.

        q:
            The coefficient of imitation.

    Examples:
        >>> from skcomplex.systems.economics.innovation._bass import BassMarketElement
        >>> BassMarketElement('market', P=900.0, A=100.0).q
        0.38
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        P: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='people')] = 1000.0,
        A: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='people')] = 0.0,
        p: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        q: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.38,
    ) -> None:
        super().__init__(label, visual=visual)
        self.P = P
        self.A = A
        self.p = p
        self.q = q


@register_system(
    field='economics/innovation',
    tags=('diffusion', 'marketing', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bass 1969',),
    defaults={'until': 20.0, 'dt': 0.25},
)
def bass_diffusion(market: float = 1000.0, *, adopters: float = 0.0, p: float = 0.03, q: float = 0.38) -> System:
    """Return the Bass diffusion system, people who adopt a new product through advertising and word of mouth.

    A market of `N` people splits into potential adopters `P` and adopters `A`. Potential adopters adopt at the rate
    `(p + q A / N) P`, where the coefficient of innovation `p` captures adoption from external influence, such as
    advertising, and the coefficient of imitation `q` adoption from contact with adopters. People are conserved, so
    the adoption is a flow from `P` to `A`. With `q > p` the adoption rate rises to a peak at the time
    `log(q / p) / (p + q)` and the adopters follow an S-shaped curve. Starting without adopters, the closed form is
    `A(t) = N (1 - exp(-(p + q) t)) / (1 + (q / p) exp(-(p + q) t))`. The defaults are the mean coefficients that
    Sultan, Farley and Lehmann (1990) estimated over many products, with time in years.

    Args:
        market:
            The market potential `N`, the number of people who eventually adopt.

        adopters:
            The initial adopters, a part of the market.

        p:
            The coefficient of innovation, per year.

        q:
            The coefficient of imitation, per year.

    Returns:
        system:
            The system, with the potential adopters `market.P`, the adopters `market.A`, and the adoption flow
            `market.adoption` in people per year, over years.

    Examples:
        >>> from skcomplex.systems.economics.innovation import bass_diffusion
        >>> bass_diffusion(market=500.0, adopters=10.0).elements[0].P
        490.0
    """
    adoption = TransferInteraction(
        'adoption',
        source='market',
        target='market',
        state='P',
        into='A',
        amount=_adoption,
        flow='adoption',
    )
    return System(
        elements=[BassMarketElement('market', P=market - adopters, A=adopters, p=p, q=q)],
        interactions=[adoption],
    )
