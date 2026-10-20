"""Innovation models: the diffusion of a new product through innovators and imitators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Element, Flow, Law, Symbols, System
from ....core.runtime import register_model


def _adoption(s: Symbols) -> list[Law]:
    """Return the adoption flow from potential adopters to adopters, driven by innovation and imitation."""
    market = s.P + s.A
    return [Flow(s.P, s.A, (s.p + s.q * s.A / market) * s.P, label='adoption')]


BassMarket = Element.define(
    'BassMarket',
    states={
        'P': state(default=1000.0, domain=Domain.NONNEGATIVE, quantity='people'),
        'A': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='people'),
    },
    parameters={
        'p': parameter(default=0.03, domain=Domain.NONNEGATIVE),
        'q': parameter(default=0.38, domain=Domain.NONNEGATIVE),
    },
    laws=_adoption,
)


@register_model(
    field='economics/innovation',
    tags=('diffusion', 'marketing', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bass 1969',),
    defaults={'until': 20.0, 'dt': 0.25},
)
def bass_diffusion(market: float = 1000.0, *, adopters: float = 0.0, p: float = 0.03, q: float = 0.38) -> System:
    """Return the Bass diffusion model, people who adopt a new product through advertising and word of mouth.

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
            The model, with the potential adopters `market.P`, the adopters `market.A`, and the adoption flow
            `market.adoption` in people per year, over years.

    Examples:
        >>> from skcomplex.models.economics.innovation import bass_diffusion
        >>> bass_diffusion(market=500.0, adopters=10.0).elements[0].P
        490.0
    """
    return System(elements=[BassMarket('market', P=market - adopters, A=adopters, p=p, q=q)])
