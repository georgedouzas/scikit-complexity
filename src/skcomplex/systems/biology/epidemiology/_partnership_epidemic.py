"""Population of Watts and May whose infections pass along partnerships that last, with a fixed exposed period."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import delayed
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

FRACTION = RealSet.closed(0, 1)


class PartnershipEpidemicElement(Element):
    """Population of Watts and May whose infections pass along partnerships that last, with a fixed exposed period.

    Susceptible people `S` acquire new partners at the rate `c`, each infecting with the probability `beta` over the
    partnership, and are infected at the rate `pi = c beta (I / A + J)` with the active population `A = S + E + I`. The
    first term is the infection by a new partner who is infectious, and the memory `J` is the infection by a current
    partner who turned infectious after the partnership formed, `dJ/dt = pi(t - T) S(t - T) / A - J / tau`, where
    partnerships last a mean time `tau`. The exposed `E` turn infectious after the fixed time `T`, and the infectious
    `I` develop the disease, which ends their sexual activity, at the rate `nu`, into `R`. The infection starts at time
    zero, so the infectious and the memory are zero before it.

    Args:
        label:
            The label of the population.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        S:
            The initial fraction of susceptible people.

        E:
            The initial fraction of exposed people.

        I:
            The initial fraction of infectious people.

        R:
            The initial fraction of people who developed the disease.

        J:
            The initial infection memory of current partnerships.

        c:
            The rate of acquiring new partners.

        beta:
            The probability of infection by an infectious partner over a partnership.

        tau:
            The mean duration of a partnership.

        T:
            The duration of the exposed period.

        nu:
            The rate at which the infectious develop the disease.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import PartnershipEpidemicElement
        >>> PartnershipEpidemicElement('population').tau
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        S: Annotated[float, StateSpec(domain=FRACTION)] = 0.99,
        E: Annotated[float, StateSpec(domain=FRACTION)] = 0.0,
        I: Annotated[float, StateSpec(domain=FRACTION, history=0.0)] = 0.01,
        R: Annotated[float, StateSpec(domain=FRACTION)] = 0.0,
        J: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), history=0.0)] = 0.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        beta: Annotated[float, ParameterSpec(domain=FRACTION)] = 0.2,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.S = S
        self.E = E
        self.I = I
        self.R = R
        self.J = J
        self.c = c
        self.beta = beta
        self.tau = tau
        self.T = T
        self.nu = nu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the infection by new and current partners, the exposed period, and the progression.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        active = s.S + s.E + s.I
        infection = s.c * s.beta * (s.I / active + s.J) * s.S
        past = [delayed(state, s.T) for state in (s.S, s.E, s.I, s.J)]
        past_active = past[0] + past[1] + past[2]
        matured = s.c * s.beta * (past[2] / past_active + past[3]) * past[0]
        return [
            RateLaw(s.S, -infection),
            RateLaw(s.E, infection - matured),
            RateLaw(s.I, matured - s.nu * s.I),
            RateLaw(s.R, s.nu * s.I),
            RateLaw(s.J, matured / active - s.J / s.tau),
        ]
