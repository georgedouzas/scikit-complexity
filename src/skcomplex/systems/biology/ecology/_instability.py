"""Jensen's coupled populations that lose the stability of their steady state.

They lose it from within.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import bernoulli, clip, normal, where
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class UnstablePopulationsElement(Element):
    """Resident population `x` and invading population `y`.

    They follow equations (13.3) and (13.4) of Jensen.

    The maps of `IntrinsicInstabilitySystem` step both sizes. At every step `x ← [x - eta (x - X)] exp(-alpha y) + √x
    sigma chi`, an Ornstein-Uhlenbeck step damped by the invader. With probability `p` a mutation from `x` adds `nu x`
    to `y`, and otherwise `y ← y (1 + beta) + √y sigma chi'`. The noises `chi` and `chi'` are independent standard
    normal values, scaled by the square root of the population sizes as in birth and death processes, and both sizes are
    cut at 0.

    Args:
        label:
            The label of the populations.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial resident population.

        y:
            The initial invading population.

        eta:
            The restoring rate `eta` of the resident population.

        mean:
            The level `X` the resident population returns to.

        alpha:
            The suppression `alpha` of the resident population by the invader.

        beta:
            The growth rate `beta` of the invader.

        nu:
            The share `nu` of the resident population that a mutation moves to the invader.

        p:
            The probability of a mutation per step.

        sigma:
            The amplitude `sigma` of the noise.

    Examples:
        >>> from skcomplex.systems.biology.ecology import UnstablePopulationsElement
        >>> UnstablePopulationsElement('populations').p
        0.001
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.1,
        y: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        mean: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.eta = eta
        self.mean = mean
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.p = p
        self.sigma = sigma


class IntrinsicInstabilitySystem(BaseSystem):
    """Jensen's coupled populations, whose steady state becomes unstable.

    It becomes unstable without a change of the surroundings.

    The resident population fluctuates about its level `X = 1` until a rare mutation seeds the invader. The invader then
    grows by the factor `1 + beta` per step and suppresses the resident, so the trajectory leaves the fixed point `(X,
    0)` along its unstable direction, with eigenvalue `beta`, and the resident declines although no parameter has
    changed. The system has the populations `populations.x` and `populations.y`.

    Args:
        x:
            The initial resident population.

        eta:
            The restoring rate of the resident population.

        alpha:
            The suppression of the resident population by the invader.

        beta:
            The growth rate of the invader.

        nu:
            The share of the resident population that a mutation moves to the invader.

        p:
            The probability of a mutation per step.

        sigma:
            The amplitude of the noise.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import IntrinsicInstabilitySystem
        >>> IntrinsicInstabilitySystem().prepare().system_.elements[0].beta
        0.01
    """

    draws = True

    field = 'biology/ecology'
    topics = ('tipping point', 'stochastic', 'discrete map', 'early warning')
    references = ('Jensen 2023',)
    until = 2000
    dt = 10

    def __init__(
        self,
        *,
        x: float = 1.1,
        eta: float = 0.1,
        alpha: float = 0.01,
        beta: float = 0.01,
        nu: float = 0.01,
        p: float = 0.001,
        sigma: float = 0.01,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.p = p
        self.sigma = sigma

    @staticmethod
    def _get_resident(namespace: SimpleNamespace) -> Expression:
        """Return the next resident population.

        It is damped towards its mean by the invader, with noise, and cut at 0.
        """
        s = namespace.target
        resident = (s.x - s.eta * (s.x - s.mean)) * exp(-s.alpha * s.y) + sqrt(s.x) * s.sigma * normal('chi_x')
        return clip(resident, low=0)

    @staticmethod
    def _get_invader(namespace: SimpleNamespace) -> Expression:
        """Return the next invading population.

        It grows with noise or is fed by a mutation of the resident, and is cut at 0.
        """
        s = namespace.target
        grown = s.y * (1 + s.beta) + sqrt(s.y) * s.sigma * normal('chi_y')
        return clip(where(bernoulli('mutation', s.p) > 0, s.y + s.nu * s.x, grown), low=0)

    def _build_system(self) -> System:
        """Return the parts of Jensen's coupled populations."""
        populations = UnstablePopulationsElement(
            'populations',
            x=self.x,
            eta=self.eta,
            alpha=self.alpha,
            beta=self.beta,
            nu=self.nu,
            p=self.p,
            sigma=self.sigma,
        )
        return System(
            self.label,
            elements=[populations],
            interactions=[
                MapInteraction('resident', target='populations', state='x', function=self._get_resident),
                MapInteraction('invader', target='populations', state='y', function=self._get_invader),
            ],
        )
