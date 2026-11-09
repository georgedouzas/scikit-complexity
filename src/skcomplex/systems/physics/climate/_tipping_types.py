"""Tipping types.

The module covers bifurcation-, noise-, and rate-induced tipping of a stochastic energy balance and a ramped fold.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import ramp
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.errors import ValidationError
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations, Noises


class FoldFollowerElement(Element):
    """Stochastic fold `dy = (-y³ + y + c) dt + sigma dW`.

    A leading element shifts its control.

    The fold has a lower and an upper stable branch while `|c| < 2 / (3 √3)`, and a coupling adds to its control.

    Args:
        label:
            The label of the follower.

        visual:
            How the follower is drawn, or `None` for `default_visual_`.

        y:
            The initial state.

        c:
            The control of the fold, without the coupling.

        sigma:
            The amplitude of the noise.

    Attributes:
        default_visual_:
            The visual of the follower when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.climate import FoldFollowerElement
        >>> FoldFollowerElement('follower').c
        0.0
    """

    law_classes = (
        NoiseLaw,
        RateLaw,
    )

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[float, StateSpec()] = -1.0,
        c: Annotated[float, ParameterSpec()] = 0.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.c = c
        self.sigma = sigma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the fold and its noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate law and the noise law of the state.
        """
        return [RateLaw(s.y, -(s.y**3) + s.y + s.c), NoiseLaw(s.y, s.sigma)]


class HopfFollowerElement(Element):
    """Stochastic Hopf normal form `dz = ((mu + i omega) z - |z|² z) dt + sigma dW`.

    A leader shifts its `mu`.

    The complex amplitude `z = y + i z_im` is held as two real states. For `mu < 0` it decays to a focus at the origin,
    and for `mu > 0` it settles on a limit cycle of radius `sqrt(mu)`, and a coupling adds to `mu`.

    Args:
        label:
            The label of the follower.

        visual:
            How the follower is drawn, or `None` for `default_visual_`.

        y:
            The initial real part of the amplitude.

        z:
            The initial imaginary part of the amplitude.

        mu:
            The distance from the Hopf bifurcation, without the coupling.

        omega:
            The frequency of the oscillation.

        sigma:
            The amplitude of the noise.

    Attributes:
        default_visual_:
            The visual of the follower when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.climate import HopfFollowerElement
        >>> HopfFollowerElement('follower').omega
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[float, StateSpec()] = 0.1,
        z: Annotated[float, StateSpec()] = 0.0,
        mu: Annotated[float, ParameterSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.z = z
        self.mu = mu
        self.omega = omega
        self.sigma = sigma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Hopf normal form and its noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws and the noise laws of the two parts of the amplitude.
        """
        radius = s.y**2 + s.z**2
        return [
            RateLaw(s.y, s.mu * s.y - s.omega * s.z - radius * s.y),
            RateLaw(s.z, s.omega * s.y + s.mu * s.z - radius * s.z),
            NoiseLaw(s.y, s.sigma),
            NoiseLaw(s.z, s.sigma),
        ]


class RampedFoldElement(Element):
    """Stochastic fold `dx = (a x^3 + b x + phi(t)) dt + sigma dW`.

    The control rises linearly from `phi0` to `phi1`.

    For `a < 0 < b` the fold has two stable branches for `|phi| < phi_c = sqrt(-4 b^3 / (27 a))`, and the lower
    branch vanishes at `phi = phi_c`. The control rises at the rate `rate` and stays at `phi1` once it reaches it.

    Args:
        label:
            The label of the system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial state.

        a:
            The cubic coefficient, negative.

        b:
            The linear coefficient, positive.

        sigma:
            The amplitude of the noise.

        phi0:
            The initial control.

        phi1:
            The final control.

        rate:
            The rate at which the control rises.

    Examples:
        >>> from skcomplex.systems.physics.climate import RampedFoldElement
        >>> RampedFoldElement('system').rate
        0.004
    """

    law_classes = (
        NoiseLaw,
        RateLaw,
    )

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = -1.3247,
        a: Annotated[float, ParameterSpec()] = -1.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        phi0: float = -1.0,
        phi1: float = 1.0,
        rate: float = 0.004,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.a = a
        self.b = b
        self.sigma = sigma
        self.phi0 = phi0
        self.phi1 = phi1
        self.rate = rate

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the fold with its ramped control and its noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        control = self.phi0 + ramp(s.t, 0.0, (self.phi1 - self.phi0) / self.rate, self.phi1 - self.phi0)
        return [RateLaw(s.x, s.a * s.x**3 + s.b * s.x + control), NoiseLaw(s.x, s.sigma)]


C0 = 1.10869e-8


C1 = 9.7137e9


REAL_TOLERANCE = 1e-9


TIPPING_TYPES = {
    'bifurcation': {'drift': 4e-4, 'b_ini': 1.04e-5, 'b_fin': 1.04e-5, 'a_ini': 0.2, 'nu': 0.0, 'lam': 0.0},
    'noise': {'drift': 0.0, 'b_ini': 1.69e-5, 'b_fin': 1.69e-5, 'a_ini': 0.6927, 'nu': 1.0, 'lam': 0.0},
    'rate': {'drift': 0.0, 'b_ini': 1.69e-5, 'b_fin': 1.835e-5, 'a_ini': 0.6927, 'nu': 0.0, 'lam': 1e-6},
}


FOLLOWERS = ('fold', 'hopf')


def _lower_branch(a: float, b: float, phi: float) -> float:
    """Return the lowest real root of `a x^3 + b x + phi`.

    The root is the lower branch of the fold.
    """
    roots = np.roots([a, 0.0, b, phi])
    return float(min(root.real for root in roots if abs(root.imag) < REAL_TOLERANCE))


class SuteraFraedrichTippingSystem(DynamicalSystem):
    """The Sutera-Fraedrich stochastic energy balance set up for tipping.

    The tipping is bifurcation-, noise-, or rate-induced.

    The temperature follows `dT = c0 [-T^4 + c1 mu (b2 T^2 - a2)] dt + sqrt(nu) dW` with the parameters of Table 1 of
    Fang et al. and starts at its warm stable equilibrium. In bifurcation-induced tipping, `mu` decreases at `4e-4` per
    year until the warm equilibrium meets the unstable one in a fold near `mu = 0.76`. In noise-induced tipping, the
    controls are fixed and noise of variance `nu = 1` carries the state over the unstable equilibrium. In rate-induced
    tipping, `b2` and `a2` ramp logistically at the rate `rho` with the equilibria preserved, and the state fails to
    track the moving warm equilibrium when `rho` exceeds about 0.175. A tipped state leaves the warm branch and falls to
    the other stable root of the quartic. The system has the temperature `earth.T`, the control `earth.mu`, and the ramp
    progress `earth.lam`.

    Args:
        tipping:
            The type of tipping, `'bifurcation'`, `'noise'`, or `'rate'`.

        rho:
            The rate of the logistic ramp of rate-induced tipping, per year.

        nu:
            The variance of the noise per year, or `None` for the value of the type, 1 for noise-induced tipping and
            0 otherwise.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The type of tipping is unknown.

    Examples:
        >>> from skcomplex.systems.physics.climate import SuteraFraedrichTippingSystem
        >>> round(SuteraFraedrichTippingSystem('rate').prepare().system_.elements[0].get_value('T'), 1)
        291.6
    """

    field = 'physics/climate'
    topics = ('tipping points', 'energy balance', 'sde', 'B-tipping', 'N-tipping', 'R-tipping')
    references = ('Fang, Wang, Kurths, and Fan 2025', 'Sutera 1981', 'Fraedrich 1978', 'Ashwin et al. 2012')
    until = 300.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.05)

    def __init__(
        self,
        tipping: str = 'noise',
        *,
        rho: float = 0.18,
        nu: float | None = None,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.tipping = tipping
        self.rho = rho
        self.nu = nu

    @staticmethod
    def _warm_state(mu: float, b2: float, a2: float) -> float:
        """Return the warm stable equilibrium of the energy balance.

        The equilibrium is the largest root of `T^4 = c1 mu (b2 T^2 - a2)`.
        """
        root = C1 * mu * b2 + math.sqrt((C1 * mu * b2) ** 2 - 4 * C1 * mu * a2)
        return math.sqrt(root / 2)

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'earth'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters of the type of tipping with their specs.

        Returns:
            variables:
                The temperature, the controls, and the parameters of the type of tipping.

        Raises:
            ValidationError:
                The type of tipping is unknown.
        """
        if self.tipping not in TIPPING_TYPES:
            error_msg = f'Unknown tipping type {self.tipping!r}. Use one of {", ".join(TIPPING_TYPES)}.'
            raise ValidationError(error_msg)
        values = TIPPING_TYPES[self.tipping]
        nonnegative = RealSet.closed_open(0, oo)
        return {
            'T': DeclaredVariableSpec(self._warm_state(1.0, values['b_ini'], values['a_ini']), StateSpec()),
            'mu': DeclaredVariableSpec(1.0, StateSpec(domain=nonnegative)),
            'lam': DeclaredVariableSpec(values['lam'], StateSpec(domain=nonnegative)),
            'drift': DeclaredVariableSpec(values['drift'], ParameterSpec()),
            'rho': DeclaredVariableSpec(self.rho if self.tipping == 'rate' else 0.0, ParameterSpec(domain=nonnegative)),
            'b_ini': DeclaredVariableSpec(values['b_ini'], ParameterSpec(domain=nonnegative)),
            'b_fin': DeclaredVariableSpec(values['b_fin'], ParameterSpec(domain=nonnegative)),
            'a_ini': DeclaredVariableSpec(values['a_ini'], ParameterSpec(domain=nonnegative)),
            'nu': DeclaredVariableSpec(values['nu'] if self.nu is None else self.nu, ParameterSpec(domain=nonnegative)),
            'c0': DeclaredVariableSpec(C0, ParameterSpec(domain=RealSet.open(0, oo))),
            'c1': DeclaredVariableSpec(C1, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the energy balance and the controls.

        The control `mu` drifts, and `b2` and `a2` follow a logistic ramp.
        """
        return {
            'T': self._balance,
            'mu': lambda s: -s.target.drift,
            'lam': lambda s: s.target.rho * s.target.lam * (1 - s.target.lam),
        }

    def _get_noises(self) -> Noises:
        """Return the noise amplitude of the temperature."""
        return {'T': lambda s: s.target.nu**0.5}

    @staticmethod
    def _balance(s: SimpleNamespace) -> Expression:
        """Return the absorbed minus the emitted radiation, with `b2` and `a2` on their ramp."""
        earth = s.target
        b2 = earth.b_ini + earth.lam * (earth.b_fin - earth.b_ini)
        a2 = earth.a_ini + earth.c1 * earth.mu * (b2**2 - earth.b_ini**2) / 4
        return earth.c0 * (-(earth.T**4) + earth.c1 * earth.mu * (b2 * earth.T**2 - a2))


class RampedStochasticFoldSystem(BaseSystem):
    """A stochastic fold whose control is ramped through a tipping point.

    The tipping point is that of its lower branch.

    The state follows `dx = (a x^3 + b x + phi(t)) dt + sigma dW` with `phi = phi0 + rate t` until it reaches `phi1`,
    Eq. 4 of Fang et al. with `phi = -1 + 0.004 t`. It starts on the lower branch, which vanishes at the static fold
    `phi_c = sqrt(-4 b^3 / (27 a))`, `2 / (3 sqrt(3))` for `a = -1` and `b = 1`. Near the fold, recovery slows down, so
    the variance and the lag-1 autocorrelation of the state rise. A ramp faster than the relaxation carries the control
    past `phi_c` before the state tips, by a lag that grows with the rate. The system has the state `system.x`, starting
    on the lower branch at `phi0`.

    Args:
        rate:
            The rate at which the control rises.

        sigma:
            The amplitude of the noise.

        a:
            The cubic coefficient, negative.

        b:
            The linear coefficient, positive.

        phi0:
            The initial control.

        phi1:
            The final control.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import RampedStochasticFoldSystem
        >>> RampedStochasticFoldSystem(rate=0.01).prepare().system_.elements[0].rate
        0.01
    """

    field = 'physics/climate'
    topics = ('tipping points', 'fold', 'ramp', 'sde', 'early warning signals', 'critical slowing down')
    references = ('Fang, Wang, Kurths, and Fan 2025', 'Scheffer et al. 2009')
    until = 500.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        *,
        rate: float = 0.004,
        sigma: float = 0.02,
        a: float = -1.0,
        b: float = 1.0,
        phi0: float = -1.0,
        phi1: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.rate = rate
        self.sigma = sigma
        self.a = a
        self.b = b
        self.phi0 = phi0
        self.phi1 = phi1

    def _build_system(self) -> CompositeSystem:
        """Return a stochastic fold whose control is ramped through a tipping point.

        The tipping point is that of its lower branch.
        """
        fold = RampedFoldElement(
            'system',
            x=_lower_branch(self.a, self.b, self.phi0),
            a=self.a,
            b=self.b,
            sigma=self.sigma,
            phi0=self.phi0,
            phi1=self.phi1,
            rate=self.rate,
        )
        return CompositeSystem(self.label, elements=[fold])


class LeadingFollowingCascadeSystem(BaseSystem):
    """A cascade of two tipping elements.

    A leading fold is driven by a ramp and a follower is forced by the leader.

    The leading element follows `dx = (-x³ + x + phi(t)) dt + sigma dW`, whose control rises as `phi = phi0 + rate t`
    until `phi1`, and it tips from its lower to its upper branch when `phi` passes `2 / (3 √3)`. The following element
    is forced linearly by the state of the leader, Eq. 10 of Fang et al. after Dekker et al. A fold follower obeys `dy =
    (-y³ + y + c + coupling x) dt + sigma dW`, and it tips too, a domino cascade, when the tipped leader moves its
    control `c + coupling x` past `2 / (3 √3)`. A Hopf follower obeys `dw = ((mu + coupling x + i omega) w - |w|² w) dt
    + sigma dW`, and the tipped leader can carry it through its Hopf bifurcation onto an oscillation of radius `sqrt(mu
    + coupling x)`. The follower can show rising variance and autocorrelation before the leader tips even when it stays
    far from its own tipping point. The system has the leader `leader.x` and the follower `follower.y`, and also
    `follower.z` for the Hopf follower.

    Args:
        follower:
            The normal form of the following element, `'fold'` or `'hopf'`.

        coupling:
            The strength of the linear forcing of the follower by the leader.

        rate:
            The rate at which the control of the leader rises.

        sigma:
            The amplitude of the noise of both elements.

        phi0:
            The initial control of the leader.

        phi1:
            The final control of the leader.

        c:
            The control of a fold follower, without the coupling.

        mu:
            The distance of a Hopf follower from its bifurcation, without the coupling.

        omega:
            The frequency of a Hopf follower.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The follower is neither `'fold'` nor `'hopf'`.

    Examples:
        >>> from skcomplex.systems.physics.climate import LeadingFollowingCascadeSystem
        >>> LeadingFollowingCascadeSystem('hopf').prepare().system_.elements[1].omega
        1.0
    """

    field = 'physics/climate'
    topics = ('tipping points', 'cascades', 'fold', 'hopf bifurcation', 'ramp', 'sde')
    references = ('Dekker, von der Heydt, and Dijkstra 2018', 'Fang, Wang, Kurths, and Fan 2025')
    until = 500.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        follower: str = 'fold',
        *,
        coupling: float = 0.4,
        rate: float = 0.004,
        sigma: float = 0.02,
        phi0: float = -1.0,
        phi1: float = 1.0,
        c: float = 0.0,
        mu: float = 0.0,
        omega: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.follower = follower
        self.coupling = coupling
        self.rate = rate
        self.sigma = sigma
        self.phi0 = phi0
        self.phi1 = phi1
        self.c = c
        self.mu = mu
        self.omega = omega

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        if self.follower not in FOLLOWERS:
            error_msg = f'Unknown follower {self.follower!r}. Use one of {", ".join(FOLLOWERS)}.'
            raise ValidationError(error_msg)
        start = _lower_branch(-1.0, 1.0, self.phi0)
        leader = RampedFoldElement('leader', x=start, sigma=self.sigma, phi0=self.phi0, phi1=self.phi1, rate=self.rate)
        if self.follower == 'fold':
            follows: Element = FoldFollowerElement(
                'follower',
                y=_lower_branch(-1.0, 1.0, self.c + self.coupling * start),
                c=self.c,
                sigma=self.sigma,
            )
            forcing = [
                RateInteraction(
                    'forcing',
                    target='follower',
                    state='y',
                    sources=['leader'],
                    rate=lambda s: self.coupling * s.sources[0].x,
                ),
            ]
        else:
            follows = HopfFollowerElement('follower', mu=self.mu, omega=self.omega, sigma=self.sigma)
            forcing = [
                RateInteraction(
                    'forcing_y',
                    target='follower',
                    state='y',
                    sources=['leader'],
                    rate=lambda s: self.coupling * s.sources[0].x * s.target.y,
                ),
                RateInteraction(
                    'forcing_z',
                    target='follower',
                    state='z',
                    sources=['leader'],
                    rate=lambda s: self.coupling * s.sources[0].x * s.target.z,
                ),
            ]
        return CompositeSystem(self.label, elements=[leader, follows], interactions=forcing)
