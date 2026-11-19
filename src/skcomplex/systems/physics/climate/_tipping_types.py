"""Tipping types.

The module covers bifurcation-, noise-, and rate-induced tipping of a stochastic energy balance and a ramped fold.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.laws.utils import ramp
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role

C0 = 1.10869e-8
C1 = 9.7137e9
REAL_TOLERANCE = 1e-9
TIPPING_TYPES = {
    'bifurcation': {'drift': 4e-4, 'b_ini': 1.04e-5, 'b_fin': 1.04e-5, 'a_ini': 0.2, 'nu': 0.0, 'lam': 0.0},
    'noise': {'drift': 0.0, 'b_ini': 1.69e-5, 'b_fin': 1.69e-5, 'a_ini': 0.6927, 'nu': 1.0, 'lam': 0.0},
    'rate': {'drift': 0.0, 'b_ini': 1.69e-5, 'b_fin': 1.835e-5, 'a_ini': 0.6927, 'nu': 0.0, 'lam': 1e-6},
}
FOLLOWERS = ('fold', 'hopf')
NONNEGATIVE = RealSet.closed_open(0, oo)


class FoldFollowerElement(Element):
    """Follower of a cascade in the normal form of a fold, its state `y`.

    Args:
        label:
            The label of the follower.

        y:
            The initial state.

    Examples:
        >>> from skcomplex.systems.physics.climate import FoldFollowerElement
        >>> FoldFollowerElement('follower').y
        -1.0
    """

    def __init__(self, label: str, *, y: float = -1.0) -> None:
        super().__init__(label)
        self.y = y


class FoldFollowerInteraction(Interaction):
    """Stochastic fold `dy = (-y³ + y + c + coupling x) dt + sigma dW` of a follower forced by a leader.

    The fold has a lower and an upper stable branch while `|c + coupling x| < 2 / (3 √3)`, and the state `x` of the
    leader shifts its control.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the follower.

        leader:
            The path of the leader.

        c:
            The control of the fold, without the coupling.

        coupling:
            The strength of the linear forcing by the leader.

        sigma:
            The amplitude of the noise.

    Examples:
        >>> from skcomplex.systems.physics.climate import FoldFollowerInteraction
        >>> FoldFollowerInteraction('fold', target='follower', leader='leader').c
        0.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        leader: Role,
        c: float = 0.0,
        coupling: float = 0.4,
        sigma: Annotated[float, NONNEGATIVE] = 0.02,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.leader = leader
        self.c = c
        self.coupling = coupling
        self.sigma = sigma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the forced fold of the follower and its noise.

        Args:
            context:
                The namespace of the follower and the leader, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            y = context.target.y
            rate = -(y**3) + y + context.c + context.coupling * context.leader.x
            self.laws_ = [RateLaw(y.diff(context.t), rate), NoiseLaw(y, context.sigma)]


class HopfFollowerElement(Element):
    """Follower of a cascade in the Hopf normal form, whose complex amplitude is `y + i z`.

    Args:
        label:
            The label of the follower.

        y:
            The initial real part of the amplitude.

        z:
            The initial imaginary part of the amplitude.

    Examples:
        >>> from skcomplex.systems.physics.climate import HopfFollowerElement
        >>> HopfFollowerElement('follower').y
        0.1
    """

    def __init__(self, label: str, *, y: float = 0.1, z: float = 0.0) -> None:
        super().__init__(label)
        self.y = y
        self.z = z


class HopfFollowerInteraction(Interaction):
    """Stochastic Hopf normal form of a follower whose distance from the bifurcation a leader shifts.

    The complex amplitude `w = y + i z` follows `dw = ((mu + coupling x + i omega) w - |w|² w) dt + sigma dW`, where `x`
    is the state of the leader. For `mu + coupling x < 0` it decays to a focus at the origin, and for `mu + coupling x >
    0` it settles on a limit cycle of radius `sqrt(mu + coupling x)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the follower.

        leader:
            The path of the leader.

        mu:
            The distance from the Hopf bifurcation, without the coupling.

        omega:
            The frequency of the oscillation.

        coupling:
            The strength of the linear forcing by the leader.

        sigma:
            The amplitude of the noise.

    Examples:
        >>> from skcomplex.systems.physics.climate import HopfFollowerInteraction
        >>> HopfFollowerInteraction('hopf', target='follower', leader='leader').omega
        1.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        leader: Role,
        mu: float = 0.0,
        omega: float = 1.0,
        coupling: float = 0.4,
        sigma: Annotated[float, NONNEGATIVE] = 0.02,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.leader = leader
        self.mu = mu
        self.omega = omega
        self.coupling = coupling
        self.sigma = sigma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the forced Hopf normal form of the follower and its noise.

        Args:
            context:
                The namespace of the follower and the leader, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            y, z, t = context.target.y, context.target.z, context.t
            radius = y**2 + z**2
            forcing = context.coupling * context.leader.x
            self.laws_ = [
                RateLaw(y.diff(t), context.mu * y - context.omega * z - radius * y + forcing * y),
                RateLaw(z.diff(t), context.omega * y + context.mu * z - radius * z + forcing * z),
                NoiseLaw(y, context.sigma),
                NoiseLaw(z, context.sigma),
            ]


class RampedFoldElement(Element):
    """Element in the normal form of a fold whose control is ramped, its state `x`.

    Args:
        label:
            The label of the element.

        x:
            The initial state.

    Examples:
        >>> from skcomplex.systems.physics.climate import RampedFoldElement
        >>> RampedFoldElement('system').x
        -1.3247
    """

    def __init__(self, label: str, *, x: float = -1.3247) -> None:
        super().__init__(label)
        self.x = x


class RampedFoldInteraction(Interaction):
    """Stochastic fold `dx = (a x^3 + b x + phi(t)) dt + sigma dW` whose control rises from `phi0` to `phi1`.

    For `a < 0 < b` the fold has two stable branches for `|phi| < phi_c = sqrt(-4 b^3 / (27 a))`, and the lower
    branch vanishes at `phi = phi_c`. The control rises at the rate `rate` and stays at `phi1` once it reaches it.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element.

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
        >>> from skcomplex.systems.physics.climate import RampedFoldInteraction
        >>> RampedFoldInteraction('fold', target='system').rate
        0.004
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        a: float = -1.0,
        b: float = 1.0,
        sigma: Annotated[float, NONNEGATIVE] = 0.02,
        phi0: float = -1.0,
        phi1: float = 1.0,
        rate: float = 0.004,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.a = a
        self.b = b
        self.sigma = sigma
        self.phi0 = phi0
        self.phi1 = phi1
        self.rate = rate

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the fold with its ramped control and its noise.

        Args:
            context:
                The namespace of the element, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            control = self.phi0 + ramp(t, 0.0, (self.phi1 - self.phi0) / self.rate, self.phi1 - self.phi0)
            self.laws_ = [RateLaw(x.diff(t), context.a * x**3 + context.b * x + control), NoiseLaw(x, context.sigma)]


class StochasticEnergyBalanceInteraction(Interaction):
    """Stochastic energy balance of Sutera and Fraedrich, with a drifting and a ramped control.

    The temperature `T` of the target follows `dT = c0 [-T^4 + c1 mu (b2 T^2 - a2)] dt + sqrt(nu) dW`, the control `mu`
    decreases at the rate `drift`, and the ramp progress `lam` grows logistically at the rate `rho`, with `b2 = b_ini +
    lam (b_fin - b_ini)` and `a2 = a_ini + c1 mu (b2^2 - b_ini^2) / 4`, so the equilibria are kept along the ramp.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the planet.

        drift:
            The rate at which the control `mu` decreases.

        rho:
            The rate of the logistic ramp.

        b_ini:
            The initial value of `b2`.

        b_fin:
            The final value of `b2`.

        a_ini:
            The initial value of `a2`.

        nu:
            The variance of the noise per unit of time.

        c0:
            The inverse heat capacity.

        c1:
            The scale of the absorbed radiation.

    Examples:
        >>> from skcomplex.systems.physics.climate import StochasticEnergyBalanceInteraction
        >>> StochasticEnergyBalanceInteraction('balance', target='earth').nu
        0.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        drift: float = 0.0,
        rho: Annotated[float, NONNEGATIVE] = 0.0,
        b_ini: Annotated[float, NONNEGATIVE] = 1.69e-5,
        b_fin: Annotated[float, NONNEGATIVE] = 1.69e-5,
        a_ini: Annotated[float, NONNEGATIVE] = 0.6927,
        nu: Annotated[float, NONNEGATIVE] = 0.0,
        c0: Annotated[float, RealSet.open(0, oo)] = C0,
        c1: Annotated[float, RealSet.open(0, oo)] = C1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.drift = drift
        self.rho = rho
        self.b_ini = b_ini
        self.b_fin = b_fin
        self.a_ini = a_ini
        self.nu = nu
        self.c0 = c0
        self.c1 = c1

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the energy balance, the drift of the control, the ramp, and the noise of the temperature.

        Args:
            context:
                The namespace of the planet, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            T, mu, lam, t = context.target.T, context.target.mu, context.target.lam, context.t
            b2 = context.b_ini + lam * (context.b_fin - context.b_ini)
            a2 = context.a_ini + context.c1 * mu * (b2**2 - context.b_ini**2) / 4
            self.laws_ = [
                RateLaw(T.diff(t), context.c0 * (-(T**4) + context.c1 * mu * (b2 * T**2 - a2))),
                RateLaw(mu.diff(t), -context.drift),
                RateLaw(lam.diff(t), context.rho * lam * (1 - lam)),
                NoiseLaw(T, context.nu**0.5),
            ]


class SuteraFraedrichTippingSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The type of tipping is unknown.

    Examples:
        >>> from skcomplex.systems.physics.climate import SuteraFraedrichTippingSystem
        >>> round(SuteraFraedrichTippingSystem('rate').prepare().system_.elements[0].values['T'], 1)
        291.6
    """

    field = 'physics/climate'
    topics = ('tipping points', 'energy balance', 'sde', 'B-tipping', 'N-tipping', 'R-tipping')
    references = ('Fang, Wang, Kurths, and Fan 2025', 'Sutera 1981', 'Fraedrich 1978', 'Ashwin et al. 2012')
    until = 300.0
    dt = 0.5

    def __init__(
        self,
        tipping: str = 'noise',
        *,
        rho: float = 0.18,
        nu: float | None = None,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the planet at its warm equilibrium and its energy balance.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The type of tipping is unknown.
        """
        super()._prepare(context)
        if self.tipping not in TIPPING_TYPES:
            error_msg = f'Unknown tipping type {self.tipping!r}. Use one of {", ".join(TIPPING_TYPES)}.'
            raise ValidationError(error_msg)
        values = TIPPING_TYPES[self.tipping]
        earth = Element(
            'earth',
            values={'T': self._warm_state(1.0, values['b_ini'], values['a_ini']), 'mu': 1.0, 'lam': values['lam']},
        )
        balance = StochasticEnergyBalanceInteraction(
            'balance',
            target='earth',
            drift=values['drift'],
            rho=self.rho if self.tipping == 'rate' else 0.0,
            b_ini=values['b_ini'],
            b_fin=values['b_fin'],
            a_ini=values['a_ini'],
            nu=values['nu'] if self.nu is None else self.nu,
        )
        self.system_ = System(self.label, elements=[earth], interactions=[balance], solver=EulerMaruyamaSolver(dt=0.05))


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.climate import RampedStochasticFoldSystem
        >>> RampedStochasticFoldSystem(rate=0.01).prepare().system_.interactions[0].rate
        0.01
    """

    field = 'physics/climate'
    topics = ('tipping points', 'fold', 'ramp', 'sde', 'early warning signals', 'critical slowing down')
    references = ('Fang, Wang, Kurths, and Fan 2025', 'Scheffer et al. 2009')
    until = 500.0
    dt = 0.5

    @staticmethod
    def _lower_branch(a: float, b: float, phi: float) -> float:
        """Return the lowest real root of `a x^3 + b x + phi`.

        The root is the lower branch of the fold.
        """
        roots = np.roots([a, 0.0, b, phi])
        return float(min(root.real for root in roots if abs(root.imag) < REAL_TOLERANCE))

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.rate = rate
        self.sigma = sigma
        self.a = a
        self.b = b
        self.phi0 = phi0
        self.phi1 = phi1

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a stochastic fold ramped through the tipping point of its lower branch.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        fold = RampedFoldElement('system', x=RampedStochasticFoldSystem._lower_branch(self.a, self.b, self.phi0))
        ramped = RampedFoldInteraction(
            'fold',
            target='system',
            a=self.a,
            b=self.b,
            sigma=self.sigma,
            phi0=self.phi0,
            phi1=self.phi1,
            rate=self.rate,
        )
        self.system_ = System(self.label, elements=[fold], interactions=[ramped], solver=EulerMaruyamaSolver(dt=0.01))


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The follower is neither `'fold'` nor `'hopf'`.

    Examples:
        >>> from skcomplex.systems.physics.climate import LeadingFollowingCascadeSystem
        >>> LeadingFollowingCascadeSystem('hopf').prepare().system_.interactions[1].omega
        1.0
    """

    field = 'physics/climate'
    topics = ('tipping points', 'cascades', 'fold', 'hopf bifurcation', 'ramp', 'sde')
    references = ('Dekker, von der Heydt, and Dijkstra 2018', 'Fang, Wang, Kurths, and Fan 2025')
    until = 500.0
    dt = 0.5

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.follower = follower
        self.coupling = coupling
        self.rate = rate
        self.sigma = sigma
        self.phi0 = phi0
        self.phi1 = phi1
        self.c = c
        self.mu = mu
        self.omega = omega

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the ramped leader and its forced follower.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The follower is neither `'fold'` nor `'hopf'`.
        """
        super()._prepare(context)
        if self.follower not in FOLLOWERS:
            error_msg = f'Unknown follower {self.follower!r}. Use one of {", ".join(FOLLOWERS)}.'
            raise ValidationError(error_msg)
        start = RampedStochasticFoldSystem._lower_branch(-1.0, 1.0, self.phi0)
        leader = RampedFoldElement('leader', x=start)
        ramped = RampedFoldInteraction(
            'leader_fold',
            target='leader',
            sigma=self.sigma,
            phi0=self.phi0,
            phi1=self.phi1,
            rate=self.rate,
        )
        follows: Element
        forcing: Interaction
        if self.follower == 'fold':
            follows = FoldFollowerElement(
                'follower',
                y=RampedStochasticFoldSystem._lower_branch(-1.0, 1.0, self.c + self.coupling * start),
            )
            forcing = FoldFollowerInteraction(
                'forcing',
                target='follower',
                leader='leader',
                c=self.c,
                coupling=self.coupling,
                sigma=self.sigma,
            )
        else:
            follows = HopfFollowerElement('follower')
            forcing = HopfFollowerInteraction(
                'forcing',
                target='follower',
                leader='leader',
                mu=self.mu,
                omega=self.omega,
                coupling=self.coupling,
                sigma=self.sigma,
            )
        self.system_ = System(
            self.label,
            elements=[leader, follows],
            interactions=[ramped, forcing],
            solver=EulerMaruyamaSolver(dt=0.01),
        )
