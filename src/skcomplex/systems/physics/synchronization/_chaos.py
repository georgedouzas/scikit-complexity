"""Chaotic synchronization systems.

The module covers coupled Rössler oscillators and logistic maps coupled by aggregate averaging.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import DeclaredElement, Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw, UpdateLaw
from ....core.components.laws.utils import mod
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.specs import DeclaredVariableSpec, Role, StateSpec
from ....core.systems import BaseSystem, System

MAPS = ('logistic', 'bernoulli')
PAIR = 2


class DelayedUnitElement(DeclaredElement):
    """Value `x` of a map that keeps its last `delay` values.

    A delayed coupling reads the values.

    The state `x` has `delay` lags, `x_lag1` to `x_lag<delay>`, and before the start every lag holds the initial value.

    Args:
        label:
            The label of the unit.

        value:
            The initial value of `x`, also its value before the start.

        delay:
            The number of past values the unit keeps, at least 1.

        variables:
            The declared variables that replace the derived ones, or `None` for none.

    Attributes:
        default_visual_:
            How the unit is drawn by default, the `visual` of its class.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import DelayedUnitElement
        >>> DelayedUnitElement('unit', delay=2).get_value('x')
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        value: float = 0.3,
        delay: int = 1,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(label, variables=variables)
        self.value = value
        self.delay = delay

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the value `x` with its `delay` lags, and the replacing variables.

        The lags all hold the initial value.
        """
        lagged = StateSpec(lags=self.delay, initial=(self.value,) * self.delay)
        return {'x': DeclaredVariableSpec(self.value, lagged), **(self.variables or {})}


class LogisticUnitElement(Element):
    """Value `x` in `[0, 1]` of a logistic map.

    The maps of the coupling iterate the value.

    Args:
        label:
            The label of the unit.

        x:
            The initial value.

        values:
            Further values by name, such as the past values `'x[-1]'` that a delayed coupling reads, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import LogisticUnitElement
        >>> LogisticUnitElement('unit').x
        0.3
    """

    def __init__(self, label: str, *, x: float = 0.3, values: Mapping[str, object] | None = None) -> None:
        super().__init__(label, values=values)
        self.x = x


class RosslerOscillatorElement(Element):
    """Rössler oscillator with the three variables `x1`, `x2`, and `x3`.

    `RosslerFlowInteraction` moves the variables.

    Args:
        label:
            The label of the oscillator.

        x1:
            The initial first variable.

        x2:
            The initial second variable.

        x3:
            The initial third variable.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import RosslerOscillatorElement
        >>> RosslerOscillatorElement('oscillator', x1=2.0).x1
        2.0
    """

    def __init__(self, label: str, *, x1: float = 1.0, x2: float = 0.0, x3: float = 0.0) -> None:
        super().__init__(label)
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3


class RosslerFlowInteraction(Interaction):
    """Rössler flow `x1' = -omega x2 - x3`, `x2' = omega x1 + a x2`, `x3' = b + x3 (x1 - c)`.

    The flow is chaotic for `a = b = 0.2` and `c = 5.7`. The time scale `p` divides every rate, `p x' = f(x)`, so `p <
    1` speeds the oscillator up.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the oscillator whose variables `x1`, `x2`, and `x3` the flow moves.

        omega:
            The natural frequency.

        a:
            The first Rössler parameter.

        b:
            The second Rössler parameter.

        c:
            The third Rössler parameter.

        p:
            The time scale, which divides every rate.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import RosslerFlowInteraction
        >>> RosslerFlowInteraction('flow', target='oscillator').c
        5.7
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        omega: float = 1.0,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        p: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.omega = omega
        self.a = a
        self.b = b
        self.c = c
        self.p = p

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the three variables of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x1, x2, x3 = s.target.x1, s.target.x2, s.target.x3
            self.laws_ = [
                RateLaw(x1.diff(t), (-s.omega * x2 - x3) / s.p),
                RateLaw(x2.diff(t), (s.omega * x1 + s.a * x2) / s.p),
                RateLaw(x3.diff(t), (s.b + x3 * (x1 - s.c)) / s.p),
            ]


class LorenzFlowInteraction(Interaction):
    """Lorenz flow `x' = sigma (y - x)`, `y' = x (rho - z) - y`, `z' = x y - beta z`.

    For `sigma = 10`, `rho = 28`, and `beta = 8/3` the runs never settle and never repeat, the strange attractor of
    Lorenz.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element whose values `x`, `y`, and `z` the flow moves.

        sigma:
            The Prandtl number.

        rho:
            The Rayleigh number relative to its critical value.

        beta:
            The geometric factor.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import LorenzFlowInteraction
        >>> LorenzFlowInteraction('flow', target='response', beta=2.666).beta
        2.666
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the three values of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x, y, z = s.target.x, s.target.y, s.target.z
            self.laws_ = [
                RateLaw(x.diff(t), s.sigma * (y - x)),
                RateLaw(y.diff(t), x * (s.rho - z) - y),
                RateLaw(z.diff(t), x * y - s.beta * z),
            ]


class AggregateAveragingInteraction(Interaction):
    """Logistic or Bernoulli map of an aggregate of a target value and a source value.

    The value of the source is possibly delayed.

    The map sets `x_t(t + 1) = f(a)` with the aggregate `a = (1 - kappa) x_t(t) + kappa x_s(t - delay)`, and with the
    logistic map `f(a) = r a (1 - a)` or the Bernoulli shift `f(a) = r a mod 1`. A delayed source reads its past values.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        source:
            The path of the unit whose value the aggregate reads.

        target:
            The path of the unit whose value the map sets.

        r:
            The growth rate of the logistic map.

        kappa:
            The weight of the source in the aggregate.

        kind:
            The map, `'logistic'` or `'bernoulli'`.

        delay:
            The number of steps by which the source is delayed.

    Raises:
        ValidationError:
            The map is unknown, or the delay is not a nonnegative integer.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AggregateAveragingInteraction
        >>> AggregateAveragingInteraction('averaging', source='a', target='b', kind='bernoulli').kind
        'bernoulli'
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        source: Role,
        target: Role,
        r: Annotated[float, RealSet.closed_open(0, oo)] = 4.0,
        kappa: Annotated[float, RealSet.closed_open(0, oo)] = 0.4,
        kind: str = 'logistic',
        delay: int = 0,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.r = r
        self.kappa = kappa
        self.kind = kind
        self.delay = delay

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the map of the aggregate `a = (1 - kappa) x_t + kappa x_s(t - delay)`.

        Args:
            context:
                The namespace of the source, the target, the time `t`, and the parameters.

        Raises:
            ValidationError:
                The map is unknown, or the delay is not a nonnegative integer.
        """
        super()._prepare(context)
        if self.kind not in MAPS:
            error_msg = f'Unknown map {self.kind!r}. Use one of {", ".join(MAPS)}.'
            raise ValidationError(error_msg)
        if isinstance(self.delay, bool) or not isinstance(self.delay, int) or self.delay < 0:
            error_msg = f'The delay of {self.label!r} is {self.delay!r}. Give a nonnegative integer.'
            raise ValidationError(error_msg)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            source = context.source.x(t=t - self.delay) if self.delay else context.source.x
            aggregate = (1 - context.kappa) * x + context.kappa * source
            if self.kind == 'bernoulli':
                self.laws_ = [UpdateLaw(x, mod(context.r * aggregate, 1))]
                return
            self.laws_ = [UpdateLaw(x, context.r * aggregate * (1 - aggregate))]


class DiffusiveCouplingInteraction(Interaction):
    """Diffusive coupling of a state of one element to the same state of another.

    The coupling is `x_t' += k (x_s - x_t)`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        source:
            The path of the element whose state the coupling reads.

        target:
            The path of the element whose state the coupling changes.

        state:
            The name of the coupled state.

        k:
            The coupling strength.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import DiffusiveCouplingInteraction
        >>> DiffusiveCouplingInteraction('coupling', source='a', target='b', k=0.5).k
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        source: Role,
        target: Role,
        state: str = 'x2',
        k: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.state = state
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate `k (x_s - x_t)` of the state of the target.

        Args:
            context:
                The namespace of the source, the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = getattr(context.target, self.state)
            self.laws_ = [RateLaw(x.diff(context.t), context.k * (getattr(context.source, self.state) - x))]


class DriveInteraction(Interaction):
    """Drive of a state of the target by a state of the source, `x_t' += k x_s`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        source:
            The path of the element that drives.

        target:
            The path of the element that is driven.

        read:
            The name of the state of the source that drives.

        write:
            The name of the state of the target whose rate the drive adds to.

        k:
            The strength of the drive.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import DriveInteraction
        >>> DriveInteraction('coupling', source='drive', target='response', k=8.0).k
        8.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        source: Role,
        target: Role,
        read: str = 'x2',
        write: str = 'y',
        k: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.target = target
        self.read = read
        self.write = write
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate `k x_s` of the state of the target.

        Args:
            context:
                The namespace of the source, the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = getattr(context.target, self.write)
            self.laws_ = [RateLaw(x.diff(context.t), context.k * getattr(context.source, self.read))]


class CoupledRosslerSystem(BaseSystem):
    """Chaotic Rössler oscillators on a ring.

    The oscillators are diffusively coupled through their second variable.

    Oscillator `i` follows `x1' = -ω_i x2 - x3`, `x2' = ω_i x1 + a x2 + k sum_j (x2_j - x2_i)`, and `x3' = b + x3 (x1 -
    c)`, summed over its neighbours on the ring. Two oscillators with frequencies `1 ± mismatch` pass from no
    synchronization to chaotic phase synchronization as `k` grows. Identical oscillators, with no mismatch, synchronize
    completely when every nonzero eigenvalue of the ring's Laplacian times `k` falls where the master stability function
    is negative. The system has the states `oscillator_<i>.x1`, `oscillator_<i>.x2`, and `oscillator_<i>.x3`.

    Args:
        n_oscillators:
            The number of oscillators on the ring.

        k:
            The coupling strength.

        mismatch:
            The half width of the interval of natural frequencies around 1.

        a:
            The first Rössler parameter.

        b:
            The second Rössler parameter.

        c:
            The third Rössler parameter.

        seed:
            The seed of the initial states.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import CoupledRosslerSystem
        >>> trajectory = CoupledRosslerSystem(k=0.5, mismatch=0.0).prepare().simulate(until=80.0)
        >>> first, second = trajectory.get_values('oscillator_0.x1'), trajectory.get_values('oscillator_1.x1')
        >>> bool(abs(first[-1] - second[-1]) < 1e-4)
        True
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'chaos', 'ode', 'master stability function', 'networks')
    references = ('Rössler 1976', 'Pecora and Carroll 1998', 'Datseris and Parlitz 2022', 'Arenas et al. 2008')
    until = 100.0
    dt = 0.1

    def __init__(
        self,
        n_oscillators: int = 2,
        *,
        k: float = 0.04,
        mismatch: float = 0.02,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        seed: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_oscillators = n_oscillators
        self.k = k
        self.mismatch = mismatch
        self.a = a
        self.b = b
        self.c = c
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of chaotic Rössler oscillators on a ring.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        omegas = np.linspace(1 - self.mismatch, 1 + self.mismatch, self.n_oscillators)
        starts = np.random.default_rng(self.seed).uniform(-1.0, 1.0, (self.n_oscillators, 2))
        oscillators = [
            RosslerOscillatorElement(f'oscillator_{index}', x1=float(starts[index, 0]), x2=float(starts[index, 1]))
            for index in range(self.n_oscillators)
        ]
        flows: list[Interaction] = [
            RosslerFlowInteraction(
                f'flow_{index}',
                target=f'oscillator_{index}',
                omega=round(float(omegas[index]), 12),
                a=self.a,
                b=self.b,
                c=self.c,
            )
            for index in range(self.n_oscillators)
        ]
        links = [(index, index + 1) for index in range(self.n_oscillators - 1)]
        if self.n_oscillators > PAIR:
            links.append((self.n_oscillators - 1, 0))
        couplings = [
            DiffusiveCouplingInteraction(
                f'coupling_{source}_{target}',
                source=f'oscillator_{source}',
                target=f'oscillator_{target}',
                k=self.k,
            )
            for first, second in links
            for source, target in ((first, second), (second, first))
        ]
        self.system_ = System(self.label, elements=oscillators, interactions=[*flows, *couplings])


class AggregateAveragingSynchronizationSystem(BaseSystem):
    """Two chaotic maps coupled by aggregate averaging.

    The maps synchronize their chaos for strong coupling.

    Each map iterates on the aggregate of its own value and the other's, `x_i(t + 1) = f(a_i)` with `a_1 = (1 - kappa)
    x_1(t) + kappa x_2(t - T)` and `a_2 = (1 - kappa) x_2(t) + kappa x_1(t - T)`, for the delay `T`. The logistic map is
    `f(a) = r a (1 - a)`. Without delay, its synchronized orbit is stable for `kappa > (r - 1) / (2 r)`, which is `3 /
    8` for `r = 4`, and the maps agree in one step for `kappa = 1 / 2`. The Bernoulli shift `f(a) = r a mod 1` has the
    constant slope `r`, so a difference `d` of the maps follows `d(t + 1) = r ((1 - kappa) d(t) - kappa d(t - T))`
    exactly, and the maps synchronize when every root of `z^(T + 1) - r (1 - kappa) z^T + r kappa` lies in the unit
    disk, which for `T = 0` is `|r (1 - 2 kappa)| < 1`. The delay shrinks the window of synchronizing couplings: for `r
    = 1.5` it is `0.17 < kappa < 0.83` without delay, `0.17 < kappa < 0.66` for `T = 1`, and `0.17 < kappa < 0.44` for
    `T = 2`, and it closes for `T >= 5`. Before the start, the delayed values equal the initial values. An integer slope
    such as 2 loses one binary digit of the value per step in floating point and falls onto 0, so a slope that is not an
    integer, such as 1.5, keeps the chaos. The system has the values `first.x` and `second.x`, and their past values
    `first.x[-<j>]` and `second.x[-<j>]` for a delay.

    Args:
        x1:
            The initial value of the first map.

        x2:
            The initial value of the second map.

        r:
            The growth rate of the logistic map.

        kappa:
            The coupling strength.

        kind:
            The map, `'logistic'` or `'bernoulli'`.

        delay:
            The delay `T` of the coupling, in steps.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AggregateAveragingSynchronizationSystem
        >>> trajectory = AggregateAveragingSynchronizationSystem(kappa=0.5).prepare().simulate(until=2)
        >>> bool(trajectory.get_values('first.x')[-1] == trajectory.get_values('second.x')[-1])
        True
    """

    field = 'physics/synchronization'
    topics = ('synchronization', 'chaos', 'map', 'logistic map', 'bernoulli map', 'delay')
    references = ('Gros 2015',)
    until = 50
    dt = 1

    def __init__(
        self,
        x1: float = 0.3,
        x2: float = 0.6,
        *,
        r: float = 4.0,
        kappa: float = 0.4,
        kind: str = 'logistic',
        delay: int = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x1 = x1
        self.x2 = x2
        self.r = r
        self.kappa = kappa
        self.kind = kind
        self.delay = delay

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of two chaotic maps coupled by aggregate averaging.

        Before the start, the delayed values equal the initial values.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        lags = range(1, self.delay + 1) if isinstance(self.delay, int) else range(0)
        units = [
            LogisticUnitElement(label, x=value, values={f'x[-{lag}]': value for lag in lags} or None)
            for label, value in (('first', self.x1), ('second', self.x2))
        ]
        averaging = [
            AggregateAveragingInteraction(
                f'averaging_{target}',
                source=source,
                target=target,
                r=self.r,
                kappa=self.kappa,
                kind=self.kind,
                delay=self.delay,
            )
            for source, target in (('second', 'first'), ('first', 'second'))
        ]
        self.system_ = System(self.label, elements=units, interactions=averaging)


class RosslerDrivenLorenzSystem(BaseSystem):
    """A chaotic Rössler system that drives a Lorenz-63 system.

    The drive acts through the second variable.

    The drive follows `p x1' = -x2 - x3`, `p x2' = x1 + 0.2 x2`, `p x3' = 0.2 + x3 (x1 - 5.7)`, with the time scale `p =
    1 / 6` that brings it close to the Lorenz system, and the response follows `y1' = 10 (y2 - y1)`, `y2' = 28 y1 - y2 -
    y1 y3 + k x2`, `y3' = y1 y2 - 2.666 y3`. The two systems have nothing in common, yet for a coupling above `k ≈ 6.66`
    the response forgets its initial state and becomes a function of the state of the drive, generalized
    synchronization. With `auxiliary`, a second copy of the response from another initial state receives the same drive,
    and the two copies converge to each other exactly when generalized synchronization holds, the auxiliary system
    method. The system has the drive `drive.x1`, `drive.x2`, `drive.x3`, the response `response.x`, `response.y`,
    `response.z`, and with `auxiliary` the copy `auxiliary.x`, `auxiliary.y`, `auxiliary.z`.

    Args:
        k:
            The strength of the coupling.

        auxiliary:
            Whether a second copy of the response, `auxiliary`, receives the drive too.

        p:
            The time scale of the drive.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import RosslerDrivenLorenzSystem
        >>> trajectory = RosslerDrivenLorenzSystem(auxiliary=False).prepare().simulate(until=1.0, dt=0.5)
        >>> [round(float(value), 1) for value in trajectory.get_values('response.x')]
        [1.0, -0.1, -8.9]
    """

    field = 'physics/synchronization'
    topics = ('chaos', 'generalized synchronization', 'drive-response', 'ode')
    references = ('Rulkov et al. 1995', 'Abarbanel, Rulkov, and Sushchik 1996', 'Datseris and Parlitz 2022')
    until = 100.0
    dt = 0.01

    def __init__(
        self,
        k: float = 8.0,
        *,
        auxiliary: bool = True,
        p: float = 1 / 6,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.k = k
        self.auxiliary = auxiliary
        self.p = p

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a chaotic Rössler system that drives a Lorenz-63 system.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        responses = [Element('response', values={'x': 1.0, 'y': 1.0, 'z': 1.0})]
        if self.auxiliary:
            responses.append(Element('auxiliary', values={'x': -5.0, 'y': 3.0, 'z': 20.0}))
        interactions: list[Interaction] = [RosslerFlowInteraction('drive_flow', target='drive', p=self.p)]
        for response in responses:
            interactions.append(LorenzFlowInteraction(f'{response.label}_flow', target=response.label, beta=2.666))
            interactions.append(
                DriveInteraction(f'coupling_{response.label}', source='drive', target=response.label, k=self.k),
            )
        self.system_ = System(
            self.label,
            elements=[RosslerOscillatorElement('drive', x1=1.0, x2=0.0, x3=0.0), *responses],
            interactions=interactions,
        )
