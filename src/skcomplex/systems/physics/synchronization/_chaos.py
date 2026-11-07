"""Chaotic synchronization systems: coupled Rössler oscillators and logistic maps coupled by aggregate averaging."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import DeclaredElement, Element
from ....core.components.interactions import MapInteraction, RateInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import lag, mod
from ....core.components.spaces import DiscreteTime
from ....core.components.system import BaseSystem, System
from ....core.errors import ValidationError
from ....core.specs import DeclaredVariableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec


class LorenzElement(Element):
    """Lorenz flow `x' = sigma (y - x)`, `y' = x (rho - z) - y`, `z' = x y - beta z`.

    It is the response that a Rössler oscillator drives into generalized synchronization, and its auxiliary copy.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial convection intensity.

        y:
            The initial temperature difference between rising and falling currents.

        z:
            The initial deviation of the vertical temperature profile.

        sigma:
            The Prandtl number.

        rho:
            The Rayleigh number relative to its critical value.

        beta:
            The geometric factor.

    Examples:
        >>> from skcomplex.core.components.system import System
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.physics.synchronization._chaos import LorenzElement
        >>> system = System(elements=[LorenzElement('lorenz', x=0.0, y=0.0, z=2.0, beta=0.5)])
        >>> trajectory = Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_
        >>> [round(float(value), 4) for value in trajectory.values('lorenz.z')]
        [2.0, 1.5576, 1.2131]
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 1.0,
        z: Annotated[float, StateSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec()] = 10.0,
        rho: Annotated[float, ParameterSpec()] = 28.0,
        beta: Annotated[float, ParameterSpec()] = 8 / 3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the three modes.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of `x`, `y`, and `z`.
        """
        return [
            RateLaw(s.x, s.sigma * (s.y - s.x)),
            RateLaw(s.y, s.x * (s.rho - s.z) - s.y),
            RateLaw(s.z, s.x * s.y - s.beta * s.z),
        ]


class DelayedUnitElement(DeclaredElement):
    """Value `x` of a map that keeps its last `delay` values, which a delayed coupling reads.

    The state `x` has `delay` lags, `x_lag1` to `x_lag<delay>`, and before the start every lag holds the initial value.

    Args:
        label:
            The label of the unit.

        visual:
            How the unit is drawn, or `None` for `default_visual_`.

        value:
            The initial value of `x`, also its value before the start.

        delay:
            The number of past values the unit keeps, at least 1.

        variables:
            The declared variables that replace the derived ones, or `None` for none.

    Attributes:
        default_visual_:
            The visual of the unit when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import DelayedUnitElement
        >>> DelayedUnitElement('unit', delay=2).get_value('x')
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        value: float = 0.3,
        delay: int = 1,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(label, visual=visual, variables=variables)
        self.value = value
        self.delay = delay

    def _declare_variables(self) -> Mapping[str, DeclaredVariableSpec]:
        """Return the value `x` with its `delay` lags, all at the initial value, and the replacing variables."""
        lagged = StateSpec(lags=self.delay, initial=(self.value,) * self.delay)
        return {'x': DeclaredVariableSpec(self.value, lagged), **(self.variables or {})}


class LogisticUnitElement(Element):
    """Value `x` in `[0, 1]` of a logistic map, which maps of the coupling iterate.

    Args:
        label:
            The label of the unit.

        visual:
            How the unit is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

    Attributes:
        default_visual_:
            The visual of the unit when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import LogisticUnitElement
        >>> LogisticUnitElement('unit').x
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class RosslerOscillatorElement(Element):
    """Rössler oscillator `x1' = -omega x2 - x3`, `x2' = omega x1 + a x2`, `x3' = b + x3 (x1 - c)`.

    It is chaotic for `a = b = 0.2` and `c = 5.7`. The time scale `p` divides every rate, `p x' = f(x)`, so `p < 1`
    speeds the oscillator up.

    Args:
        label:
            The label of the oscillator.

        visual:
            How the oscillator is drawn, or `None` for `default_visual_`.

        x1:
            The initial first variable.

        x2:
            The initial second variable.

        x3:
            The initial third variable.

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

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import RosslerOscillatorElement
        >>> RosslerOscillatorElement('oscillator').c
        5.7
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x1: Annotated[float, StateSpec()] = 1.0,
        x2: Annotated[float, StateSpec()] = 0.0,
        x3: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
        a: Annotated[float, ParameterSpec()] = 0.2,
        b: Annotated[float, ParameterSpec()] = 0.2,
        c: Annotated[float, ParameterSpec()] = 5.7,
        p: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.omega = omega
        self.a = a
        self.b = b
        self.c = c
        self.p = p

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the Rössler flow.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of the three variables.
        """
        s = namespace
        return [
            RateLaw(s.x1, (-s.omega * s.x2 - s.x3) / s.p),
            RateLaw(s.x2, (s.omega * s.x1 + s.a * s.x2) / s.p),
            RateLaw(s.x3, (s.b + s.x3 * (s.x1 - s.c)) / s.p),
        ]


MAPS = ('logistic', 'bernoulli')


class AggregateAveragingInteraction(MapInteraction):
    """Logistic or Bernoulli map of an aggregate of the value of a target and the value of a source, possibly delayed.

    The map sets `x_t(t + 1) = f(a)` with the aggregate `a = (1 - kappa) x_t(t) + kappa x_s(t - delay)`, and with the
    logistic map `f(a) = r a (1 - a)` or the Bernoulli shift `f(a) = r a mod 1`. A delayed source keeps `delay` lags.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

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
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.synchronization import AggregateAveragingInteraction
        >>> averaging = AggregateAveragingInteraction('averaging', source='a', target='b').prepare()
        >>> source, target = SimpleNamespace(x=var('u')), SimpleNamespace(x=var('v'))
        >>> namespace = SimpleNamespace(source=source, target=target, r=4, kappa=var('kappa'))
        >>> [law.value.factor() for law in averaging.run(namespace).laws_]
        [-4*(kappa*u - kappa*v + v)*(kappa*u - kappa*v + v - 1)]
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        kind: str = 'logistic',
        delay: int = 0,
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._average)
        self.source = source
        self.r = r
        self.kappa = kappa
        self.kind = kind
        self.delay = delay

    def _check(self) -> None:
        """Raise an error for an unknown map or a delay that is not a nonnegative integer.

        Raises:
            ValidationError:
                The map is unknown, or the delay is not a nonnegative integer.
        """
        super()._check()
        if self.kind not in MAPS:
            error_msg = f'Unknown map {self.kind!r}. Use one of {", ".join(MAPS)}.'
            raise ValidationError(error_msg)
        if isinstance(self.delay, bool) or not isinstance(self.delay, int) or self.delay < 0:
            error_msg = f'The delay of {self.label!r} is {self.delay!r}. Give a nonnegative integer.'
            raise ValidationError(error_msg)

    def _average(self, namespace: SimpleNamespace) -> Expression:
        """Return the map of the aggregate `a = (1 - kappa) x_t + kappa x_s(t - delay)`."""
        source = lag(namespace.source.x, self.delay)
        aggregate = (1 - namespace.kappa) * namespace.target.x + namespace.kappa * source
        if self.kind == 'bernoulli':
            return mod(namespace.r * aggregate, 1)
        return namespace.r * aggregate * (1 - aggregate)


class DiffusiveCouplingInteraction(RateInteraction):
    """Diffusive coupling of a state of one element to the same state of another, `x_t' += k (x_s - x_t)`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        source:
            The path of the element whose state the coupling reads.

        target:
            The path of the element whose state the coupling changes.

        state:
            The name of the coupled state.

        k:
            The coupling strength.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.synchronization import DiffusiveCouplingInteraction
        >>> coupling = DiffusiveCouplingInteraction('coupling', source='a', target='b').prepare()
        >>> source, target = SimpleNamespace(x2=var('u')), SimpleNamespace(x2=var('v'))
        >>> [law.value for law in coupling.run(SimpleNamespace(source=source, target=target, k=var('k'))).laws_]
        [k*(u - v)]
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        state: str = 'x2',
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.source = source
        self.k = k

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return `k (x_s - x_t)`.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The rate of the state of the target.
        """
        return namespace.k * (getattr(namespace.source, self.state) - getattr(namespace.target, self.state))


PAIR = 2


class CoupledRosslerSystem(BaseSystem):
    """Chaotic Rössler oscillators on a ring, diffusively coupled through their second variable.

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

    Examples:
        >>> from skcomplex.systems.physics.synchronization import CoupledRosslerSystem
        >>> [element.omega for element in CoupledRosslerSystem().prepare().system_.elements]
        [0.98, 1.02]
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
    ) -> None:
        super().__init__(label)
        self.n_oscillators = n_oscillators
        self.k = k
        self.mismatch = mismatch
        self.a = a
        self.b = b
        self.c = c
        self.seed = seed

    def _build_system(self) -> System:
        """Return chaotic Rössler oscillators on a ring, diffusively coupled through their second variable."""
        omegas = np.linspace(1 - self.mismatch, 1 + self.mismatch, self.n_oscillators)
        starts = np.random.default_rng(self.seed).uniform(-1.0, 1.0, (self.n_oscillators, 2))
        oscillators = [
            RosslerOscillatorElement(
                f'oscillator_{index}',
                x1=float(starts[index, 0]),
                x2=float(starts[index, 1]),
                x3=0.0,
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
        return System(self.label, elements=oscillators, interactions=couplings)


class AggregateAveragingSynchronizationSystem(BaseSystem):
    """Two chaotic maps coupled by aggregate averaging, which synchronize their chaos for strong coupling.

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
    integer, such as 1.5, keeps the chaos. The system has the values `first.x` and `second.x`, and their lags
    `first.x_lag<j>` and `second.x_lag<j>` for a delay.

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

    Examples:
        >>> from skcomplex.systems.physics.synchronization import AggregateAveragingSynchronizationSystem
        >>> AggregateAveragingSynchronizationSystem(kappa=0.1).prepare().system_.interactions[0].kappa
        0.1
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
    ) -> None:
        super().__init__(label)
        self.x1 = x1
        self.x2 = x2
        self.r = r
        self.kappa = kappa
        self.kind = kind
        self.delay = delay

    def _build_system(self) -> System:
        """Return two chaotic maps coupled by aggregate averaging, which synchronize their chaos for strong coupling."""
        if self.delay:
            units: list[Element] = [
                DelayedUnitElement('first', value=self.x1, delay=self.delay),
                DelayedUnitElement('second', value=self.x2, delay=self.delay),
            ]
        else:
            units = [LogisticUnitElement('first', x=self.x1), LogisticUnitElement('second', x=self.x2)]
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
        return System(self.label, elements=units, interactions=averaging, time=DiscreteTime())


class RosslerDrivenLorenzSystem(BaseSystem):
    """A chaotic Rössler system that drives a Lorenz-63 system through its second variable.

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

    Examples:
        >>> from skcomplex.systems.physics.synchronization import RosslerDrivenLorenzSystem
        >>> [element.label for element in RosslerDrivenLorenzSystem(auxiliary=False).prepare().system_.elements]
        ['drive', 'response']
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
    ) -> None:
        super().__init__(label)
        self.k = k
        self.auxiliary = auxiliary
        self.p = p

    def _build_system(self) -> System:
        """Return a chaotic Rössler system that drives a Lorenz-63 system through its second variable."""
        drive = RosslerOscillatorElement('drive', x1=1.0, x2=0.0, x3=0.0, p=self.p)
        responses = [LorenzElement('response', x=1.0, y=1.0, z=1.0, beta=2.666)]
        if self.auxiliary:
            responses.append(LorenzElement('auxiliary', x=-5.0, y=3.0, z=20.0, beta=2.666))
        couplings = [
            RateInteraction(
                f'coupling_{response.label}',
                target=response.label,
                state='y',
                sources=['drive'],
                rate=lambda s: self.k * s.sources[0].x2,
            )
            for response in responses
        ]
        return System(self.label, elements=[drive, *responses], interactions=couplings)
