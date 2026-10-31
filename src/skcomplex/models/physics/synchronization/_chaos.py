"""Chaotic synchronization models: coupled Rössler oscillators and logistic maps coupled by aggregate averaging."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import MapInteraction, RateInteraction
from ....core.laws import BaseLaw, RateLaw
from ....core.system import System
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model

PAIR = 2


class RosslerOscillatorElement(Element):
    """Rössler oscillator `x1' = -omega x2 - x3`, `x2' = omega x1 + a x2`, `x3' = b + x3 (x1 - c)`.

    It is chaotic for `a = b = 0.2` and `c = 5.7`.

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

    Attributes:
        default_visual_:
            The visual of the oscillator when it sets none.

    Examples:
        >>> from skcomplex.models.physics.synchronization import RosslerOscillatorElement
        >>> RosslerOscillatorElement('oscillator').c
        5.7
    """

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
    ) -> None:
        super().__init__(label, visual=visual)
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.omega = omega
        self.a = a
        self.b = b
        self.c = c

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
            RateLaw(s.x1, -s.omega * s.x2 - s.x3),
            RateLaw(s.x2, s.omega * s.x1 + s.a * s.x2),
            RateLaw(s.x3, s.b + s.x3 * (s.x1 - s.c)),
        ]


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
        >>> from skcomplex.models.physics.synchronization import DiffusiveCouplingInteraction
        >>> coupling = DiffusiveCouplingInteraction('coupling', source='a', target='b').check().prepare()
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
        >>> from skcomplex.models.physics.synchronization import LogisticUnitElement
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


class AggregateAveragingInteraction(MapInteraction):
    """Logistic map of an aggregate, `x_t(t + 1) = r a (1 - a)` with `a = (1 - kappa) x_t + kappa x_s`.

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

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.models.physics.synchronization import AggregateAveragingInteraction
        >>> averaging = AggregateAveragingInteraction('averaging', source='a', target='b').check().prepare()
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
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._average)
        self.source = source
        self.r = r
        self.kappa = kappa

    def _average(self, namespace: SimpleNamespace) -> Expression:
        """Return the logistic map `r a (1 - a)` of the aggregate `a = (1 - kappa) x_t + kappa x_s`."""
        aggregate = (1 - namespace.kappa) * namespace.target.x + namespace.kappa * namespace.source.x
        return namespace.r * aggregate * (1 - aggregate)


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'chaos', 'ode', 'master stability function', 'networks'),
    references=('Rössler 1976', 'Pecora and Carroll 1998', 'Datseris and Parlitz 2022', 'Arenas et al. 2008'),
    defaults={'until': 100.0, 'dt': 0.1},
)
def coupled_rossler(
    n_oscillators: int = 2,
    *,
    k: float = 0.04,
    mismatch: float = 0.02,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    seed: int = 0,
) -> System:
    """Return chaotic Rössler oscillators on a ring, diffusively coupled through their second variable.

    Oscillator `i` follows `x1' = -ω_i x2 - x3`, `x2' = ω_i x1 + a x2 + k sum_j (x2_j - x2_i)`, and
    `x3' = b + x3 (x1 - c)`, summed over its neighbours on the ring. Two oscillators with frequencies `1 ± mismatch`
    pass from no synchronization to chaotic phase synchronization as `k` grows. Identical oscillators, with no mismatch,
    synchronize completely when every nonzero eigenvalue of the ring's Laplacian times `k` falls where the master
    stability function is negative.

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

    Returns:
        system:
            The model, with the states `oscillator_<i>.x1`, `oscillator_<i>.x2`, and `oscillator_<i>.x3`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import coupled_rossler
        >>> [element.omega for element in coupled_rossler().elements]
        [0.98, 1.02]
    """
    omegas = np.linspace(1 - mismatch, 1 + mismatch, n_oscillators)
    starts = np.random.default_rng(seed).uniform(-1.0, 1.0, (n_oscillators, 2))
    oscillators = [
        RosslerOscillatorElement(
            f'oscillator_{index}',
            x1=float(starts[index, 0]),
            x2=float(starts[index, 1]),
            x3=0.0,
            omega=round(float(omegas[index]), 12),
            a=a,
            b=b,
            c=c,
        )
        for index in range(n_oscillators)
    ]
    links = [(index, index + 1) for index in range(n_oscillators - 1)]
    if n_oscillators > PAIR:
        links.append((n_oscillators - 1, 0))
    couplings = [
        DiffusiveCouplingInteraction(
            f'coupling_{source}_{target}',
            source=f'oscillator_{source}',
            target=f'oscillator_{target}',
            k=k,
        )
        for first, second in links
        for source, target in ((first, second), (second, first))
    ]
    return System(elements=oscillators, interactions=couplings)


@register_model(
    field='physics/synchronization',
    tags=('synchronization', 'chaos', 'map', 'logistic map'),
    references=('Gros 2015',),
    defaults={'until': 50, 'dt': 1},
)
def aggregate_averaging_synchronization(
    x1: float = 0.3,
    x2: float = 0.6,
    *,
    r: float = 4.0,
    kappa: float = 0.4,
) -> System:
    """Return two logistic maps coupled by aggregate averaging, which synchronize their chaos for strong coupling.

    Each map iterates on the aggregate of its own value and the other's, `x_i(t + 1) = r a_i (1 - a_i)` with
    `a_1 = (1 - kappa) x_1 + kappa x_2` and `a_2 = (1 - kappa) x_2 + kappa x_1`. The synchronized orbit is stable for
    `kappa > (r - 1) / (2 r)`, which is `3 / 8` for `r = 4`, and the maps agree in one step for `kappa = 1 / 2`.

    Args:
        x1:
            The initial value of the first map.

        x2:
            The initial value of the second map.

        r:
            The growth rate of the logistic map.

        kappa:
            The coupling strength.

    Returns:
        system:
            The model, with the values `first.x` and `second.x`.

    Examples:
        >>> from skcomplex.models.physics.synchronization import aggregate_averaging_synchronization
        >>> aggregate_averaging_synchronization(kappa=0.1).interactions[0].kappa
        0.1
    """
    return System(
        elements=[LogisticUnitElement('first', x=x1), LogisticUnitElement('second', x=x2)],
        interactions=[
            AggregateAveragingInteraction(f'averaging_{target}', source=source, target=target, r=r, kappa=kappa)
            for source, target in (('second', 'first'), ('first', 'second'))
        ],
    )
