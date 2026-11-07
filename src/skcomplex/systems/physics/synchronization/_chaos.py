"""Chaotic synchronization systems: coupled Rössler oscillators and logistic maps coupled by aggregate averaging."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import Element
from ....core.components.interactions import RateInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ..._catalog import register_system
from ...mathematics.common.flows import LorenzElement
from ._aggregate_averaging import AggregateAveragingInteraction
from ._delayed_unit import DelayedUnitElement
from ._diffusive_coupling import DiffusiveCouplingInteraction
from ._logistic_unit import LogisticUnitElement
from ._rossler_oscillator import RosslerOscillatorElement

PAIR = 2


@register_system(
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
            The system, with the states `oscillator_<i>.x1`, `oscillator_<i>.x2`, and `oscillator_<i>.x3`.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import coupled_rossler
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


@register_system(
    field='physics/synchronization',
    tags=('synchronization', 'chaos', 'map', 'logistic map', 'bernoulli map', 'delay'),
    references=('Gros 2015',),
    defaults={'until': 50, 'dt': 1},
)
def aggregate_averaging_synchronization(
    x1: float = 0.3,
    x2: float = 0.6,
    *,
    r: float = 4.0,
    kappa: float = 0.4,
    kind: str = 'logistic',
    delay: int = 0,
) -> System:
    """Return two chaotic maps coupled by aggregate averaging, which synchronize their chaos for strong coupling.

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
    integer, such as 1.5, keeps the chaos.

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

    Returns:
        system:
            The system, with the values `first.x` and `second.x`, and their lags `first.x_lag<j>` and
            `second.x_lag<j>` for a delay.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import aggregate_averaging_synchronization
        >>> aggregate_averaging_synchronization(kappa=0.1).interactions[0].kappa
        0.1
    """
    if delay:
        units: list[Element] = [
            DelayedUnitElement('first', value=x1, delay=delay),
            DelayedUnitElement('second', value=x2, delay=delay),
        ]
    else:
        units = [LogisticUnitElement('first', x=x1), LogisticUnitElement('second', x=x2)]
    averaging = [
        AggregateAveragingInteraction(
            f'averaging_{target}',
            source=source,
            target=target,
            r=r,
            kappa=kappa,
            kind=kind,
            delay=delay,
        )
        for source, target in (('second', 'first'), ('first', 'second'))
    ]
    return System(elements=units, interactions=averaging, time=DiscreteTime())


@register_system(
    field='physics/synchronization',
    tags=('chaos', 'generalized synchronization', 'drive-response', 'ode'),
    references=('Rulkov et al. 1995', 'Abarbanel, Rulkov, and Sushchik 1996', 'Datseris and Parlitz 2022'),
    defaults={'until': 100.0, 'dt': 0.01},
)
def rossler_driven_lorenz(k: float = 8.0, *, auxiliary: bool = True, p: float = 1 / 6) -> System:
    """Return a chaotic Rössler system that drives a Lorenz-63 system through its second variable.

    The drive follows `p x1' = -x2 - x3`, `p x2' = x1 + 0.2 x2`, `p x3' = 0.2 + x3 (x1 - 5.7)`, with the time scale
    `p = 1 / 6` that brings it close to the Lorenz system, and the response follows `y1' = 10 (y2 - y1)`,
    `y2' = 28 y1 - y2 - y1 y3 + k x2`, `y3' = y1 y2 - 2.666 y3`. The two systems have nothing in common, yet for a
    coupling above `k ≈ 6.66` the response forgets its initial state and becomes a function of the state of the drive,
    generalized synchronization. With `auxiliary`, a second copy of the response from another initial state receives
    the same drive, and the two copies converge to each other exactly when generalized synchronization holds, the
    auxiliary system method.

    Args:
        k:
            The strength of the coupling.

        auxiliary:
            Whether a second copy of the response, `auxiliary`, receives the drive too.

        p:
            The time scale of the drive.

    Returns:
        system:
            The system, with the drive `drive.x1`, `drive.x2`, `drive.x3`, the response `response.x`, `response.y`,
            `response.z`, and with `auxiliary` the copy `auxiliary.x`, `auxiliary.y`, `auxiliary.z`.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import rossler_driven_lorenz
        >>> [element.label for element in rossler_driven_lorenz(auxiliary=False).elements]
        ['drive', 'response']
    """
    drive = RosslerOscillatorElement('drive', x1=1.0, x2=0.0, x3=0.0, p=p)
    responses = [LorenzElement('response', x=1.0, y=1.0, z=1.0, beta=2.666)]
    if auxiliary:
        responses.append(LorenzElement('auxiliary', x=-5.0, y=3.0, z=20.0, beta=2.666))
    couplings = [
        RateInteraction(
            f'coupling_{response.label}',
            target=response.label,
            state='y',
            sources=['drive'],
            rate=lambda s: k * s.sources[0].x2,
        )
        for response in responses
    ]
    return System(elements=[drive, *responses], interactions=couplings)
