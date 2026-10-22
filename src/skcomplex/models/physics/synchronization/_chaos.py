"""Chaotic synchronization models: coupled Rössler oscillators and logistic maps coupled by aggregate averaging."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import Domain, Expression, parameter
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Rate, Update
from ....core.runtime import register_model
from ....core.system import System

PAIR = 2


def _logistic(r: Expression, x: Expression) -> Expression:
    """Return the logistic map `r x (1 - x)`."""
    return r * x * (1 - x)


RosslerOscillator = Element.define(
    'RosslerOscillator',
    states={'x1': 1.0, 'x2': 0.0, 'x3': 0.0},
    parameters={'omega': 1.0, 'a': 0.2, 'b': 0.2, 'c': 5.7},
    laws=lambda s: [
        Rate(s.x1, -s.omega * s.x2 - s.x3),
        Rate(s.x2, s.omega * s.x1 + s.a * s.x2),
        Rate(s.x3, s.b + s.x3 * (s.x1 - s.c)),
    ],
)

DiffusiveCoupling = Interaction.define(
    'DiffusiveCoupling',
    roles=['first', 'second'],
    parameters={'k': parameter(default=0.1, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [
        Rate(s.first.x2, s.k * (s.second.x2 - s.first.x2)),
        Rate(s.second.x2, s.k * (s.first.x2 - s.second.x2)),
    ],
)

LogisticUnit = Element.define('LogisticUnit', states={'x': 0.3})

AggregateAveraging = Interaction.define(
    'AggregateAveraging',
    roles=['first', 'second'],
    parameters={
        'r': parameter(default=4.0, domain=Domain.NONNEGATIVE),
        'kappa': parameter(default=0.4, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Update(s.first.x, _logistic(s.r, (1 - s.kappa) * s.first.x + s.kappa * s.second.x)),
        Update(s.second.x, _logistic(s.r, (1 - s.kappa) * s.second.x + s.kappa * s.first.x)),
    ],
)


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
        RosslerOscillator(
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
        DiffusiveCoupling(f'coupling_{first}_{second}', first=f'oscillator_{first}', second=f'oscillator_{second}', k=k)
        for first, second in links
    ]
    return System(components=oscillators, interactions=couplings)


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
        components=[LogisticUnit('first', x=x1), LogisticUnit('second', x=x2)],
        interactions=[AggregateAveraging('averaging', first='first', second='second', r=r, kappa=kappa)],
    )
