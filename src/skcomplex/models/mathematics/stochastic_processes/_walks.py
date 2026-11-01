"""Stochastic process models: random walks, persistent and autoregressive walks, and diffusions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
from sage.all__sagemath_symbolics import sqrt

from ....core.elements import GroupElement
from ....core.interactions import JumpInteraction, MapInteraction, RateInteraction
from ....core.laws.utils import bernoulli, normal
from ....core.system import System
from ..._catalog import register_model
from ._ageing_walker import AgeingWalkerElement
from ._memory_walker import MemoryWalkerElement
from ._persistent_walker import PersistentWalkerElement
from ._random_walker import RandomWalkerElement
from ._weakening_particle import WeakeningParticleElement


@register_model(
    field='mathematics/stochastic_processes',
    tags=('random walk', 'jump process', 'diffusion'),
    references=('Pearson 1905', 'Argun et al. 2021', 'Charbonneau 2017', 'Jensen 2023'),
    defaults={'until': 100.0, 'dt': 1.0},
)
def random_walk(n_walkers: int = 100, *, rate: float = 1.0, bias: float = 0.5) -> System:
    """Return independent walkers on the integers that step at random times, a continuous-time random walk.

    Every walker steps up at rate `rate * bias` and down at rate `rate * (1 - bias)`, so its mean moves with velocity
    `rate (2 bias - 1)` and its variance grows as `4 rate bias (1 - bias) t`.

    Args:
        n_walkers:
            The number of walkers.

        rate:
            The rate of steps of a walker.

        bias:
            The probability that a step goes up.

    Returns:
        system:
            The model, with the positions `walkers.x`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import random_walk
        >>> random_walk(20).elements[0].check().prepare().size_
        20
    """
    walkers = GroupElement('walkers', element=RandomWalkerElement, size=n_walkers, values={'rate': rate, 'bias': bias})
    up = JumpInteraction('up', target='walkers', changes={'x': 1}, rate=lambda s: s.target.rate * s.target.bias)
    down = JumpInteraction(
        'down',
        target='walkers',
        changes={'x': -1},
        rate=lambda s: s.target.rate * (1 - s.target.bias),
    )
    return System(elements=[walkers], interactions=[up, down])


@register_model(
    field='mathematics/stochastic_processes',
    tags=('random walk', 'sde', 'heavy tails', 'power law'),
    references=('Jensen 2023',),
    defaults={'until': 50.0, 'dt': 0.05},
)
def weakening_ornstein_uhlenbeck(
    n_particles: int = 200,
    *,
    k: float = 1.0,
    ell: float = 1.0,
    D: float = 0.25,
) -> System:
    """Return Brownian particles held by a restoring force that weakens with distance.

    The particles follow `dx = -k x / (1 + x²/ℓ²) dt + √(2D) dW`. Near the origin the force is that of an
    Ornstein-Uhlenbeck process, and far from it the force decays as `1 / x`, so the stationary density
    `(1 + x²/ℓ²)^(-k ℓ² / 2D)` has power-law tails.

    Args:
        n_particles:
            The number of independent particles, all started at the origin.

        k:
            The stiffness of the force near the origin.

        ell:
            The distance where the force starts to weaken.

        D:
            The diffusion coefficient.

    Returns:
        system:
            The model, with the positions `particles.x`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import weakening_ornstein_uhlenbeck
        >>> weakening_ornstein_uhlenbeck(10).elements[0].check().prepare().size_
        10
    """
    particles = GroupElement(
        'particles',
        element=WeakeningParticleElement,
        size=n_particles,
        values={'k': k, 'ell': ell, 'D': D},
    )
    restoring = RateInteraction(
        'restoring',
        target='particles',
        state='x',
        rate=lambda s: -s.target.k * s.target.x / (1 + (s.target.x / s.target.ell) ** 2),
        noise=lambda s: sqrt(2 * s.target.D),
    )
    return System(elements=[particles], interactions=[restoring])


@register_model(
    field='mathematics/stochastic_processes',
    tags=('random walk', 'agents', 'history dependence', 'ageing'),
    references=('Hanel, Thurner, and Gell-Mann 2014', 'Thurner et al. 2018'),
    defaults={'until': 100},
)
def ageing_random_walk(n_walkers: int = 50, *, p: float = 0.5, nu: float = 1.0) -> System:
    """Return ageing random walks, whose up steps become rarer the more of them a walker has taken.

    After `k` up steps a walker steps up with probability `p / (1 + k)^nu`, so the walk is history dependent and,
    for `nu > 0`, its number of up steps grows as `n^(1 / (1 + nu))` after `n` steps.

    Args:
        n_walkers:
            The number of independent walkers, all started at 0.

        p:
            The probability of the first up step.

        nu:
            The exponent of the decay of the up probability.

    Returns:
        system:
            The model, with the group `walkers`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import ageing_random_walk
        >>> ageing_random_walk(5).elements[0].check().prepare().size_
        5
    """
    ageing = MapInteraction(
        'ageing',
        target='walkers',
        state='ups',
        function=lambda agent, model: float(model.rng.random() < agent.p / (1 + agent.ups) ** agent.nu),
        mode='increment',
        form='agent',
    )
    step = MapInteraction(
        'step',
        target='walkers',
        state='x',
        function=lambda agent, model: 2 * (agent.ups - agent.lags('ups')[-1]) - 1,
        mode='increment',
        form='agent',
    )
    return System(
        elements=[GroupElement('walkers', element=AgeingWalkerElement, size=n_walkers, values={'p': p, 'nu': nu})],
        interactions=[ageing, step],
    )


@register_model(
    field='mathematics/stochastic_processes',
    tags=('random walk', 'discrete map', 'persistence', 'diffusion'),
    references=('Jensen 2023',),
    defaults={'until': 200, 'dt': 1},
)
def persistent_random_walk(
    n_walkers: int = 2000,
    *,
    persistence: float = 0.8,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return walkers on the integers that keep the direction of their last step with a probability.

    At every step a walker moves by its direction `d = ±1`, and keeps the direction with probability `q` and reverses
    it otherwise. Successive steps are correlated with `rho = 2 q - 1`, so the walk diffuses faster than a simple
    random walk for `q > 1/2`, with a mean squared displacement that grows as `t (1 + rho) / (1 - rho)` at long times.

    Args:
        n_walkers:
            The number of independent walkers, all started at 0.

        persistence:
            The probability `q` of keeping the direction.

        random_state:
            The seed or generator of the initial directions.

    Returns:
        system:
            The model, with the directions `walkers.d` and the positions `walkers.x`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import persistent_random_walk
        >>> persistent_random_walk(10).elements[0].check().prepare().size_
        10
    """
    directions = np.random.default_rng(random_state).choice([-1.0, 1.0], size=n_walkers)
    walkers = GroupElement(
        'walkers',
        element=PersistentWalkerElement,
        size=n_walkers,
        values={'d': directions, 'persistence': persistence},
    )
    turn = MapInteraction(
        'turn',
        target='walkers',
        state='d',
        function=lambda s: s.target.d * (1 - 2 * bernoulli('turn', 1 - s.target.persistence)),
    )
    step = MapInteraction(
        'step',
        target='walkers',
        state='x',
        function=lambda s: s.target.d,
        mode='increment',
    )
    return System(elements=[walkers], interactions=[turn, step])


@register_model(
    field='mathematics/stochastic_processes',
    tags=('random walk', 'discrete map', 'memory', 'autoregressive'),
    references=('Bertin 2021',),
    defaults={'until': 200, 'dt': 1},
)
def random_walk_with_memory(n_walkers: int = 2000, *, alpha: float = 0.5, sigma: float = 1.0) -> System:
    """Return random walks whose velocity is an autoregressive process, `x ← x + v` and `v ← alpha v + xi`.

    The noise `xi` is normal with mean 0 and deviation `sigma`, and the walks start from `x = v = 0`. The pair
    `(x, v)` is Markovian while the displacements are correlated, so the mean squared displacement grows as
    `sigma² t / (1 - alpha)²` at long times instead of `sigma² t / (1 - alpha²)` for uncorrelated steps of the same
    variance.

    Args:
        n_walkers:
            The number of independent walkers.

        alpha:
            The memory `alpha` of the velocity, between 0 and 1.

        sigma:
            The deviation `sigma` of the noise.

    Returns:
        system:
            The model, with the positions `walkers.x` and the velocities `walkers.v`.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import random_walk_with_memory
        >>> random_walk_with_memory(10).elements[0].check().prepare().size_
        10
    """
    walkers = GroupElement(
        'walkers',
        element=MemoryWalkerElement,
        size=n_walkers,
        values={'alpha': alpha, 'sigma': sigma},
    )
    step = MapInteraction('step', target='walkers', state='x', function=lambda s: s.target.v, mode='increment')
    velocity = MapInteraction(
        'velocity',
        target='walkers',
        state='v',
        function=lambda s: s.target.alpha * s.target.v + normal('xi', 0, s.target.sigma),
    )
    return System(elements=[walkers], interactions=[step, velocity])
