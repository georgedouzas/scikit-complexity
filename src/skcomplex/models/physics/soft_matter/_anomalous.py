"""Anomalous diffusion: random walks with power-law waiting times and flights, heterogeneity, and memory."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import sqrt

from ....core.base import Domain, Expression, VisualSpec, parameter, state, vector
from ....core.components import Element
from ....core.laws import FractionalNoise, Jump, Law, Noise, Rate, Symbols, Update, draw, normal
from ....core.runtime import register_model
from ....core.solvers import RK4, EulerMaruyama, FractionalEulerMaruyama
from ....core.space import ContinuousSpace
from ....core.system import System

POINTS = VisualSpec('points', {'position': 'x'})


def _direction(name: str, size: int) -> list[Expression]:
    """Return a unit vector of uniformly random direction drawn from standard normal components."""
    components = [normal(f'{name}{index}') for index in range(size)]
    length = sqrt(sum(component**2 for component in components))
    return [component / length for component in components]


class ContinuousTimeWalker(Element):
    """Walker that waits a power-law time `tau0 r^(-1 / alpha)` for `r` uniform, then jumps by a Gaussian step.

    Args:
        label:
            The label of the walker or the group.

        x:
            The initial position.

        alpha:
            The tail exponent of the waiting times, below 1 for waiting times without a mean.

        tau0:
            The shortest waiting time.

        sigma:
            The standard deviation of the jumps along every axis.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import ContinuousTimeWalker
        >>> ContinuousTimeWalker('walker', alpha=0.7).alpha
        0.7
    """

    visual: ClassVar[VisualSpec] = POINTS
    x: npt.ArrayLike = state(default=0.0, vector=True)
    alpha: float = parameter(default=0.5, domain=Domain.POSITIVE)
    tau0: float = parameter(default=1.0, domain=Domain.POSITIVE)
    sigma: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the jumps after power-law waiting times.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        jumps = [item + normal(f'jump{index}', 0, s.sigma) for index, item in enumerate(s.x)]
        return [
            Jump(
                waiting=draw('wait', 'pareto', b=s.alpha, scale=s.tau0),
                reset=dict(zip(s.x, jumps, strict=True)),
                label='jumps',
            ),
        ]


class LevyWalker(Element):
    """Walker that flies at constant speed in a random direction for a power-law flight time, then turns.

    Args:
        label:
            The label of the walker or the group.

        x:
            The initial position.

        v:
            The initial velocity.

        sigma:
            The tail exponent of the flight times, between 1 and 2 for superdiffusion.

        tau0:
            The shortest flight time.

        speed:
            The speed of every flight.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import LevyWalker
        >>> LevyWalker('walker', speed=2.0).speed
        2.0
    """

    visual: ClassVar[VisualSpec] = POINTS
    x: npt.ArrayLike = state(default=0.0, vector=True)
    v: npt.ArrayLike = state(default=0.0, vector=True)
    sigma: float = parameter(default=1.5, domain=Domain.POSITIVE)
    tau0: float = parameter(default=1.0, domain=Domain.POSITIVE)
    speed: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the flight at constant velocity and the turns after power-law flight times.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        turn = [s.speed * item for item in _direction('direction', len(list(s.v)))]
        return [
            Rate(s.x, s.v),
            Jump(
                waiting=draw('flight', 'pareto', b=s.sigma, scale=s.tau0),
                reset=dict(zip(s.v, turn, strict=True)),
                label='turns',
            ),
        ]


class TransientWalker(Element):
    """Brownian walker whose diffusion coefficient `D` is redrawn after waiting times of mean `D^(-gamma)`.

    The coefficient takes values of density `sigma D^(sigma - 1)` on `(0, 1)`, the beta distribution `B(sigma, 1)`, so
    slow walkers wait long before they change.

    Args:
        label:
            The label of the walker or the group.

        x:
            The initial position.

        D:
            The initial diffusion coefficient.

        sigma:
            The exponent of the density of diffusion coefficients near zero.

        gamma:
            The exponent of the mean waiting time `D^(-gamma)`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import TransientWalker
        >>> TransientWalker('walker', gamma=2.0).gamma
        2.0
    """

    visual: ClassVar[VisualSpec] = VisualSpec('points', {'position': 'x', 'color_by': 'D', 'color_range': (0.0, 1.0)})
    x: npt.ArrayLike = state(default=0.0, vector=True)
    D: float = state(default=0.5, domain=Domain.POSITIVE)
    sigma: float = parameter(default=0.5, domain=Domain.POSITIVE)
    gamma: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the diffusion and the renewals of the diffusion coefficient.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        waiting = draw('wait', 'expon', scale=s.D ** (-s.gamma))
        return [
            Noise(s.x, sqrt(2 * s.D)),
            Jump(waiting=waiting, reset={s.D: draw('coefficient', 'beta', a=s.sigma, b=1)}, label='changes'),
        ]


class FractionalWalker(Element):
    """Walker driven by fractional Brownian motion, `dx = sqrt(2 D) dB_H`, whose increments are correlated in time.

    Args:
        label:
            The label of the walker or the group.

        x:
            The initial position.

        D:
            The diffusion coefficient.

        H:
            The Hurst index, below 1/2 for anticorrelated and above 1/2 for correlated increments.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import FractionalWalker
        >>> FractionalWalker('walker', H=0.3).H
        0.3
    """

    visual: ClassVar[VisualSpec] = POINTS
    x: npt.ArrayLike = state(default=0.0, vector=True)
    D: float = parameter(default=1.0, domain=Domain.POSITIVE)
    H: float = parameter(default=0.3, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the fractional noise.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [FractionalNoise(s.x, sqrt(2 * s.D), hurst=s.H)]


class LevyFlier(Element):
    """Walker that jumps at every step by a length of power-law tail `l^(-1 - mu)` in a random direction.

    Args:
        label:
            The label of the walker or the group.

        x:
            The initial position.

        mu:
            The tail exponent of the jump lengths, below 2 for jumps of infinite variance.

        l0:
            The shortest jump length.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import LevyFlier
        >>> LevyFlier('walker', mu=1.2).mu
        1.2
    """

    visual: ClassVar[VisualSpec] = POINTS
    x: npt.ArrayLike = state(default=0.0, vector=True)
    mu: float = parameter(default=1.2, domain=Domain.POSITIVE)
    l0: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the jumps of power-law lengths.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        length = draw('length', 'pareto', b=s.mu, scale=s.l0)
        jumps = [item + length * step for item, step in zip(s.x, _direction('direction', len(list(s.x))), strict=True)]
        return [Update(s.x, vector(jumps))]


@register_model(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'renewal process', 'subdiffusion', 'random walk'),
    references=('Argun et al. 2021', 'Metzler and Klafter 2000', 'Montroll and Weiss 1965', 'Bertin 2021'),
    defaults={'until': 1000.0, 'dt': 5.0},
)
def continuous_time_random_walk(
    n_walkers: int = 500,
    *,
    alpha: float = 0.5,
    tau0: float = 1.0,
    sigma: float = 1.0,
    n_dim: int = 2,
) -> System:
    """Return the continuous-time random walk, Gaussian jumps after waiting times with the tail `tau^(-1 - alpha)`.

    For `alpha < 1` the waiting times have no mean, so the number of jumps grows as `t^alpha` and the ensemble mean
    squared displacement grows as `t^alpha`, slower than Brownian motion. The walkers renew on their own clocks.

    Args:
        n_walkers:
            The number of independent walkers.

        alpha:
            The tail exponent of the waiting times.

        tau0:
            The shortest waiting time.

        sigma:
            The standard deviation of the jumps along every axis.

        n_dim:
            The number of dimensions.

    Returns:
        system:
            The model, with the positions `walkers.x` and the jump count `jumps.renewals`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import continuous_time_random_walk
        >>> continuous_time_random_walk(10, alpha=0.7).elements[0].values['alpha']
        0.7
    """
    walkers = ContinuousTimeWalker.group(
        'walkers',
        n_walkers,
        x=np.zeros((n_walkers, n_dim)),
        alpha=alpha,
        tau0=tau0,
        sigma=sigma,
    )
    return System(components=[walkers], space=ContinuousSpace(n_dim=n_dim))


@register_model(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'renewal process', 'superdiffusion', 'random walk'),
    references=('Argun et al. 2021', 'Zaburdaev, Denisov, and Klafter 2015', 'Shlesinger, West, and Klafter 1987'),
    defaults={'until': 500.0, 'dt': 2.5, 'solver': RK4(dt=0.5)},
)
def levy_walk(
    n_walkers: int = 500,
    *,
    sigma: float = 1.5,
    tau0: float = 1.0,
    speed: float = 1.0,
    n_dim: int = 2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Lévy walk, flights at constant speed in random directions for times with the tail `tau^(-1 - sigma)`.

    For `1 < sigma < 2` the flight times have a mean but no variance, so long flights dominate and the ensemble mean
    squared displacement grows as `t^(3 - sigma)`, faster than Brownian motion and slower than ballistic motion.

    Args:
        n_walkers:
            The number of independent walkers.

        sigma:
            The tail exponent of the flight times.

        tau0:
            The shortest flight time.

        speed:
            The speed of every flight.

        n_dim:
            The number of dimensions.

        random_state:
            The seed or generator of the directions of the first flights.

    Returns:
        system:
            The model, with the positions `walkers.x`, the velocities `walkers.v`, and the turn count `turns.renewals`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import levy_walk
        >>> levy_walk(10, sigma=1.2).elements[0].values['sigma']
        1.2
    """
    directions = np.random.default_rng(random_state).standard_normal((n_walkers, n_dim))
    velocities = speed * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    walkers = LevyWalker.group(
        'walkers',
        n_walkers,
        x=np.zeros((n_walkers, n_dim)),
        v=velocities,
        sigma=sigma,
        tau0=tau0,
        speed=speed,
    )
    return System(components=[walkers], space=ContinuousSpace(n_dim=n_dim))


@register_model(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'renewal process', 'heterogeneity', 'sde'),
    references=('Argun et al. 2021', 'Massignan et al. 2014'),
    defaults={'until': 500.0, 'dt': 2.5, 'solver': EulerMaruyama(dt=0.5)},
)
def annealed_transient_time_motion(
    n_walkers: int = 500,
    *,
    sigma: float = 0.5,
    gamma: float = 1.0,
    n_dim: int = 2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return annealed transient time motion, Brownian walkers whose diffusion coefficient changes at random times.

    Every walker diffuses with a coefficient `D` of density `sigma D^(sigma - 1)` on `(0, 1)` and keeps it for an
    exponential time of mean `D^(-gamma)`, so slow walkers stay slow for long. For `sigma < gamma` the ensemble mean
    squared displacement grows as `t^(sigma / gamma)`.

    Args:
        n_walkers:
            The number of independent walkers.

        sigma:
            The exponent of the density of diffusion coefficients near zero.

        gamma:
            The exponent of the mean waiting time `D^(-gamma)`.

        n_dim:
            The number of dimensions.

        random_state:
            The seed or generator of the first diffusion coefficients.

    Returns:
        system:
            The model, with the positions `walkers.x`, the coefficients `walkers.D`, and the change count
            `changes.renewals`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import annealed_transient_time_motion
        >>> annealed_transient_time_motion(10, gamma=2.0).elements[0].values['gamma']
        2.0
    """
    coefficients = np.random.default_rng(random_state).beta(sigma, 1.0, n_walkers)
    walkers = TransientWalker.group(
        'walkers',
        n_walkers,
        x=np.zeros((n_walkers, n_dim)),
        D=coefficients,
        sigma=sigma,
        gamma=gamma,
    )
    return System(components=[walkers], space=ContinuousSpace(n_dim=n_dim))


@register_model(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'fractional noise', 'long memory', 'sde'),
    references=('Argun et al. 2021', 'Mandelbrot and Van Ness 1968'),
    defaults={'until': 10.0, 'dt': 0.05, 'solver': FractionalEulerMaruyama(dt=0.01)},
)
def fractional_brownian_motion(
    n_walkers: int = 500,
    *,
    H: float = 0.3,
    D: float = 1.0,
    n_dim: int = 2,
) -> System:
    """Return fractional Brownian motion, `dx = sqrt(2 D) dB_H`, with `<x^2> = 2 D t^(2 H)` along every axis.

    The increments are correlated over the whole run: they anticorrelate for `H < 1/2`, which makes the walkers
    subdiffusive, and correlate for `H > 1/2`, which makes them superdiffusive.

    Args:
        n_walkers:
            The number of independent walkers.

        H:
            The Hurst index.

        D:
            The diffusion coefficient.

        n_dim:
            The number of dimensions.

    Returns:
        system:
            The model, with the positions `walkers.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import fractional_brownian_motion
        >>> fractional_brownian_motion(10, H=0.7).elements[0].values['H']
        0.7
    """
    walkers = FractionalWalker.group('walkers', n_walkers, x=np.zeros((n_walkers, n_dim)), D=D, H=H)
    return System(components=[walkers], space=ContinuousSpace(n_dim=n_dim))


@register_model(
    field='physics/soft_matter',
    tags=('anomalous diffusion', 'superdiffusion', 'random walk', 'stochastic map'),
    references=('Bertin 2021', 'Shlesinger, Zaslavsky, and Frisch 1995'),
    defaults={'until': 200.0, 'dt': 1.0},
)
def levy_flight(
    n_walkers: int = 500,
    *,
    mu: float = 1.2,
    l0: float = 1.0,
    n_dim: int = 2,
) -> System:
    """Return the Lévy flight, a walk whose jumps have lengths with the tail `l^(-1 - mu)` in random directions.

    For `mu < 2` the jump lengths have no variance, so the walk is dominated by its longest jumps and the typical
    distance travelled after `n` steps grows as `n^(1 / mu)`, faster than the `n^(1/2)` of a random walk.

    Args:
        n_walkers:
            The number of independent walkers.

        mu:
            The tail exponent of the jump lengths.

        l0:
            The shortest jump length.

        n_dim:
            The number of dimensions.

    Returns:
        system:
            The model, with the positions `walkers.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import levy_flight
        >>> levy_flight(10, mu=1.5).elements[0].values['mu']
        1.5
    """
    walkers = LevyFlier.group('walkers', n_walkers, x=np.zeros((n_walkers, n_dim)), mu=mu, l0=l0)
    return System(components=[walkers], space=ContinuousSpace(n_dim=n_dim))
