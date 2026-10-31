"""Stochastic process models: random walks, persistent and autoregressive walks, and diffusions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import JumpInteraction, MapInteraction, RateInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.laws.utils import bernoulli, normal
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._ageing_walker_view import AgeingWalkerView


class PersistentWalkerElement(Element):
    """Walker on the integers with a direction `d = ±1`, which it keeps with the probability `persistence`.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        d:
            The initial direction, 1 or -1.

        x:
            The initial position.

        persistence:
            The probability of keeping the direction at a step.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import PersistentWalkerElement
        >>> PersistentWalkerElement('walker', persistence=0.9).persistence
        0.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        d: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        x: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
        persistence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
    ) -> None:
        super().__init__(label, visual=visual)
        self.d = d
        self.x = x
        self.persistence = persistence


class MemoryWalkerElement(Element):
    """Walker on the real line whose velocity is an autoregressive process of memory `alpha` and noise `sigma`.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        v:
            The initial velocity.

        alpha:
            The memory of the velocity.

        sigma:
            The deviation of the noise of the velocity.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import MemoryWalkerElement
        >>> MemoryWalkerElement('walker', alpha=0.3).alpha
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.alpha = alpha
        self.sigma = sigma


class RandomWalkerElement(Element):
    """Walker on the integers that steps at the rate `rate`, up with the probability `bias`.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        rate:
            The rate of steps.

        bias:
            The probability that a step goes up.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import RandomWalkerElement
        >>> RandomWalkerElement('walker', bias=0.7).bias
        0.7
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        bias: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.rate = rate
        self.bias = bias


class WeakeningParticleElement(Element):
    """Brownian particle on the real line held by a restoring force of stiffness `k` that weakens beyond `ell`.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        k:
            The stiffness of the force near the origin.

        ell:
            The distance where the force starts to weaken.

        D:
            The diffusion coefficient.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import WeakeningParticleElement
        >>> WeakeningParticleElement('particle', D=0.5).D
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        ell: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.25,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.k = k
        self.ell = ell
        self.D = D


class AgeingWalkerElement(Element):
    """Walker that steps up with a probability that decays with the number of its earlier up steps.

    After `k` up steps, the next step is up with probability `p / (1 + k)^nu` and down otherwise.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        ups:
            The initial number of up steps.

        p:
            The probability of the first up step.

        nu:
            The exponent of the decay of the up probability with the number of up steps.

    Examples:
        >>> from skcomplex.models.mathematics.stochastic_processes import AgeingWalkerElement
        >>> group = GroupElement('walkers', element=AgeingWalkerElement, size=10, values={'nu': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('walkers', 'AgeingWalkerElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'x'}),
        *,
        x: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
        ups: Annotated[float, StateSpec(domain=NN)] = 0.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        nu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.ups = ups
        self.p = p
        self.nu = nu

    @classmethod
    def _get_view(cls) -> type[AgeingWalkerView]:
        """Return the view of the agents, which types their states and parameters."""
        return AgeingWalkerView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the stepping rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.walk)]

    def walk(self, agent: AgeingWalkerView, model: ModelView) -> None:
        """Take one step up or down.

        Args:
            agent:
                The walker.

            model:
                The model.
        """
        if model.rng.random() < agent.p / (1 + agent.ups) ** agent.nu:
            agent.x += 1
            agent.ups += 1
            return
        agent.x -= 1


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
    return System(
        elements=[GroupElement('walkers', element=AgeingWalkerElement, size=n_walkers, values={'p': p, 'nu': nu})],
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
        form='increment',
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
    step = MapInteraction('step', target='walkers', state='x', function=lambda s: s.target.v, form='increment')
    velocity = MapInteraction(
        'velocity',
        target='walkers',
        state='v',
        function=lambda s: s.target.alpha * s.target.v + normal('xi', 0, s.target.sigma),
    )
    return System(elements=[walkers], interactions=[step, velocity])
