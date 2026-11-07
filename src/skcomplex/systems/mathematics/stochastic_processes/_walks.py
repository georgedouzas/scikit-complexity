"""Stochastic process systems: random walks, persistent and autoregressive walks, and diffusions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import JumpInteraction, MapInteraction, RateInteraction
from ....core.components.laws.utils import bernoulli, normal
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


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
        >>> from skcomplex.systems.mathematics.stochastic_processes import MemoryWalkerElement
        >>> MemoryWalkerElement('walker', alpha=0.3).alpha
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(role='position')] = 0.0,
        v: Annotated[float, StateSpec(role='velocity')] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.alpha = alpha
        self.sigma = sigma


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
        >>> from skcomplex.systems.mathematics.stochastic_processes import PersistentWalkerElement
        >>> PersistentWalkerElement('walker', persistence=0.9).persistence
        0.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        d: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        x: Annotated[float, StateSpec(domain=ZZ, role='position')] = 0.0,
        persistence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
    ) -> None:
        super().__init__(label, visual=visual)
        self.d = d
        self.x = x
        self.persistence = persistence


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
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomWalkerElement
        >>> RandomWalkerElement('walker', bias=0.7).bias
        0.7
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=ZZ, role='position')] = 0.0,
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
        >>> from skcomplex.systems.mathematics.stochastic_processes import WeakeningParticleElement
        >>> WeakeningParticleElement('particle', D=0.5).D
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(role='position')] = 0.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        ell: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.25,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.k = k
        self.ell = ell
        self.D = D


class AgeingWalkerView(AgentView):
    """View of an ageing walker, with its states and parameters as attributes."""

    x: float
    ups: float
    p: float
    nu: float


class AgeingWalkerElement(Element):
    """Walker that steps up with a probability that decays with the number of its earlier up steps.

    After `k` up steps, the next step is up with probability `p / (1 + k)^nu` and down otherwise. The walker remembers
    its number of up steps at the end of the last step, from which its step follows.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.mathematics.stochastic_processes import AgeingWalkerElement
        >>> group = GroupElement('walkers', element=AgeingWalkerElement, size=10, values={'nu': 0.5})
        >>> group.label, group.element.__name__, group.size
        ('walkers', 'AgeingWalkerElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'x'}),
        *,
        x: Annotated[float, StateSpec(domain=ZZ, role='position')] = 0.0,
        ups: Annotated[float, StateSpec(domain=NN, lags=1)] = 0.0,
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


class RandomWalkSystem(BaseSystem):
    """Independent walkers on the integers that step at random times, a continuous-time random walk.

    Every walker steps up at rate `rate * bias` and down at rate `rate * (1 - bias)`, so its mean moves with velocity
    `rate (2 bias - 1)` and its variance grows as `4 rate bias (1 - bias) t`. The system has the positions `walkers.x`.

    Args:
        n_walkers:
            The number of walkers.

        rate:
            The rate of steps of a walker.

        bias:
            The probability that a step goes up.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomWalkSystem
        >>> RandomWalkSystem(20).prepare().system_.elements[0].prepare().size_
        20
    """

    field = 'mathematics/stochastic_processes'
    topics = ('random walk', 'jump process', 'diffusion')
    references = ('Pearson 1905', 'Argun et al. 2021', 'Charbonneau 2017', 'Jensen 2023')
    until = 100.0
    dt = 1.0

    def __init__(
        self,
        n_walkers: int = 100,
        *,
        rate: float = 1.0,
        bias: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_walkers = n_walkers
        self.rate = rate
        self.bias = bias

    def _build_system(self) -> System:
        """Return independent walkers on the integers that step at random times, a continuous-time random walk."""
        walkers = GroupElement(
            'walkers',
            element=RandomWalkerElement,
            size=self.n_walkers,
            values={'rate': self.rate, 'bias': self.bias},
        )
        up = JumpInteraction('up', target='walkers', changes={'x': 1}, rate=lambda s: s.target.rate * s.target.bias)
        down = JumpInteraction(
            'down',
            target='walkers',
            changes={'x': -1},
            rate=lambda s: s.target.rate * (1 - s.target.bias),
        )
        return System(self.label, elements=[walkers], interactions=[up, down])


class WeakeningOrnsteinUhlenbeckSystem(BaseSystem):
    """Brownian particles held by a restoring force that weakens with distance.

    The particles follow `dx = -k x / (1 + x²/ℓ²) dt + √(2D) dW`. Near the origin the force is that of an
    Ornstein-Uhlenbeck process, and far from it the force decays as `1 / x`, so the stationary density `(1 + x²/ℓ²)^(-k
    ℓ² / 2D)` has power-law tails. The system has the positions `particles.x`.

    Args:
        n_particles:
            The number of independent particles, all started at the origin.

        k:
            The stiffness of the force near the origin.

        ell:
            The distance where the force starts to weaken.

        D:
            The diffusion coefficient.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import WeakeningOrnsteinUhlenbeckSystem
        >>> WeakeningOrnsteinUhlenbeckSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'mathematics/stochastic_processes'
    topics = ('random walk', 'sde', 'heavy tails', 'power law')
    references = ('Jensen 2023',)
    until = 50.0
    dt = 0.05

    def __init__(
        self,
        n_particles: int = 200,
        *,
        k: float = 1.0,
        ell: float = 1.0,
        D: float = 0.25,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.k = k
        self.ell = ell
        self.D = D

    def _build_system(self) -> System:
        """Return Brownian particles held by a restoring force that weakens with distance."""
        particles = GroupElement(
            'particles',
            element=WeakeningParticleElement,
            size=self.n_particles,
            values={'k': self.k, 'ell': self.ell, 'D': self.D},
        )
        restoring = RateInteraction(
            'restoring',
            target='particles',
            state='x',
            rate=lambda s: -s.target.k * s.target.x / (1 + (s.target.x / s.target.ell) ** 2),
            noise=lambda s: sqrt(2 * s.target.D),
        )
        return System(self.label, elements=[particles], interactions=[restoring])


class AgeingRandomWalkSystem(BaseSystem):
    """Ageing random walks, whose up steps become rarer the more of them a walker has taken.

    After `k` up steps a walker steps up with probability `p / (1 + k)^nu`, so the walk is history dependent and, for
    `nu > 0`, its number of up steps grows as `n^(1 / (1 + nu))` after `n` steps. The system has the group `walkers`.

    Args:
        n_walkers:
            The number of independent walkers, all started at 0.

        p:
            The probability of the first up step.

        nu:
            The exponent of the decay of the up probability.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import AgeingRandomWalkSystem
        >>> AgeingRandomWalkSystem(5).prepare().system_.elements[0].prepare().size_
        5
    """

    field = 'mathematics/stochastic_processes'
    topics = ('random walk', 'agents', 'history dependence', 'ageing')
    references = ('Hanel, Thurner, and Gell-Mann 2014', 'Thurner et al. 2018')
    until = 100

    def __init__(
        self,
        n_walkers: int = 50,
        *,
        p: float = 0.5,
        nu: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_walkers = n_walkers
        self.p = p
        self.nu = nu

    def _build_system(self) -> System:
        """Return ageing random walks, whose up steps become rarer the more of them a walker has taken."""
        ageing = MapInteraction(
            'ageing',
            target='walkers',
            state='ups',
            function=lambda agent, system: float(system.rng.random() < agent.p / (1 + agent.ups) ** agent.nu),
            mode='increment',
            form='agent',
        )
        step = MapInteraction(
            'step',
            target='walkers',
            state='x',
            function=lambda agent, system: 2 * (agent.ups - agent.lags('ups')[-1]) - 1,
            mode='increment',
            form='agent',
        )
        return System(
            self.label,
            elements=[
                GroupElement(
                    'walkers',
                    element=AgeingWalkerElement,
                    size=self.n_walkers,
                    values={'p': self.p, 'nu': self.nu},
                ),
            ],
            interactions=[ageing, step],
        )


class PersistentRandomWalkSystem(BaseSystem):
    """Walkers on the integers that keep the direction of their last step with a probability.

    At every step a walker moves by its direction `d = ±1`, and keeps the direction with probability `q` and reverses it
    otherwise. Successive steps are correlated with `rho = 2 q - 1`, so the walk diffuses faster than a simple random
    walk for `q > 1/2`, with a mean squared displacement that grows as `t (1 + rho) / (1 - rho)` at long times. The
    system has the directions `walkers.d` and the positions `walkers.x`.

    Args:
        n_walkers:
            The number of independent walkers, all started at 0.

        persistence:
            The probability `q` of keeping the direction.

        random_state:
            The seed or generator of the initial directions.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PersistentRandomWalkSystem
        >>> PersistentRandomWalkSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'mathematics/stochastic_processes'
    topics = ('random walk', 'discrete map', 'persistence', 'diffusion')
    references = ('Jensen 2023',)
    until = 200
    dt = 1

    def __init__(
        self,
        n_walkers: int = 2000,
        *,
        persistence: float = 0.8,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_walkers = n_walkers
        self.persistence = persistence
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return walkers on the integers that keep the direction of their last step with a probability."""
        directions = np.random.default_rng(self.random_state).choice([-1.0, 1.0], size=self.n_walkers)
        walkers = GroupElement(
            'walkers',
            element=PersistentWalkerElement,
            size=self.n_walkers,
            values={'d': directions, 'persistence': self.persistence},
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
        return System(self.label, elements=[walkers], interactions=[turn, step])


class RandomWalkWithMemorySystem(BaseSystem):
    """Random walks whose velocity is an autoregressive process, `x ← x + v` and `v ← alpha v + xi`.

    The noise `xi` is normal with mean 0 and deviation `sigma`, and the walks start from `x = v = 0`. The pair `(x, v)`
    is Markovian while the displacements are correlated, so the mean squared displacement grows as `sigma² t / (1 -
    alpha)²` at long times instead of `sigma² t / (1 - alpha²)` for uncorrelated steps of the same variance. The system
    has the positions `walkers.x` and the velocities `walkers.v`.

    Args:
        n_walkers:
            The number of independent walkers.

        alpha:
            The memory `alpha` of the velocity, between 0 and 1.

        sigma:
            The deviation `sigma` of the noise.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomWalkWithMemorySystem
        >>> RandomWalkWithMemorySystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'mathematics/stochastic_processes'
    topics = ('random walk', 'discrete map', 'memory', 'autoregressive')
    references = ('Bertin 2021',)
    until = 200
    dt = 1

    def __init__(
        self,
        n_walkers: int = 2000,
        *,
        alpha: float = 0.5,
        sigma: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_walkers = n_walkers
        self.alpha = alpha
        self.sigma = sigma

    def _build_system(self) -> System:
        """Return random walks whose velocity is an autoregressive process, `x ← x + v` and `v ← alpha v + xi`."""
        walkers = GroupElement(
            'walkers',
            element=MemoryWalkerElement,
            size=self.n_walkers,
            values={'alpha': self.alpha, 'sigma': self.sigma},
        )
        step = MapInteraction('step', target='walkers', state='x', function=lambda s: s.target.v, mode='increment')
        velocity = MapInteraction(
            'velocity',
            target='walkers',
            state='v',
            function=lambda s: s.target.alpha * s.target.v + normal('xi', 0, s.target.sigma),
        )
        return System(self.label, elements=[walkers], interactions=[step, velocity])
