"""Stochastic process systems of walks.

The module covers random walks, persistent and autoregressive walks, and diffusions.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, sqrt

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli, both, normal, where
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Role

TIMES = ('continuous', 'discrete')


class MemoryWalkerElement(Element):
    """Walkers on the real line, each with its position `x` and its velocity `v`.

    Args:
        label:
            The label of the walkers.

        x:
            The initial position, one per walker or shared.

        v:
            The initial velocity, one per walker or shared.

        size:
            The number of walkers, or `None` for one.

        space:
            The line the walkers live on, whose axis `x` makes `x` their position, or `None`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import MemoryWalkerElement
        >>> MemoryWalkerElement('walker', v=0.3).v
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        v: object = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.v = v


class MemoryStepInteraction(Interaction):
    """Step of a walk whose velocity is an autoregressive process, `x <- x + v` and `v <- alpha v + xi`.

    The noise `xi` is normal with mean 0 and deviation `sigma`, drawn anew at every step for every walker.

    Args:
        label:
            The label of the interaction.

        target:
            The walkers whose positions `x` and velocities `v` the step sets.

        alpha:
            The memory of the velocity.

        sigma:
            The deviation of the noise of the velocity.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import MemoryStepInteraction
        >>> MemoryStepInteraction('step', alpha=0.3).alpha
        0.3
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'walkers', alpha: float = 0.5, sigma: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.sigma = sigma

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next position and the next velocity as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, v = context.target.x, context.target.v
        self.laws_: list[BaseLaw] = [
            UpdateLaw(x, x + v),
            UpdateLaw(v, context.alpha * v + normal('xi', 0, context.sigma)),
        ]


class PersistentWalkerElement(Element):
    """Walkers on the integers, each with its direction `d = ±1` and its position `x`.

    Args:
        label:
            The label of the walkers.

        d:
            The initial direction, 1 or -1, one per walker or shared.

        x:
            The initial position, one per walker or shared.

        size:
            The number of walkers, or `None` for one.

        space:
            The line the walkers live on, whose axis `x` makes `x` their position, or `None`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PersistentWalkerElement
        >>> PersistentWalkerElement('walker', d=-1.0).d
        -1.0
    """

    def __init__(
        self,
        label: str,
        *,
        d: Annotated[object, ZZ] = 1.0,
        x: Annotated[object, ZZ] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.d = d
        self.x = x


class PersistentStepInteraction(Interaction):
    """Step of a persistent walk, `x <- x + d`, whose direction `d` reverses with the probability `1 - persistence`.

    The reversal is drawn anew at every step for every walker.

    Args:
        label:
            The label of the interaction.

        target:
            The walkers whose directions `d` and positions `x` the step sets.

        persistence:
            The probability of keeping the direction at a step.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PersistentStepInteraction
        >>> PersistentStepInteraction('step', persistence=0.9).persistence
        0.9
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'walkers', persistence: float = 0.8) -> None:
        super().__init__(label)
        self.target = target
        self.persistence = persistence

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next direction and the next position as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        d, x = context.target.d, context.target.x
        self.laws_: list[BaseLaw] = [
            UpdateLaw(d, d * (1 - 2 * bernoulli('turn', 1 - context.persistence))),
            UpdateLaw(x, x + d),
        ]


class RandomWalkerElement(Element):
    """Walkers on the integers, each with its position `x`.

    Args:
        label:
            The label of the walkers.

        x:
            The initial position, one per walker or shared.

        size:
            The number of walkers, or `None` for one.

        space:
            The line the walkers live on, whose axis `x` makes `x` their position, or `None`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomWalkerElement
        >>> RandomWalkerElement('walker', x=2.0).x
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, ZZ] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x


class RandomStepInteraction(Interaction):
    """Steps of a walker on the integers at the rate `rate`, each up with the probability `bias`.

    The walker steps up at the rate `rate * bias` and down at the rate `rate * (1 - bias)`.

    Args:
        label:
            The label of the interaction.

        target:
            The walkers whose positions `x` the steps change.

        rate:
            The rate of steps.

        bias:
            The probability that a step goes up.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomStepInteraction
        >>> RandomStepInteraction('steps', bias=0.7).bias
        0.7
    """

    law_classes = (JumpLaw,)

    def __init__(self, label: str, *, target: Role = 'walkers', rate: float = 1.0, bias: float = 0.5) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate
        self.bias = bias

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the steps up and down as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x = context.target.x
        self.laws_: list[BaseLaw] = [
            JumpLaw(context.rate * context.bias, {x: 1}),
            JumpLaw(context.rate * (1 - context.bias), {x: -1}),
        ]


class DiscreteStepInteraction(Interaction):
    """Steps of a walker on the integers once per step, each up with the probability `bias`.

    Every step moves the walker by `+1` with probability `bias` and by `-1` otherwise. With a floor `f`, the walker
    moves to `max(f, x + step)`, so it reflects at the floor. With absorbing ends `(a, b)`, a walker at `a` or `b`
    stays there.

    Args:
        label:
            The label of the interaction.

        target:
            The walkers whose positions `x` the steps change.

        bias:
            The probability that a step goes up.

        absorbing:
            The ends `(a, b)` where a walker stops, or `None` for none.

        floor:
            The lowest position, where a walker reflects, or `None` for none.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import DiscreteStepInteraction
        >>> DiscreteStepInteraction('steps', absorbing=(0, 10)).absorbing
        (0, 10)
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'walkers',
        bias: float = 0.5,
        absorbing: tuple[float, float] | None = None,
        floor: float | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.bias = bias
        self.absorbing = absorbing
        self.floor = floor

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the next position of every walker as the rule.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x = context.target.x
        moved = x + 2 * bernoulli('up', self.bias) - 1
        if self.floor is not None:
            moved = where(moved < self.floor, self.floor, moved)
        if self.absorbing is not None:
            low, high = self.absorbing
            moved = where(both(x > low, x < high), moved, x)
        self.laws_: list[BaseLaw] = [UpdateLaw(x, moved)]


class WeakeningParticleElement(Element):
    """Particles on the real line, each with its position `x`.

    Args:
        label:
            The label of the particles.

        x:
            The initial position, one per particle or shared.

        size:
            The number of particles, or `None` for one.

        space:
            The line the particles live on, whose axis `x` makes `x` their position, or `None`.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import WeakeningParticleElement
        >>> WeakeningParticleElement('particle', x=0.5).x
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x


class WeakeningForceInteraction(Interaction):
    """Brownian motion in a restoring force that weakens with distance, `dx = -k x / (1 + x²/ℓ²) dt + √(2D) dW`.

    Args:
        label:
            The label of the interaction.

        target:
            The particles whose positions `x` the force and the noise move.

        k:
            The stiffness of the force near the origin.

        ell:
            The distance where the force starts to weaken.

        D:
            The diffusion coefficient.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import WeakeningForceInteraction
        >>> WeakeningForceInteraction('force', D=0.5).D
        0.5
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'particles',
        k: float = 1.0,
        ell: float = 1.0,
        D: float = 0.25,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k
        self.ell = ell
        self.D = D

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the restoring force and the noise as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x = context.target.x
        self.laws_: list[BaseLaw] = [
            RateLaw(x, -context.k * x / (1 + (x / context.ell) ** 2)),
            NoiseLaw(x, sqrt(2 * context.D)),
        ]


class AgeingWalkerView(AgentViewProtocol, Protocol):
    """View of an ageing walker, with its values as attributes."""

    x: float
    ups: float


class AgeingWalkerElement(Element):
    """Walkers on the integers, each with its position `x` and its number `ups` of up steps.

    Args:
        label:
            The label of the walkers.

        x:
            The initial position, one per walker or shared.

        ups:
            The initial number of up steps, one per walker or shared.

        size:
            The number of walkers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import AgeingWalkerElement
        >>> AgeingWalkerElement('walkers', ups=3.0, size=10).ups
        3.0
    """

    visual = VisualSpec('agents', {'color_by': 'x'})

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, ZZ] = 0.0,
        ups: Annotated[object, NN] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.ups = ups


class AgeingStepInteraction(Interaction):
    """Step of an ageing walker, up with the probability `p / (1 + k)^nu` after `k` up steps and down otherwise.

    Every walker runs the rule once per step.

    Args:
        label:
            The label of the interaction.

        target:
            The walkers whose positions `x` and numbers `ups` of up steps the rule sets.

        p:
            The probability of the first up step.

        nu:
            The exponent of the decay of the up probability with the number of up steps.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import AgeingStepInteraction
        >>> AgeingStepInteraction('step', nu=0.5).nu
        0.5
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(self, label: str, *, target: Role = 'walkers', p: float = 0.5, nu: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.p = p
        self.nu = nu

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the step of every walker as the rule.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseLaw] = [UpdateLaw(context.target, self._step)]

    def _step(self, agent: AgeingWalkerView, system: SystemView) -> None:
        """Step one walker up or down, counting its up steps."""
        up = float(system.rng.random() < self.p / (1 + agent.ups) ** self.nu)
        agent.ups += up
        agent.x += 2 * up - 1


class RandomWalkSystem(BaseSystem):
    """Independent walkers on the integers that step at random times or once per step.

    The walk runs in continuous time by default.

    In continuous time every walker steps up at rate `rate * bias` and down at rate `rate * (1 - bias)`, so its mean
    moves with velocity `rate (2 bias - 1)` and its variance grows as `4 rate bias (1 - bias) t`. In discrete time every
    walker steps by `+1` with probability `bias` or `-1` otherwise once per unit of time, and it may stop at absorbing
    ends, as in the gambler's ruin, or reflect at a floor. The system has the positions `walkers.x`.

    Args:
        n_walkers:
            The number of walkers.

        rate:
            The rate of steps of a walker in continuous time.

        bias:
            The probability that a step goes up.

        time:
            The time of the steps, `'continuous'` for steps at the times of a Poisson process of rate `rate` or
            `'discrete'` for one step per unit of time.

        start:
            The initial position of every walker.

        absorbing:
            The ends `(a, b)` where a walker stops for good, or `None` for none. It needs `time='discrete'`.

        floor:
            The lowest position, so a walker moves to `max(floor, x + step)`, or `None` for none. It needs
            `time='discrete'`.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The time is unknown, or the absorbing ends or the floor come in continuous time.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomWalkSystem
        >>> RandomWalkSystem(20).prepare().system_.elements[0].size
        20
        >>> walk = RandomWalkSystem(4, time='discrete', start=3, absorbing=(0, 10)).prepare()
        >>> walk.simulate(1, random_state=0).get_values('walkers.x')[-1].tolist()
        [2.0, 4.0, 2.0, 4.0]
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
        time: str = 'continuous',
        start: float = 0.0,
        absorbing: tuple[float, float] | None = None,
        floor: float | None = None,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.rate = rate
        self.bias = bias
        self.time = time
        self.start = start
        self.absorbing = absorbing
        self.floor = floor

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the walkers on a line and their steps in the chosen time."""
        super()._prepare(context)
        walkers = RandomWalkerElement(
            'walkers',
            x=float(self.start),
            size=self.n_walkers,
            space=ContinuousSpace(n_dim=1),
        )
        self.system_ = System(self.label, elements=[walkers], interactions=[self._get_steps()])

    def _get_steps(self) -> Interaction:
        """Return the steps of the walkers in the chosen time.

        Raises:
            ValidationError:
                The time is unknown, or the absorbing ends or the floor come in continuous time.
        """
        if self.time not in TIMES:
            error_msg = f'Unknown time {self.time!r}. Use one of {", ".join(TIMES)}.'
            raise ValidationError(error_msg)
        if self.time == 'discrete':
            return DiscreteStepInteraction(
                'steps',
                target='walkers',
                bias=self.bias,
                absorbing=self.absorbing,
                floor=self.floor,
            )
        if self.absorbing is not None or self.floor is not None:
            error_msg = (
                f"The absorbing ends and the floor of {self.label!r} need steps once per unit of time. "
                "Give time='discrete'."
            )
            raise ValidationError(error_msg)
        return RandomStepInteraction('steps', target='walkers', rate=self.rate, bias=self.bias)


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import WeakeningOrnsteinUhlenbeckSystem
        >>> WeakeningOrnsteinUhlenbeckSystem(10).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.k = k
        self.ell = ell
        self.D = D

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the particles on a line and the force that holds them."""
        super()._prepare(context)
        particles = WeakeningParticleElement('particles', size=self.n_particles, space=ContinuousSpace(n_dim=1))
        restoring = WeakeningForceInteraction('restoring', target='particles', k=self.k, ell=self.ell, D=self.D)
        self.system_ = System(self.label, elements=[particles], interactions=[restoring])


class AgeingRandomWalkSystem(BaseSystem):
    """Ageing random walks.

    The up steps become rarer the more of them a walker has taken.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import AgeingRandomWalkSystem
        >>> AgeingRandomWalkSystem(5).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.p = p
        self.nu = nu

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the walkers and the rule of their steps."""
        super()._prepare(context)
        step = AgeingStepInteraction('step', target='walkers', p=self.p, nu=self.nu)
        walkers = AgeingWalkerElement('walkers', size=self.n_walkers)
        self.system_ = System(self.label, elements=[walkers], interactions=[step])


class PersistentRandomWalkSystem(BaseSystem):
    """Walkers on the integers that keep the direction of their last step.

    A walker keeps the direction with a probability.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import PersistentRandomWalkSystem
        >>> PersistentRandomWalkSystem(10).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.persistence = persistence
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the walkers on a line and their persistent steps."""
        super()._prepare(context)
        directions = np.random.default_rng(self.random_state).choice([-1.0, 1.0], size=self.n_walkers)
        walkers = PersistentWalkerElement('walkers', d=directions, size=self.n_walkers, space=ContinuousSpace(n_dim=1))
        step = PersistentStepInteraction('step', target='walkers', persistence=self.persistence)
        self.system_ = System(self.label, elements=[walkers], interactions=[step])


class RandomWalkWithMemorySystem(BaseSystem):
    """Random walks whose velocity is an autoregressive process.

    The walks follow `x ← x + v` and `v ← alpha v + xi`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomWalkWithMemorySystem
        >>> RandomWalkWithMemorySystem(10).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.alpha = alpha
        self.sigma = sigma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the walkers on a line and their steps."""
        super()._prepare(context)
        walkers = MemoryWalkerElement('walkers', size=self.n_walkers, space=ContinuousSpace(n_dim=1))
        step = MemoryStepInteraction('step', target='walkers', alpha=self.alpha, sigma=self.sigma)
        self.system_ = System(self.label, elements=[walkers], interactions=[step])
