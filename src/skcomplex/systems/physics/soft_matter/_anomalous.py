"""Anomalous diffusion.

The module covers random walks with power-law waiting times and flights, heterogeneity, and memory.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression, oo, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import draw, normal
from ....core.components.spaces import ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver, FractionalEulerMaruyamaSolver, RK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ContinuousTimeWalkerElement(Element):
    """Walkers at the positions `x` that wait and jump.

    Args:
        label:
            The label of the walkers.

        x:
            The initial position, one row per walker or shared.

        size:
            The number of walkers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ContinuousTimeWalkerElement
        >>> ContinuousTimeWalkerElement('walkers', x=(1.0, 0.0), size=10).x
        (1.0, 0.0)
    """

    def __init__(self, label: str, *, x: object = (0.0, 0.0), size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class WaitingJumpInteraction(Interaction):
    """Jumps by Gaussian steps after power-law waiting times.

    The waiting time is `tau0 r^(-1 / alpha)` for `r` uniform, and every jump adds a normal step of standard deviation
    `sigma` to every coordinate of the position `x`. It is a tracer trapped in a porous rock, a charge carrier in an
    amorphous semiconductor, or a molecule in a crowded cell.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walkers.

        alpha:
            The tail exponent of the waiting times, below 1 for waiting times without a mean.

        tau0:
            The shortest waiting time.

        sigma:
            The standard deviation of the jumps along every axis.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import WaitingJumpInteraction
        >>> WaitingJumpInteraction('jumps', target='walkers', alpha=0.7).alpha
        0.7
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        alpha: Annotated[float, RealSet.open(0, oo)] = 0.5,
        tau0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        sigma: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.tau0 = tau0
        self.sigma = sigma

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the jumps of the positions of the target after the waiting times.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s = context
        waiting = draw('wait', 'pareto', b=s.alpha, scale=s.tau0)
        reset = {item: item + normal(f'jump{index}', 0, s.sigma) for index, item in enumerate(s.target.x)}
        self.laws_ = [JumpLaw(waiting=waiting, reset=reset, label=self.label)]


class FractionalWalkerElement(Element):
    """Walkers at the positions `x`, driven by fractional Brownian motion.

    Args:
        label:
            The label of the walkers.

        x:
            The initial position, one row per walker or shared.

        size:
            The number of walkers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import FractionalWalkerElement
        >>> FractionalWalkerElement('walkers', x=(1.0, 0.0), size=10).x
        (1.0, 0.0)
    """

    def __init__(self, label: str, *, x: object = (0.0, 0.0), size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class FractionalNoiseInteraction(Interaction):
    """Fractional Brownian noise, `dx = sqrt(2 D) dB_H`, whose increments are correlated in time.

    The increments of the position `x` anticorrelate for a Hurst index `H < 1/2` and correlate for `H > 1/2`. It is a
    monomer of a long polymer, a tracer in a viscoelastic gel, or the level of a river over the years.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walkers.

        D:
            The diffusion coefficient.

        H:
            The Hurst index of the driving fractional Brownian motion.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import FractionalNoiseInteraction
        >>> FractionalNoiseInteraction('noise', target='walkers', H=0.7).H
        0.7
    """

    law_classes = (NoiseLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        D: Annotated[float, RealSet.open(0, oo)] = 1.0,
        H: Annotated[float, RealSet.open(0, 1)] = 0.3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.D = D
        self.H = H

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the fractional noise of the positions of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        self.laws_ = [NoiseLaw(context.target.x, sqrt(2 * context.D), hurst=context.H)]


class LevyFlierElement(Element):
    """Walkers at the positions `x` that jump at every step.

    Args:
        label:
            The label of the walkers.

        x:
            The initial position, one row per walker or shared.

        size:
            The number of walkers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import LevyFlierElement
        >>> LevyFlierElement('walkers', x=(1.0, 0.0), size=10).x
        (1.0, 0.0)
    """

    def __init__(self, label: str, *, x: object = (0.0, 0.0), size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x


class LevyJumpInteraction(Interaction):
    """Jumps at every step in a random direction, with lengths of the power-law tail `l^(-1 - mu)`.

    The length of a jump is `l0 r^(-1 / mu)` for `r` uniform, and its direction is uniform. It is an albatross that
    forages, a photon in a hot gas, or a bank note passed between towns.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walkers.

        mu:
            The tail exponent of the jump lengths, below 2 for jumps of infinite variance.

        l0:
            The shortest jump length.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import LevyJumpInteraction
        >>> LevyJumpInteraction('flights', target='walkers', mu=1.5).mu
        1.5
    """

    law_classes = (UpdateLaw,)
    draws = True

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        mu: Annotated[float, RealSet.open(0, oo)] = 1.2,
        l0: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.l0 = l0

    def _get_direction(self, size: int) -> list[Expression]:
        """Return a unit vector of uniformly random direction, drawn from standard normal components."""
        components = [normal(f'direction{index}') for index in range(size)]
        length = sqrt(sum(component**2 for component in components))
        return [component / length for component in components]

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the position after a jump in a random direction.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        x = context.target.x
        length = draw('length', 'pareto', b=context.mu, scale=context.l0)
        steps = self._get_direction(len(list(x)))
        self.laws_ = [UpdateLaw(x, vector(SR, [item + length * step for item, step in zip(x, steps, strict=True)]))]


class LevyWalkerElement(Element):
    """Walkers at the positions `x` that fly with the velocities `v`.

    Args:
        label:
            The label of the walkers.

        x:
            The initial position, one row per walker or shared.

        v:
            The initial velocity, one row per walker or shared.

        size:
            The number of walkers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import LevyWalkerElement
        >>> LevyWalkerElement('walkers', v=(1.0, 0.0), size=10).v
        (1.0, 0.0)
    """

    def __init__(self, label: str, *, x: object = (0.0, 0.0), v: object = (0.0, 0.0), size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.v = v


class LevyFlightTimeInteraction(Interaction):
    """Flights at constant speed in random directions that last power-law flight times.

    The position `x` moves with the velocity `v`, and after a flight time `tau0 r^(-1 / sigma)` for `r` uniform the
    velocity turns to a uniformly random direction at the speed `speed`. It is a bacterium that runs and tumbles, a
    tracer in a turbulent flow, or the gaze of an eye that scans a scene.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walkers.

        sigma:
            The tail exponent of the flight times, between 1 and 2 for superdiffusion.

        tau0:
            The shortest flight time.

        speed:
            The speed of every flight.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import LevyFlightTimeInteraction
        >>> LevyFlightTimeInteraction('turns', target='walkers', speed=2.0).speed
        2.0
    """

    law_classes = (RateLaw, JumpLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        sigma: Annotated[float, RealSet.open(0, oo)] = 1.5,
        tau0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        speed: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.sigma = sigma
        self.tau0 = tau0
        self.speed = speed

    def _get_direction(self, size: int) -> list[Expression]:
        """Return a unit vector of uniformly random direction, drawn from standard normal components."""
        components = [normal(f'direction{index}') for index in range(size)]
        length = sqrt(sum(component**2 for component in components))
        return [component / length for component in components]

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the flight of the positions and the turns of the velocities of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s = context
        x, v = s.target.x, s.target.v
        waiting = draw('flight', 'pareto', b=s.sigma, scale=s.tau0)
        directions = self._get_direction(len(list(v)))
        reset = {item: s.speed * direction for item, direction in zip(v, directions, strict=True)}
        self.laws_ = [RateLaw(x, v), JumpLaw(waiting=waiting, reset=reset, label=self.label)]


class TransientWalkerElement(Element):
    """Brownian walkers at the positions `x` with the diffusion coefficients `D`.

    Args:
        label:
            The label of the walkers.

        x:
            The initial position, one row per walker or shared.

        D:
            The initial diffusion coefficient, one per walker or shared.

        size:
            The number of walkers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import TransientWalkerElement
        >>> TransientWalkerElement('walkers', D=0.2, size=10).D
        0.2
    """

    def __init__(self, label: str, *, x: object = (0.0, 0.0), D: object = 0.5, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.D = D


class TransientDiffusionInteraction(Interaction):
    """Brownian motion whose diffusion coefficient `D` is redrawn at random times.

    The position `x` diffuses with `D`, and after an exponential waiting time of mean `D^(-gamma)` the coefficient is
    redrawn from the density `sigma D^(sigma - 1)` on `(0, 1)`, the beta distribution `B(sigma, 1)`, so slow walkers
    wait long before they change. It is a protein that binds and unbinds in a cell membrane, or a tracer in a
    heterogeneous gel.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the walkers.

        sigma:
            The exponent of the density of diffusion coefficients near zero.

        gamma:
            The exponent of the mean waiting time `D^(-gamma)`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import TransientDiffusionInteraction
        >>> TransientDiffusionInteraction('changes', target='walkers', gamma=2.0).gamma
        2.0
    """

    law_classes = (NoiseLaw, JumpLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        sigma: Annotated[float, RealSet.open(0, oo)] = 0.5,
        gamma: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.sigma = sigma
        self.gamma = gamma

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the noise of the positions of the target and the changes of their diffusion coefficients.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s = context
        x, D = s.target.x, s.target.D
        waiting = draw('wait', 'expon', scale=D ** (-s.gamma))
        reset = {D: draw('coefficient', 'beta', a=s.sigma, b=1)}
        self.laws_ = [NoiseLaw(x, sqrt(2 * D)), JumpLaw(waiting=waiting, reset=reset, label=self.label)]


class ContinuousTimeRandomWalkSystem(BaseSystem):
    """The continuous-time random walk.

    Gaussian jumps follow waiting times with the tail `tau^(-1 - alpha)`.

    For `alpha < 1` the waiting times have no mean, so the number of jumps grows as `t^alpha` and the ensemble mean
    squared displacement grows as `t^alpha`, slower than Brownian motion. The walkers renew on their own clocks. The
    system has the positions `walkers.x` and the jump count `jumps.renewals`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ContinuousTimeRandomWalkSystem
        >>> ContinuousTimeRandomWalkSystem(10, alpha=0.7).prepare().system_.interactions[0].alpha
        0.7
    """

    field = 'physics/soft_matter'
    topics = ('anomalous diffusion', 'renewal process', 'subdiffusion', 'random walk')
    references = ('Argun et al. 2021', 'Metzler and Klafter 2000', 'Montroll and Weiss 1965', 'Bertin 2021')
    until = 1000.0
    dt = 5.0

    def __init__(
        self,
        n_walkers: int = 500,
        *,
        alpha: float = 0.5,
        tau0: float = 1.0,
        sigma: float = 1.0,
        n_dim: int = 2,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.alpha = alpha
        self.tau0 = tau0
        self.sigma = sigma
        self.n_dim = n_dim

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of walkers that wait power-law times and jump by Gaussian steps.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        walkers = ContinuousTimeWalkerElement('walkers', x=np.zeros((self.n_walkers, self.n_dim)), size=self.n_walkers)
        jumps = WaitingJumpInteraction('jumps', target='walkers', alpha=self.alpha, tau0=self.tau0, sigma=self.sigma)
        self.system_ = System(
            self.label,
            elements=[walkers],
            interactions=[jumps],
            space=ContinuousSpace(n_dim=self.n_dim),
        )


class LevyWalkSystem(BaseSystem):
    """The Lévy walk.

    Flights at constant speed in random directions last times with the tail `tau^(-1 - sigma)`.

    For `1 < sigma < 2` the flight times have a mean but no variance, so long flights dominate and the ensemble mean
    squared displacement grows as `t^(3 - sigma)`, faster than Brownian motion and slower than ballistic motion. The
    system has the positions `walkers.x`, the velocities `walkers.v`, and the turn count `turns.renewals`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import LevyWalkSystem
        >>> LevyWalkSystem(10, sigma=1.2).prepare().system_.interactions[0].sigma
        1.2
    """

    field = 'physics/soft_matter'
    topics = ('anomalous diffusion', 'renewal process', 'superdiffusion', 'random walk')
    references = ('Argun et al. 2021', 'Zaburdaev, Denisov, and Klafter 2015', 'Shlesinger, West, and Klafter 1987')
    until = 500.0
    dt = 2.5

    def __init__(
        self,
        n_walkers: int = 500,
        *,
        sigma: float = 1.5,
        tau0: float = 1.0,
        speed: float = 1.0,
        n_dim: int = 2,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.sigma = sigma
        self.tau0 = tau0
        self.speed = speed
        self.n_dim = n_dim
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of walkers that fly at constant speed for power-law flight times.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        directions = np.random.default_rng(self.random_state).standard_normal((self.n_walkers, self.n_dim))
        velocities = self.speed * directions / np.linalg.norm(directions, axis=1, keepdims=True)
        walkers = LevyWalkerElement(
            'walkers',
            x=np.zeros((self.n_walkers, self.n_dim)),
            v=velocities,
            size=self.n_walkers,
        )
        turns = LevyFlightTimeInteraction(
            'turns',
            target='walkers',
            sigma=self.sigma,
            tau0=self.tau0,
            speed=self.speed,
        )
        self.system_ = System(
            self.label,
            elements=[walkers],
            interactions=[turns],
            space=ContinuousSpace(n_dim=self.n_dim),
            solver=RK4Solver(dt=0.5),
        )


class AnnealedTransientTimeMotionSystem(BaseSystem):
    """Annealed transient time motion.

    The diffusion coefficient of Brownian walkers changes at random times.

    Every walker diffuses with a coefficient `D` of density `sigma D^(sigma - 1)` on `(0, 1)` and keeps it for an
    exponential time of mean `D^(-gamma)`, so slow walkers stay slow for long. For `sigma < gamma` the ensemble mean
    squared displacement grows as `t^(sigma / gamma)`. The system has the positions `walkers.x`, the coefficients
    `walkers.D`, and the change count `changes.renewals`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import AnnealedTransientTimeMotionSystem
        >>> AnnealedTransientTimeMotionSystem(10, gamma=2.0).prepare().system_.interactions[0].gamma
        2.0
    """

    field = 'physics/soft_matter'
    topics = ('anomalous diffusion', 'renewal process', 'heterogeneity', 'sde')
    references = ('Argun et al. 2021', 'Massignan et al. 2014')
    until = 500.0
    dt = 2.5

    def __init__(
        self,
        n_walkers: int = 500,
        *,
        sigma: float = 0.5,
        gamma: float = 1.0,
        n_dim: int = 2,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.sigma = sigma
        self.gamma = gamma
        self.n_dim = n_dim
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of walkers whose diffusion coefficient changes at random times.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        coefficients = np.random.default_rng(self.random_state).beta(self.sigma, 1.0, self.n_walkers)
        walkers = TransientWalkerElement(
            'walkers',
            x=np.zeros((self.n_walkers, self.n_dim)),
            D=coefficients,
            size=self.n_walkers,
        )
        changes = TransientDiffusionInteraction('changes', target='walkers', sigma=self.sigma, gamma=self.gamma)
        self.system_ = System(
            self.label,
            elements=[walkers],
            interactions=[changes],
            space=ContinuousSpace(n_dim=self.n_dim),
            solver=EulerMaruyamaSolver(dt=0.5),
        )


class FractionalBrownianMotionSystem(BaseSystem):
    """Fractional Brownian motion, `dx = sqrt(2 D) dB_H`.

    Along every axis `<x^2> = 2 D t^(2 H)`.

    The increments are correlated over the whole run: they anticorrelate for `H < 1/2`, which makes the walkers
    subdiffusive, and correlate for `H > 1/2`, which makes them superdiffusive. The system has the positions
    `walkers.x`.

    Args:
        n_walkers:
            The number of independent walkers.

        H:
            The Hurst index.

        D:
            The diffusion coefficient.

        n_dim:
            The number of dimensions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import FractionalBrownianMotionSystem
        >>> FractionalBrownianMotionSystem(10, H=0.7).prepare().system_.interactions[0].H
        0.7
    """

    field = 'physics/soft_matter'
    topics = ('anomalous diffusion', 'fractional noise', 'long memory', 'sde')
    references = ('Argun et al. 2021', 'Mandelbrot and Van Ness 1968')
    until = 10.0
    dt = 0.05

    def __init__(
        self,
        n_walkers: int = 500,
        *,
        H: float = 0.3,
        D: float = 1.0,
        n_dim: int = 2,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.H = H
        self.D = D
        self.n_dim = n_dim

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of walkers driven by fractional Brownian motion.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        walkers = FractionalWalkerElement('walkers', x=np.zeros((self.n_walkers, self.n_dim)), size=self.n_walkers)
        self.system_ = System(
            self.label,
            elements=[walkers],
            interactions=[FractionalNoiseInteraction('noise', target='walkers', D=self.D, H=self.H)],
            space=ContinuousSpace(n_dim=self.n_dim),
            solver=FractionalEulerMaruyamaSolver(dt=0.01),
        )


class LevyFlightSystem(BaseSystem):
    """The Lévy flight.

    The walk jumps in random directions by lengths with the tail `l^(-1 - mu)`.

    For `mu < 2` the jump lengths have no variance, so the walk is dominated by its longest jumps and the typical
    distance travelled after `n` steps grows as `n^(1 / mu)`, faster than the `n^(1/2)` of a random walk. The system has
    the positions `walkers.x`.

    Args:
        n_walkers:
            The number of independent walkers.

        mu:
            The tail exponent of the jump lengths.

        l0:
            The shortest jump length.

        n_dim:
            The number of dimensions.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import LevyFlightSystem
        >>> LevyFlightSystem(10, mu=1.5).prepare().system_.interactions[0].mu
        1.5
    """

    draws = True

    field = 'physics/soft_matter'
    topics = ('anomalous diffusion', 'superdiffusion', 'random walk', 'stochastic map')
    references = ('Bertin 2021', 'Shlesinger, Zaslavsky, and Frisch 1995')
    until = 200.0
    dt = 1.0

    def __init__(
        self,
        n_walkers: int = 500,
        *,
        mu: float = 1.2,
        l0: float = 1.0,
        n_dim: int = 2,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_walkers = n_walkers
        self.mu = mu
        self.l0 = l0
        self.n_dim = n_dim

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of walkers that jump with power-law lengths in random directions.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        walkers = LevyFlierElement('walkers', x=np.zeros((self.n_walkers, self.n_dim)), size=self.n_walkers)
        self.system_ = System(
            self.label,
            elements=[walkers],
            interactions=[LevyJumpInteraction('flights', target='walkers', mu=self.mu, l0=self.l0)],
            space=ContinuousSpace(n_dim=self.n_dim),
        )
