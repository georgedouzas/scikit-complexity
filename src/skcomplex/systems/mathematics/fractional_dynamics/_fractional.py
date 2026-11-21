"""Fractional dynamics systems.

The module covers relaxation, oscillators, and chaotic flows with Caputo derivatives of states.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import fractional
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class FractionalRelaxationInteraction(Interaction):
    """The law of fractional relaxation `D^alpha x = -lam x`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose state `x` relaxes.

        order:
            The order `alpha` of the Caputo derivative, in `(0, 1]`.

        lam:
            The relaxation rate.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalRelaxationInteraction
        >>> FractionalRelaxationInteraction('relaxation_law', target='relaxation', order=0.5).order
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'relaxation',
        order: float = 0.6,
        lam: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.order = order
        self.lam = lam

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the fractional derivative of the state as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x = context.target.x
        self.laws_: list[BaseLaw] = [RateLaw(fractional(x, self.order), -context.lam * x)]


class FractionalRelaxationSystem(BaseSystem):
    """Fractional relaxation `D^alpha x = -lambda x`.

    The relaxation decays as the Mittag-Leffler function `E_alpha`.

    The solution `x(0) E_alpha(-lambda t^alpha)` decays slower than any exponential, as the power law `t^(-alpha)` at
    long times, since the Caputo derivative keeps the memory of the whole past. Order 1 gives exponential decay. The
    system has the state `relaxation.x`.

    Args:
        x:
            The initial value.

        order:
            The order `alpha` of the Caputo derivative, in `(0, 1]`.

        lam:
            The relaxation rate.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalRelaxationSystem
        >>> FractionalRelaxationSystem(order=0.5).prepare().system_.interactions[0].lam
        1.0
    """

    field = 'mathematics/fractional_dynamics'
    topics = ('fractional', 'memory', 'anomalous relaxation')
    references = ('Podlubny 1999', 'Trzaska 2023')
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        x: float = 1.0,
        *,
        order: float = 0.6,
        lam: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.order = order
        self.lam = lam

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the relaxation."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('relaxation', values={'x': self.x})],
            interactions=[
                FractionalRelaxationInteraction(
                    'relaxation_law',
                    target='relaxation',
                    order=float(self.order),
                    lam=self.lam,
                ),
            ],
        )


class FractionalOscillatorInteraction(Interaction):
    """The laws of the fractional damped oscillator.

    The oscillator is `D^alpha x = v`, `D^alpha v = -omega^2 x - 2 gamma v`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `v` oscillate.

        order:
            The order `alpha` of the Caputo derivatives, in `(0, 1]`.

        omega:
            The angular frequency.

        gamma:
            The damping rate.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalOscillatorInteraction
        >>> FractionalOscillatorInteraction('oscillator_law', target='oscillator', omega=2.0).omega
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'oscillator',
        order: float = 0.8,
        omega: float = 1.0,
        gamma: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.order = order
        self.omega = omega
        self.gamma = gamma

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the fractional derivatives of the displacement and its rate as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, v = context.target.x, context.target.v
        self.laws_: list[BaseLaw] = [
            RateLaw(fractional(x, self.order), v),
            RateLaw(fractional(v, self.order), -(context.omega**2) * x - 2 * context.gamma * v),
        ]


class FractionalOscillatorSystem(BaseSystem):
    """The fractional damped oscillator.

    The oscillator is `D^alpha x = v`, `D^alpha v = -omega^2 x - 2 gamma v`.

    With `gamma = 0` the system is the fractional oscillation equation of order `2 alpha`, whose memory damps the
    oscillations for `1/2 < alpha < 1` without any friction. Order 1 gives the harmonic oscillator. The system has the
    states `oscillator.x` and `oscillator.v`.

    Args:
        x:
            The initial displacement.

        v:
            The initial value of the first state's derivative of order `alpha`.

        order:
            The order `alpha` of the Caputo derivatives, in `(0, 1]`.

        omega:
            The angular frequency.

        gamma:
            The damping rate.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalOscillatorSystem
        >>> FractionalOscillatorSystem(order=0.9).prepare().system_.interactions[0].omega
        1.0
    """

    field = 'mathematics/fractional_dynamics'
    topics = ('fractional', 'memory', 'oscillator')
    references = ('Podlubny 1999', 'Trzaska 2023')
    until = 30.0
    dt = 0.1

    def __init__(
        self,
        x: float = 1.0,
        v: float = 0.0,
        *,
        order: float = 0.8,
        omega: float = 1.0,
        gamma: float = 0.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.v = v
        self.order = order
        self.omega = omega
        self.gamma = gamma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the oscillator."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('oscillator', values={'x': self.x, 'v': self.v})],
            interactions=[
                FractionalOscillatorInteraction(
                    'oscillator_law',
                    target='oscillator',
                    order=float(self.order),
                    omega=self.omega,
                    gamma=self.gamma,
                ),
            ],
        )


class FractionalVanDerPolInteraction(Interaction):
    """The laws of the fractional Van der Pol oscillator.

    The oscillator is `D^alpha x = y`, `D^alpha y = mu (1 - x^2) y - x`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `y` oscillate.

        order:
            The order `alpha` of the Caputo derivatives, in `(0, 1]`.

        mu:
            The strength of the nonlinear damping.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalVanDerPolInteraction
        >>> FractionalVanDerPolInteraction('oscillator_law', target='oscillator', mu=2.0).mu
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'oscillator',
        order: float = 0.9,
        mu: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.order = order
        self.mu = mu

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the fractional derivatives of the displacement and its rate as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        self.laws_: list[BaseLaw] = [
            RateLaw(fractional(x, self.order), y),
            RateLaw(fractional(y, self.order), context.mu * (1 - x**2) * y - x),
        ]


class FractionalVanDerPolSystem(BaseSystem):
    """The fractional Van der Pol oscillator.

    The oscillator is `D^alpha x = y`, `D^alpha y = mu (1 - x^2) y - x`.

    For orders close to 1 the trajectory approaches a closed cycle, and the memory of lower orders shrinks it toward the
    fixed point at the origin. The system has the states `oscillator.x` and `oscillator.y`.

    Args:
        x:
            The initial displacement.

        y:
            The initial value of the displacement's derivative of order `alpha`.

        order:
            The order `alpha` of the Caputo derivatives, in `(0, 1]`.

        mu:
            The strength of the nonlinear damping.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalVanDerPolSystem
        >>> FractionalVanDerPolSystem(mu=2.0).prepare().system_.interactions[0].mu
        2.0
    """

    field = 'mathematics/fractional_dynamics'
    topics = ('fractional', 'memory', 'limit cycle')
    references = ('Podlubny 1999', 'Trzaska 2023')
    until = 40.0
    dt = 0.1

    def __init__(
        self,
        x: float = 2.0,
        y: float = 0.0,
        *,
        order: float = 0.9,
        mu: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.order = order
        self.mu = mu

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the oscillator."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('oscillator', values={'x': self.x, 'y': self.y})],
            interactions=[
                FractionalVanDerPolInteraction(
                    'oscillator_law',
                    target='oscillator',
                    order=float(self.order),
                    mu=self.mu,
                ),
            ],
        )


class FractionalLorenzInteraction(Interaction):
    """The laws of the fractional Lorenz flow of Grigorenko and Grigorenko.

    The flow is `D^q x = sigma (y - x)`, `D^q y = x (rho - z) - y`, `D^q z = x y - beta z`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x`, `y`, and `z` the flow moves.

        order:
            The order `q` of the Caputo derivatives of all three states, in `(0, 1]`.

        sigma:
            The Prandtl number.

        rho:
            The Rayleigh number relative to its critical value.

        beta:
            The geometric factor.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalLorenzInteraction
        >>> FractionalLorenzInteraction('flow', target='lorenz', rho=20.0).rho
        20.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'lorenz',
        order: float = 0.995,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.order = order
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the fractional derivatives of the three coordinates as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y, z = context.target.x, context.target.y, context.target.z
        self.laws_: list[BaseLaw] = [
            RateLaw(fractional(x, self.order), context.sigma * (y - x)),
            RateLaw(fractional(y, self.order), x * (context.rho - z) - y),
            RateLaw(fractional(z, self.order), x * y - context.beta * z),
        ]


class FractionalLorenzSystem(BaseSystem):
    """The fractional Lorenz flow of Grigorenko and Grigorenko.

    The flow is chaotic for orders close enough to 1.

    Grigorenko and Grigorenko report chaos down to the total order 2.97. With Caputo derivatives from rest and the
    predictor-corrector, the flow is chaotic for the order 0.995 of every state, a total of 2.985, and settles on a
    fixed point of the two wings for 0.99 after a chaotic transient. The system has the states `lorenz.x`, `lorenz.y`,
    and `lorenz.z`.

    Args:
        x:
            The initial convection intensity.

        y:
            The initial temperature difference between rising and falling currents.

        z:
            The initial deviation of the vertical temperature profile.

        order:
            The order `q` of the Caputo derivatives of all three states, in `(0, 1]`.

        sigma:
            The Prandtl number.

        rho:
            The Rayleigh number relative to its critical value.

        beta:
            The geometric factor.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalLorenzSystem
        >>> FractionalLorenzSystem(rho=20.0).prepare().system_.interactions[0].rho
        20.0
    """

    field = 'mathematics/fractional_dynamics'
    topics = ('fractional', 'memory', 'chaos', 'strange attractor')
    references = ('Grigorenko and Grigorenko 2003', 'Trzaska 2023')
    until = 30.0
    dt = 0.05

    def __init__(
        self,
        x: float = 1.0,
        y: float = 1.0,
        z: float = 1.0,
        *,
        order: float = 0.995,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.z = z
        self.order = order
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('lorenz', values={'x': self.x, 'y': self.y, 'z': self.z})],
            interactions=[
                FractionalLorenzInteraction(
                    'flow',
                    target='lorenz',
                    order=float(self.order),
                    sigma=self.sigma,
                    rho=self.rho,
                    beta=self.beta,
                ),
            ],
        )
