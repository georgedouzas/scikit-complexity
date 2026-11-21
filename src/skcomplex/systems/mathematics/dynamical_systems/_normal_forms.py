"""Dynamical systems of the normal forms of bifurcations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class SaddleNodeFormInteraction(Interaction):
    """The laws of the saddle-node normal form `x' = μ - x²`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose state `x` the flow moves.

        mu:
            The bifurcation parameter.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SaddleNodeFormInteraction
        >>> SaddleNodeFormInteraction('saddle_node_form', target='form', mu=0.5).mu
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'form',
        mu: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x = context.target.x
        mu = context.mu
        self.laws_: list[BaseComponent] = [
            RateLaw(x, mu - x**2),
        ]


class SaddleNodeFormSystem(BaseSystem):
    """The saddle-node normal form `x' = μ - x²`.

    Its two equilibria `±√μ` meet and vanish at μ = 0.

    The system has the state `form.x`.

    Args:
        x:
            The initial value.

        mu:
            The bifurcation parameter.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SaddleNodeFormSystem
        >>> SaddleNodeFormSystem(mu=0.5).prepare().system_.interactions[0].mu
        0.5
    """

    field = 'mathematics/dynamical_systems'
    topics = ('bifurcation', 'normal form', 'ode')
    references = ('Gros 2015', 'Dilão 2023')
    until = 10.0
    dt = 0.05

    def __init__(
        self,
        x: float = 0.0,
        *,
        mu: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.mu = mu

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('form', values={'x': self.x})],
            interactions=[SaddleNodeFormInteraction('saddle_node_form', target='form', mu=self.mu)],
        )


class HopfFormInteraction(Interaction):
    """The laws of the Hopf normal form.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `y` the flow moves.

        mu:
            The bifurcation parameter.

        omega:
            The angular frequency.

        a:
            The cubic coefficient, the sign of the first Lyapunov coefficient.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HopfFormInteraction
        >>> HopfFormInteraction('hopf_form', target='form', mu=0.5).mu
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'form',
        mu: float = 0.25,
        omega: float = 1.0,
        a: float = -1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.mu = mu
        self.omega = omega
        self.a = a

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        mu, omega, a = context.mu, context.omega, context.a
        self.laws_: list[BaseComponent] = [
            RateLaw(x, mu * x - omega * y + a * x * (x**2 + y**2)),
            RateLaw(y, omega * x + mu * y + a * y * (x**2 + y**2)),
        ]


class HopfFormSystem(BaseSystem):
    """The Hopf normal form.

    Its focus gives birth to a limit cycle of radius `√(-μ / a)` as μ crosses 0.

    `x' = μx - ωy + a x (x² + y²)` and `y' = ωx + μy + a y (x² + y²)`, supercritical for `a < 0`. The system has the
    states `form.x` and `form.y`.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        mu:
            The bifurcation parameter.

        omega:
            The angular frequency.

        a:
            The cubic coefficient, the sign of the first Lyapunov coefficient.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HopfFormSystem
        >>> HopfFormSystem(mu=-0.1).prepare().system_.interactions[0].mu
        -0.1
    """

    field = 'mathematics/dynamical_systems'
    topics = ('bifurcation', 'normal form', 'ode', 'limit cycle')
    references = ('Gros 2015', 'Dilão 2023')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        x: float = 0.1,
        y: float = 0.0,
        *,
        mu: float = 0.25,
        omega: float = 1.0,
        a: float = -1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.a = a

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('form', values={'x': self.x, 'y': self.y})],
            interactions=[HopfFormInteraction('hopf_form', target='form', mu=self.mu, omega=self.omega, a=self.a)],
        )


class TakensBogdanovFormInteraction(Interaction):
    """The laws of the Takens-Bogdanov normal form `x' = y`, `y' = β1 + β2 x + x² + s x y`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose states `x` and `y` the flow moves.

        beta1:
            The first unfolding parameter.

        beta2:
            The second unfolding parameter.

        sign:
            The sign `s` of the mixed term, `-1` or `1`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TakensBogdanovFormInteraction
        >>> TakensBogdanovFormInteraction('takens_bogdanov_form', target='form', beta1=0.5).beta1
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'form',
        beta1: float = 0.1,
        beta2: float = -1.0,
        sign: float = -1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.beta1 = beta1
        self.beta2 = beta2
        self.sign = sign

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        beta1, beta2, sign = context.beta1, context.beta2, context.sign
        self.laws_: list[BaseComponent] = [
            RateLaw(x, y),
            RateLaw(y, beta1 + beta2 * x + x**2 + sign * x * y),
        ]


class TakensBogdanovFormSystem(BaseSystem):
    """The Takens-Bogdanov normal form `x' = y`, `y' = β1 + β2 x + x² + s x y`.

    In this form a saddle-node meets a Hopf.

    The system has the states `form.x` and `form.y`.

    Args:
        x:
            The initial position.

        y:
            The initial velocity.

        beta1:
            The first unfolding parameter.

        beta2:
            The second unfolding parameter.

        sign:
            The sign `s` of the mixed term, `-1` or `1`.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TakensBogdanovFormSystem
        >>> TakensBogdanovFormSystem(beta1=0.0).prepare().system_.interactions[0].beta1
        0.0
    """

    field = 'mathematics/dynamical_systems'
    topics = ('bifurcation', 'normal form', 'ode', 'codimension two')
    references = ('Gros 2015', 'Dilão 2023')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        x: float = 0.2,
        y: float = 0.0,
        *,
        beta1: float = 0.1,
        beta2: float = -1.0,
        sign: float = -1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.beta1 = beta1
        self.beta2 = beta2
        self.sign = sign

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('form', values={'x': self.x, 'y': self.y})],
            interactions=[
                TakensBogdanovFormInteraction(
                    'takens_bogdanov_form',
                    target='form',
                    beta1=self.beta1,
                    beta2=self.beta2,
                    sign=self.sign,
                ),
            ],
        )


class CuspFormInteraction(Interaction):
    """The laws of the cusp normal form `x' = β1 + β2 x - x³`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose state `x` the flow moves.

        beta1:
            The asymmetry.

        beta2:
            The splitting parameter.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CuspFormInteraction
        >>> CuspFormInteraction('cusp_form', target='form', beta1=0.5).beta1
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'form',
        beta1: float = 0.0,
        beta2: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.beta1 = beta1
        self.beta2 = beta2

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate of every state as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x = context.target.x
        beta1, beta2 = context.beta1, context.beta2
        self.laws_: list[BaseComponent] = [
            RateLaw(x, beta1 + beta2 * x - x**3),
        ]


class CuspFormSystem(BaseSystem):
    """The cusp normal form `x' = β1 + β2 x - x³`.

    The form is bistable inside the cusp `27 β1² < 4 β2³`.

    The system has the state `form.x`.

    Args:
        x:
            The initial value.

        beta1:
            The asymmetry.

        beta2:
            The splitting parameter.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CuspFormSystem
        >>> CuspFormSystem(beta1=0.2).prepare().system_.interactions[0].beta1
        0.2
    """

    field = 'mathematics/dynamical_systems'
    topics = ('bifurcation', 'normal form', 'ode', 'bistability', 'hysteresis')
    references = ('Gros 2015', 'Dilão 2023')
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        x: float = 0.5,
        *,
        beta1: float = 0.0,
        beta2: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.beta1 = beta1
        self.beta2 = beta2

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the flow and its laws."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[Element('form', values={'x': self.x})],
            interactions=[CuspFormInteraction('cusp_form', target='form', beta1=self.beta1, beta2=self.beta2)],
        )
