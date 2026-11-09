"""Dynamical systems of the normal forms of bifurcations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class SaddleNodeFormSystem(DynamicalSystem):
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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SaddleNodeFormSystem
        >>> SaddleNodeFormSystem(mu=0.5).prepare().system_.elements[0].get_value('mu')
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
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.mu = mu

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'form'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'mu': self.mu,
        }

    def _get_equations(self) -> Equations:
        """Return the rate of the value."""
        return {
            'x': lambda s: s.target.mu - s.target.x**2,
        }


class HopfFormSystem(DynamicalSystem):
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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HopfFormSystem
        >>> HopfFormSystem(mu=-0.1).prepare().system_.elements[0].get_value('mu')
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
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.a = a

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'form'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'mu': self.mu,
            'omega': self.omega,
            'a': self.a,
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the real and the imaginary part."""
        return {
            'x': lambda s: (
                s.target.mu * s.target.x
                - s.target.omega * s.target.y
                + s.target.a * s.target.x * (s.target.x**2 + s.target.y**2)
            ),
            'y': lambda s: (
                s.target.omega * s.target.x
                + s.target.mu * s.target.y
                + s.target.a * s.target.y * (s.target.x**2 + s.target.y**2)
            ),
        }


class TakensBogdanovFormSystem(DynamicalSystem):
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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TakensBogdanovFormSystem
        >>> TakensBogdanovFormSystem(beta1=0.0).prepare().system_.elements[0].get_value('beta1')
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
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.beta1 = beta1
        self.beta2 = beta2
        self.sign = sign

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'form'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'beta1': self.beta1,
            'beta2': self.beta2,
            'sign': self.sign,
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the two coordinates."""
        return {
            'x': lambda s: s.target.y,
            'y': lambda s: (
                s.target.beta1 + s.target.beta2 * s.target.x + s.target.x**2 + s.target.sign * s.target.x * s.target.y
            ),
        }


class CuspFormSystem(DynamicalSystem):
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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CuspFormSystem
        >>> CuspFormSystem(beta1=0.2).prepare().system_.elements[0].get_value('beta1')
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
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.beta1 = beta1
        self.beta2 = beta2

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'form'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'beta1': self.beta1,
            'beta2': self.beta2,
        }

    def _get_equations(self) -> Equations:
        """Return the rate of the value."""
        return {
            'x': lambda s: s.target.beta1 + s.target.beta2 * s.target.x - s.target.x**3,
        }
