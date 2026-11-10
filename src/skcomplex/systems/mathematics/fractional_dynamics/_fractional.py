"""Fractional dynamics systems.

The module covers relaxation, oscillators, and chaotic flows with Caputo derivatives of states.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import cache
from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class VanDerPolElement(Element):
    """Van der Pol oscillator `x' = y`, `y' = mu (1 - x^2) y - x`.

    The oscillator has two first-order states.

    Its fractional subclasses replace both derivatives by Caputo derivatives of one order.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        y:
            The initial derivative of the displacement.

        mu:
            The strength of the nonlinear damping.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import VanDerPolElement
        >>> VanDerPolElement('oscillator', mu=2.0).mu
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 2.0,
        y: Annotated[float, StateSpec()] = 0.0,
        mu: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.mu = mu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the displacement and its derivative.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.y), RateLaw(s.y, s.mu * (1 - s.x**2) * s.y - s.x)]


def _published[ElementT: Element](element_type: type[ElementT], name: str) -> type[ElementT]:
    """Return an element type after naming it and binding it to its name in the module.

    The binding makes the type pickle by reference.
    """
    element_type.__name__ = element_type.__qualname__ = name
    globals()[name] = element_type
    return element_type


def _type_name(prefix: str, order: float) -> str:
    """Return the name of the element type of an order.

    The name holds the order with `p` for the decimal point.
    """
    return f'{prefix}_{repr(order).replace(".", "p")}'


@cache
def van_der_pol_type(order: float) -> type[VanDerPolElement]:
    """Return the element type of the fractional Van der Pol system of an order.

    The system is `D^alpha x = y`, `D^alpha y = mu (1 - x^2) y - x`.

    Args:
        order:
            The order `alpha` of the Caputo derivatives of `x` and `y`.

    Returns:
        element_type:
            The subclass of `VanDerPolElement` whose states have the order, the same for equal orders.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import van_der_pol_type
        >>> van_der_pol_type(0.9).__name__
        'FractionalVanDerPol_0p9'
    """

    class FractionalElement(VanDerPolElement):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[float, StateSpec(order=order)] = 2.0,
            y: Annotated[float, StateSpec(order=order)] = 0.0,
            mu: float = 1.0,
        ) -> None:
            super().__init__(label, visual=visual, x=x, y=y, mu=mu)

    return _published(FractionalElement, _type_name('FractionalVanDerPol', order))


class FractionalRelaxationSystem(DynamicalSystem):
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

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalRelaxationSystem
        >>> FractionalRelaxationSystem(order=0.5).prepare().system_.elements[0].get_value('lam')
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
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.order = order
        self.lam = lam

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'relaxation'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states, with the order of their Caputo derivatives, and the parameters."""
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec(order=float(self.order))),
            'lam': DeclaredVariableSpec(self.lam, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the parts of the system, built from its model parameters."""
        return {
            'x': lambda s: -s.target.lam * s.target.x,
        }


class FractionalOscillatorSystem(DynamicalSystem):
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

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalOscillatorSystem
        >>> FractionalOscillatorSystem(order=0.9).prepare().system_.elements[0].get_value('omega')
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
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.v = v
        self.order = order
        self.omega = omega
        self.gamma = gamma

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'oscillator'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states, with the order of their Caputo derivatives, and the parameters."""
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec(order=float(self.order))),
            'v': DeclaredVariableSpec(self.v, StateSpec(order=float(self.order))),
            'omega': DeclaredVariableSpec(self.omega, ParameterSpec()),
            'gamma': DeclaredVariableSpec(self.gamma, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the fractional damped oscillator."""
        return {
            'x': lambda s: s.target.v,
            'v': lambda s: -(s.target.omega**2) * s.target.x - 2 * s.target.gamma * s.target.v,
        }


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

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalVanDerPolSystem
        >>> FractionalVanDerPolSystem(mu=2.0).prepare().system_.elements[0].mu
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
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.order = order
        self.mu = mu

    def _build_system(self) -> CompositeSystem:
        """Return the fractional Van der Pol oscillator.

        The oscillator is `D^alpha x = y`, `D^alpha y = mu (1 - x^2) y - x`.
        """
        return CompositeSystem(
            self.label,
            elements=[van_der_pol_type(float(self.order))('oscillator', x=self.x, y=self.y, mu=self.mu)],
        )


class FractionalLorenzSystem(DynamicalSystem):
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

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalLorenzSystem
        >>> FractionalLorenzSystem(rho=20.0).prepare().system_.elements[0].get_value('rho')
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
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.z = z
        self.order = order
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'lorenz'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states, with the order of their Caputo derivatives, and the parameters."""
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec(order=float(self.order))),
            'y': DeclaredVariableSpec(self.y, StateSpec(order=float(self.order))),
            'z': DeclaredVariableSpec(self.z, StateSpec(order=float(self.order))),
            'sigma': DeclaredVariableSpec(self.sigma, ParameterSpec()),
            'rho': DeclaredVariableSpec(self.rho, ParameterSpec()),
            'beta': DeclaredVariableSpec(self.beta, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the fractional Lorenz flow of Grigorenko and Grigorenko."""
        return {
            'x': lambda s: s.target.sigma * (s.target.y - s.target.x),
            'y': lambda s: s.target.x * (s.target.rho - s.target.z) - s.target.y,
            'z': lambda s: s.target.x * s.target.y - s.target.beta * s.target.z,
        }
