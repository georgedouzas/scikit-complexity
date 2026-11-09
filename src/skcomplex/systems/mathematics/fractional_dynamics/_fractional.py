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
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.flows import LorenzElement


class OscillatorElement(Element):
    """Damped linear oscillator `x' = v`, `v' = -omega^2 x - 2 gamma v`.

    The oscillator has two first-order states.

    Its fractional subclasses replace both derivatives by Caputo derivatives of one order.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        v:
            The initial derivative of the displacement.

        omega:
            The angular frequency.

        gamma:
            The damping rate.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import OscillatorElement
        >>> OscillatorElement('oscillator', omega=2.0).omega
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        v: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
        gamma: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.omega = omega
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the displacement and its derivative.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.v), RateLaw(s.v, -(s.omega**2) * s.x - 2 * s.gamma * s.v)]


class RelaxationElement(Element):
    """Linear relaxation `x' = -lam x`.

    Its fractional subclasses replace the derivative by a Caputo derivative.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        lam:
            The relaxation rate.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import RelaxationElement
        >>> RelaxationElement('relaxation', lam=2.0).lam
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        lam: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the decay of the value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, -s.lam * s.x)]


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
def relaxation_type(order: float) -> type[RelaxationElement]:
    """Return the element type of fractional relaxation of an order.

    The relaxation is `D^alpha x = -lambda x`.

    Args:
        order:
            The order `alpha` of the Caputo derivative of `x`.

    Returns:
        element_type:
            The subclass of `RelaxationElement` whose state has the order, the same for equal orders.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import relaxation_type
        >>> relaxation_type(0.5).__name__
        'FractionalRelaxation_0p5'
    """

    class FractionalElement(RelaxationElement):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[float, StateSpec(order=order)] = 1.0,
            lam: float = 1.0,
        ) -> None:
            super().__init__(label, visual=visual, x=x, lam=lam)

    return _published(FractionalElement, _type_name('FractionalRelaxation', order))


@cache
def oscillator_type(order: float) -> type[OscillatorElement]:
    """Return the element type of the fractional oscillator of an order.

    The oscillator is `D^alpha x = v`, `D^alpha v = -omega^2 x - 2 gamma v`.

    Args:
        order:
            The order `alpha` of the Caputo derivatives of `x` and `v`.

    Returns:
        element_type:
            The subclass of `OscillatorElement` whose states have the order, the same for equal orders.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import oscillator_type
        >>> oscillator_type(0.8).__name__
        'FractionalOscillator_0p8'
    """

    class FractionalElement(OscillatorElement):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[float, StateSpec(order=order)] = 1.0,
            v: Annotated[float, StateSpec(order=order)] = 0.0,
            omega: float = 1.0,
            gamma: float = 0.0,
        ) -> None:
            super().__init__(label, visual=visual, x=x, v=v, omega=omega, gamma=gamma)

    return _published(FractionalElement, _type_name('FractionalOscillator', order))


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


@cache
def lorenz_type(order: float) -> type[LorenzElement]:
    """Return the element type of the fractional Lorenz flow of an order.

    The flow is the Lorenz equations with Caputo derivatives of the order.

    Args:
        order:
            The order `q` of the Caputo derivatives of `x`, `y`, and `z`.

    Returns:
        element_type:
            The subclass of `LorenzElement` whose states have the order, the same for equal orders.

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import lorenz_type
        >>> lorenz_type(0.99).__name__
        'FractionalLorenz_0p99'
    """

    class FractionalElement(LorenzElement):
        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[float, StateSpec(order=order)] = 1.0,
            y: Annotated[float, StateSpec(order=order)] = 1.0,
            z: Annotated[float, StateSpec(order=order)] = 1.0,
            sigma: float = 10.0,
            rho: float = 28.0,
            beta: float = 8 / 3,
        ) -> None:
            super().__init__(label, visual=visual, x=x, y=y, z=z, sigma=sigma, rho=rho, beta=beta)

    return _published(FractionalElement, _type_name('FractionalLorenz', order))


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

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalRelaxationSystem
        >>> FractionalRelaxationSystem(order=0.5).prepare().system_.elements[0].lam
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
        super().__init__(label)
        self.x = x
        self.order = order
        self.lam = lam

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[relaxation_type(float(self.order))('relaxation', x=self.x, lam=self.lam)],
        )


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

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalOscillatorSystem
        >>> FractionalOscillatorSystem(order=0.9).prepare().system_.elements[0].omega
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
        super().__init__(label)
        self.x = x
        self.v = v
        self.order = order
        self.omega = omega
        self.gamma = gamma

    def _build_system(self) -> CompositeSystem:
        """Return the fractional damped oscillator.

        The oscillator is `D^alpha x = v`, `D^alpha v = -omega^2 x - 2 gamma v`.
        """
        return CompositeSystem(
            self.label,
            elements=[
                oscillator_type(float(self.order))(
                    'oscillator',
                    x=self.x,
                    v=self.v,
                    omega=self.omega,
                    gamma=self.gamma,
                ),
            ],
        )


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

    Examples:
        >>> from skcomplex.systems.mathematics.fractional_dynamics import FractionalLorenzSystem
        >>> FractionalLorenzSystem(rho=20.0).prepare().system_.elements[0].rho
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
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.order = order
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _build_system(self) -> CompositeSystem:
        """Return the fractional Lorenz flow of Grigorenko and Grigorenko.

        The flow is chaotic for orders close enough to 1.
        """
        flow = lorenz_type(float(self.order))(
            'lorenz',
            x=self.x,
            y=self.y,
            z=self.z,
            sigma=self.sigma,
            rho=self.rho,
            beta=self.beta,
        )
        return CompositeSystem(self.label, elements=[flow])
