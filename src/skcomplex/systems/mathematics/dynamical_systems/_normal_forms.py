"""Dynamical systems of the normal forms of bifurcations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CuspElement(Element):
    """Normal form `x' = beta1 + beta2 x - x^3` of the cusp bifurcation.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        beta1:
            The first unfolding parameter, the asymmetry.

        beta2:
            The second unfolding parameter, the splitting.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CuspElement
        >>> CuspElement('form', beta2=2.0).beta2
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.5,
        beta1: Annotated[object, ParameterSpec()] = 0.0,
        beta2: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.beta1 = beta1
        self.beta2 = beta2

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.beta1 + s.beta2 * s.x - s.x**3)]


class HopfElement(Element):
    """Normal form of the Hopf bifurcation in Cartesian coordinates.

    The radial law is `r' = mu r + a r^3`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial real part.

        y:
            The initial imaginary part.

        mu:
            The bifurcation parameter.

        omega:
            The angular frequency.

        a:
            The cubic coefficient, negative for the supercritical bifurcation.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HopfElement
        >>> HopfElement('form', mu=0.5).mu
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        mu: Annotated[object, ParameterSpec()] = 0.25,
        omega: Annotated[object, ParameterSpec()] = 1.0,
        a: Annotated[object, ParameterSpec()] = -1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.a = a

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the real and the imaginary part.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            RateLaw(s.x, s.mu * s.x - s.omega * s.y + s.a * s.x * (s.x**2 + s.y**2)),
            RateLaw(s.y, s.omega * s.x + s.mu * s.y + s.a * s.y * (s.x**2 + s.y**2)),
        ]


class SaddleNodeElement(Element):
    """Normal form `x' = mu - x^2` of the saddle-node bifurcation.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value.

        mu:
            The bifurcation parameter.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SaddleNodeElement
        >>> SaddleNodeElement('form', mu=-1.0).mu
        -1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        mu: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.mu = mu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rate of the value.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.mu - s.x**2)]


class TakensBogdanovElement(Element):
    """Normal form of the Takens-Bogdanov bifurcation.

    The form is `x' = y`, `y' = beta1 + beta2 x + x^2 + sign x y`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        beta1:
            The first unfolding parameter.

        beta2:
            The second unfolding parameter.

        sign:
            The sign of the term `x y`, 1 or -1.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TakensBogdanovElement
        >>> TakensBogdanovElement('form', beta1=0.0).beta1
        0.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.2,
        y: Annotated[object, StateSpec()] = 0.0,
        beta1: Annotated[object, ParameterSpec()] = 0.1,
        beta2: Annotated[object, ParameterSpec()] = -1.0,
        sign: Annotated[object, ParameterSpec()] = -1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.beta1 = beta1
        self.beta2 = beta2
        self.sign = sign

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the two coordinates.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [RateLaw(s.x, s.y), RateLaw(s.y, s.beta1 + s.beta2 * s.x + s.x**2 + s.sign * s.x * s.y)]


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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SaddleNodeFormSystem
        >>> SaddleNodeFormSystem(mu=0.5).prepare().system_.elements[0].mu
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
        super().__init__(label)
        self.x = x
        self.mu = mu

    def _build_system(self) -> CompositeSystem:
        """Return the saddle-node normal form `x' = μ - x²`.

        Its two equilibria `±√μ` meet and vanish at μ = 0.
        """
        return CompositeSystem(self.label, elements=[SaddleNodeElement('form', x=self.x, mu=self.mu)])


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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import HopfFormSystem
        >>> HopfFormSystem(mu=-0.1).prepare().system_.elements[0].mu
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
        super().__init__(label)
        self.x = x
        self.y = y
        self.mu = mu
        self.omega = omega
        self.a = a

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[HopfElement('form', x=self.x, y=self.y, mu=self.mu, omega=self.omega, a=self.a)],
        )


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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import TakensBogdanovFormSystem
        >>> TakensBogdanovFormSystem(beta1=0.0).prepare().system_.elements[0].beta1
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
        super().__init__(label)
        self.x = x
        self.y = y
        self.beta1 = beta1
        self.beta2 = beta2
        self.sign = sign

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[
                TakensBogdanovElement('form', x=self.x, y=self.y, beta1=self.beta1, beta2=self.beta2, sign=self.sign),
            ],
        )


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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CuspFormSystem
        >>> CuspFormSystem(beta1=0.2).prepare().system_.elements[0].beta1
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
        super().__init__(label)
        self.x = x
        self.beta1 = beta1
        self.beta2 = beta2

    def _build_system(self) -> CompositeSystem:
        """Return the cusp normal form `x' = β1 + β2 x - x³`.

        The form is bistable inside the cusp `27 β1² < 4 β2³`.
        """
        return CompositeSystem(self.label, elements=[CuspElement('form', x=self.x, beta1=self.beta1, beta2=self.beta2)])
