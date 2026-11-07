"""Dynamical systems: normal forms of bifurcations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


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
    """Normal form of the Hopf bifurcation in Cartesian coordinates, with the radial law `r' = mu r + a r^3`.

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
    """Normal form `x' = y`, `y' = beta1 + beta2 x + x^2 + sign x y` of the Takens-Bogdanov bifurcation.

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


@register_system(
    field='mathematics/dynamical_systems',
    tags=('bifurcation', 'normal form', 'ode'),
    references=('Gros 2015', 'Dilão 2023'),
    defaults={'until': 10.0, 'dt': 0.05},
)
def saddle_node_form(x: float = 0.0, *, mu: float = 1.0) -> System:
    """Return the saddle-node normal form `x' = μ - x²`, whose two equilibria `±√μ` meet and vanish at μ = 0.

    Args:
        x:
            The initial value.

        mu:
            The bifurcation parameter.

    Returns:
        system:
            The system, with the state `form.x`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import saddle_node_form
        >>> saddle_node_form(mu=0.5).elements[0].mu
        0.5
    """
    return System(elements=[SaddleNodeElement('form', x=x, mu=mu)])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('bifurcation', 'normal form', 'ode', 'limit cycle'),
    references=('Gros 2015', 'Dilão 2023'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def hopf_form(x: float = 0.1, y: float = 0.0, *, mu: float = 0.25, omega: float = 1.0, a: float = -1.0) -> System:
    """Return the Hopf normal form, whose focus gives birth to a limit cycle of radius `√(-μ / a)` as μ crosses 0.

    `x' = μx - ωy + a x (x² + y²)` and `y' = ωx + μy + a y (x² + y²)`, supercritical for `a < 0`.

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

    Returns:
        system:
            The system, with the states `form.x` and `form.y`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import hopf_form
        >>> hopf_form(mu=-0.1).elements[0].mu
        -0.1
    """
    return System(elements=[HopfElement('form', x=x, y=y, mu=mu, omega=omega, a=a)])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('bifurcation', 'normal form', 'ode', 'codimension two'),
    references=('Gros 2015', 'Dilão 2023'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def takens_bogdanov_form(
    x: float = 0.2,
    y: float = 0.0,
    *,
    beta1: float = 0.1,
    beta2: float = -1.0,
    sign: float = -1.0,
) -> System:
    """Return the Takens-Bogdanov normal form `x' = y`, `y' = β1 + β2 x + x² + s x y`, where a saddle-node meets a Hopf.

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

    Returns:
        system:
            The system, with the states `form.x` and `form.y`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import takens_bogdanov_form
        >>> takens_bogdanov_form(beta1=0.0).elements[0].beta1
        0.0
    """
    return System(elements=[TakensBogdanovElement('form', x=x, y=y, beta1=beta1, beta2=beta2, sign=sign)])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('bifurcation', 'normal form', 'ode', 'bistability', 'hysteresis'),
    references=('Gros 2015', 'Dilão 2023'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def cusp_form(x: float = 0.5, *, beta1: float = 0.0, beta2: float = 1.0) -> System:
    """Return the cusp normal form `x' = β1 + β2 x - x³`, bistable inside the cusp `27 β1² < 4 β2³`.

    Args:
        x:
            The initial value.

        beta1:
            The asymmetry.

        beta2:
            The splitting parameter.

    Returns:
        system:
            The system, with the state `form.x`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import cusp_form
        >>> cusp_form(beta1=0.2).elements[0].beta1
        0.2
    """
    return System(elements=[CuspElement('form', x=x, beta1=beta1, beta2=beta2)])
