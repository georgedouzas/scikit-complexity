"""Dynamical systems of the subcritical Hopf normal form.

The form has bistability and hysteresis.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class SubcriticalHopfElement(Element):
    """Subcritical Hopf normal form.

    The form is `z' = (alpha + i omega) z + (1 + i q) |z|^2 z - (gamma + i s) |z|^4 z`.

    The complex amplitude is `z = x + i y`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial real part of `z`.

        y:
            The initial imaginary part of `z`.

        alpha:
            The bifurcation parameter, the linear growth rate.

        omega:
            The angular frequency at small amplitude.

        q:
            The nonlinear frequency shift of the cubic term.

        gamma:
            The stabilizing quintic coefficient, positive.

        s:
            The nonlinear frequency shift of the quintic term.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SubcriticalHopfElement
        >>> SubcriticalHopfElement('form', alpha=0.1).alpha
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.1,
        y: Annotated[object, StateSpec()] = 0.0,
        alpha: Annotated[object, ParameterSpec()] = -0.1,
        omega: Annotated[object, ParameterSpec()] = 1.0,
        q: Annotated[object, ParameterSpec()] = 0.0,
        gamma: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        s: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.omega = omega
        self.q = q
        self.gamma = gamma
        self.s = s

    def _get_rates(self, s: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the rates of the real and the imaginary part of the amplitude."""
        radius = s.x**2 + s.y**2
        real = s.alpha + radius - s.gamma * radius**2
        imaginary = s.omega + s.q * radius - s.s * radius**2
        return real * s.x - imaginary * s.y, imaginary * s.x + real * s.y

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the real and the imaginary part of the amplitude.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        real, imaginary = self._get_rates(s)
        return [RateLaw(s.x, real), RateLaw(s.y, imaginary)]


class SubcriticalHopfFormSystem(BaseSystem):
    """The subcritical Hopf normal form.

    A stable focus coexists with a stable limit cycle.

    The complex amplitude `z = x + iy` follows `z' = (alpha + iω) z + (1 + iq) |z|² z - (gamma + is) |z|⁴ z`, so its
    modulus follows `r' = alpha r + r³ - gamma r⁵`. The focus `z = 0` is stable for `alpha < 0`. The stable limit cycle
    of radius `r² = (1 + √(1 + 4 alpha gamma)) / (2 gamma)` exists for `alpha > -1 / (4 gamma)`, and an unstable cycle
    separates the two attractors while `alpha < 0`. A slow sweep of `alpha` up and down therefore jumps to the cycle at
    `alpha = 0` and falls back to the focus at `alpha = -1 / (4 gamma)`, a hysteresis loop. The system has the states
    `form.x` and `form.y`.

    Args:
        x:
            The initial real part of `z`.

        y:
            The initial imaginary part of `z`.

        alpha:
            The bifurcation parameter `alpha`, the linear growth rate.

        omega:
            The angular frequency `ω` at small amplitude.

        q:
            The nonlinear frequency shift `q` of the cubic term.

        gamma:
            The stabilizing quintic coefficient `gamma`, positive.

        s:
            The nonlinear frequency shift `s` of the quintic term.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SubcriticalHopfFormSystem
        >>> SubcriticalHopfFormSystem(alpha=0.1).prepare().system_.elements[0].alpha
        0.1
    """

    field = 'mathematics/dynamical_systems'
    topics = ('bifurcation', 'normal form', 'ode', 'limit cycle', 'bistability', 'hysteresis')
    references = ('Bertin 2021',)
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        x: float = 0.1,
        y: float = 0.0,
        *,
        alpha: float = -0.1,
        omega: float = 1.0,
        q: float = 0.0,
        gamma: float = 1.0,
        s: float = 0.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.omega = omega
        self.q = q
        self.gamma = gamma
        self.s = s

    def _build_system(self) -> CompositeSystem:
        """Return the subcritical Hopf normal form.

        A stable focus coexists with a stable limit cycle.
        """
        return CompositeSystem(
            self.label,
            elements=[
                SubcriticalHopfElement(
                    'form',
                    x=self.x,
                    y=self.y,
                    alpha=self.alpha,
                    omega=self.omega,
                    q=self.q,
                    gamma=self.gamma,
                    s=self.s,
                ),
            ],
        )
