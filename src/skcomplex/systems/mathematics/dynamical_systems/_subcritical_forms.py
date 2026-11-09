"""Dynamical systems of the subcritical Hopf normal form.

The form has bistability and hysteresis.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class SubcriticalHopfFormSystem(DynamicalSystem):
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
        >>> SubcriticalHopfFormSystem(alpha=0.1).prepare().system_.elements[0].get_value('alpha')
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
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.omega = omega
        self.q = q
        self.gamma = gamma
        self.s = s

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'form'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'x': DeclaredVariableSpec(self.x, StateSpec()),
            'y': DeclaredVariableSpec(self.y, StateSpec()),
            'alpha': DeclaredVariableSpec(self.alpha, ParameterSpec()),
            'omega': DeclaredVariableSpec(self.omega, ParameterSpec()),
            'q': DeclaredVariableSpec(self.q, ParameterSpec()),
            'gamma': DeclaredVariableSpec(self.gamma, ParameterSpec(domain=RealSet.open(0, oo))),
            's': DeclaredVariableSpec(self.s, ParameterSpec()),
        }

    def _get_equations(self) -> Equations:
        """Return the rates of the real and the imaginary part of the amplitude."""
        return {'x': lambda s: self._get_rates(s.target)[0], 'y': lambda s: self._get_rates(s.target)[1]}

    @staticmethod
    def _get_rates(form: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the rates of the real and the imaginary part of the amplitude."""
        radius = form.x**2 + form.y**2
        real = form.alpha + radius - form.gamma * radius**2
        imaginary = form.omega + form.q * radius - form.s * radius**2
        return real * form.x - imaginary * form.y, imaginary * form.x + real * form.y
