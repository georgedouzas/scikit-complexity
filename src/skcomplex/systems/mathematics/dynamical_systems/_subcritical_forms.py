"""Dynamical systems of the subcritical Hopf normal form.

The form has bistability and hysteresis.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class SubcriticalHopfFormInteraction(Interaction):
    """The laws of the subcritical Hopf normal form of the amplitude `z = x + iy`.

    The amplitude follows `z' = (alpha + iω) z + (1 + iq) |z|² z - (gamma + is) |z|⁴ z`.

    Args:
        label:
            The label of the interaction.

        target:
            The element whose real part `x` and imaginary part `y` the form moves.

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

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SubcriticalHopfFormInteraction
        >>> SubcriticalHopfFormInteraction('subcritical_hopf_form', target='form', alpha=0.1).alpha
        0.1
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'form',
        alpha: float = -0.1,
        omega: float = 1.0,
        q: float = 0.0,
        gamma: float = 1.0,
        s: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.alpha = alpha
        self.omega = omega
        self.q = q
        self.gamma = gamma
        self.s = s

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rates of the real and the imaginary part of the amplitude as the laws.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        radius = x**2 + y**2
        real = context.alpha + radius - context.gamma * radius**2
        imaginary = context.omega + context.q * radius - context.s * radius**2
        self.laws_: list[BaseLaw] = [
            RateLaw(x, real * x - imaginary * y),
            RateLaw(y, imaginary * x + real * y),
        ]


class SubcriticalHopfFormSystem(BaseSystem):
    """The subcritical Hopf normal form.

    A stable focus coexists with a stable limit cycle.

    The complex amplitude `z = x + iy` follows `z' = (alpha + iω) z + (1 + iq) |z|² z - (gamma + is) |z|⁴ z`, so its
    modulus follows `r' = alpha r + r³ - gamma r⁵`. The focus `z = 0` is stable for `alpha < 0`. The stable limit cycle
    of radius `r² = (1 + √(1 + 4 alpha gamma)) / (2 gamma)` exists for `alpha > -1 / (4 gamma)`, and an unstable cycle
    separates the two attractors while `alpha < 0`. A slow sweep of `alpha` up and down therefore jumps to the cycle at
    `alpha = 0` and falls back to the focus at `alpha = -1 / (4 gamma)`, a hysteresis loop. The system has the states
    `form.x` and `form.y`, and the parameters of the interaction `subcritical_hopf_form`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import SubcriticalHopfFormSystem
        >>> SubcriticalHopfFormSystem(alpha=0.1).prepare().system_.interactions[0].alpha
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.omega = omega
        self.q = q
        self.gamma = gamma
        self.s = s

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the normal form and its laws."""
        super()._prepare(context)
        form = SubcriticalHopfFormInteraction(
            'subcritical_hopf_form',
            target='form',
            alpha=self.alpha,
            omega=self.omega,
            q=self.q,
            gamma=self.gamma,
            s=self.s,
        )
        self.system_ = System(
            self.label,
            elements=[Element('form', values={'x': self.x, 'y': self.y})],
            interactions=[form],
        )
