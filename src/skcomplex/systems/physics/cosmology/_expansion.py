"""Expansion systems.

The scale factor of a homogeneous universe evolves under matter and dark energy.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class ScaleFactorElement(Element):
    """Homogeneous universe whose size is the scale factor `a`.

    The expansion rate is the derivative `a'`.

    Args:
        label:
            The label of the universe.

        a:
            The initial scale factor.

        values:
            Further values by name, such as the initial expansion rate `a'`, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import ScaleFactorElement
        >>> ScaleFactorElement('universe', a=2.0).a
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[float, RealSet.open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.a = a


class FriedmannAccelerationInteraction(Interaction):
    """Acceleration of the scale factor of a universe of matter and dark energy.

    The scale factor `a` of the target follows `a'' = -Ω_m H0² / (2 a²) + Ω_Λ H0² a`, the force of the potential
    `-Ω_m H0² / (2 a) - Ω_Λ H0² a² / 2`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the universe.

        H0:
            The Hubble constant.

        omega_matter:
            The density parameter of matter.

        omega_lambda:
            The density parameter of dark energy.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import FriedmannAccelerationInteraction
        >>> FriedmannAccelerationInteraction('acceleration', target='universe', H0=2.0).H0
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        H0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        omega_matter: Annotated[float, RealSet.closed_open(0, oo)] = 0.3,
        omega_lambda: float = 0.7,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.H0 = H0
        self.omega_matter = omega_matter
        self.omega_lambda = omega_lambda

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the scale factor.

        Args:
            context:
                The namespace of the universe, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            a = context.target.a
            acceleration = -context.omega_matter * context.H0**2 / (2 * a**2) + context.omega_lambda * context.H0**2 * a
            self.laws_ = [RateLaw(a.diff(context.t, 2), acceleration)]


class FriedmannExpansionSystem(BaseSystem):
    """The scale factor of a homogeneous universe of matter and dark energy.

    The system follows Friedmann and Lemaître.

    The acceleration equation `a² a'' = -4 pi G M / 3 + Lambda a³ / 3` is written with the density parameters of matter
    `Ω_m = 8 pi G M / (3 H0²)` and of the cosmological constant `Ω_Λ = Lambda / (3 H0²)` in units of the Hubble constant
    `H0` at `a = 1`, so `a² a'' = -Ω_m H0² / 2 + Ω_Λ H0² a³`. The initial expansion rate follows from the Friedmann
    equation `a'² = H0² (Ω_m / a + Ω_Λ a² + 1 - Ω_m - Ω_Λ)`, which stays a first integral of the motion. Matter
    decelerates the expansion and dark energy accelerates it once `a` exceeds `(Ω_m / (2 Ω_Λ))^(1/3)`. The system has
    the scale factor `universe.a` and its rate `universe.a'`.

    Args:
        a:
            The initial scale factor.

        H0:
            The Hubble constant, the expansion rate at `a = 1`.

        omega_matter:
            The density parameter of matter.

        omega_lambda:
            The density parameter of the cosmological constant.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The universe does not expand at the initial scale factor.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import FriedmannExpansionSystem
        >>> system = FriedmannExpansionSystem(omega_matter=1.0, omega_lambda=0.0).prepare()
        >>> float(system.simulate(until=1.0).get_values("universe.a'")[0])
        1.0
    """

    field = 'physics/cosmology'
    topics = ('cosmology', 'ode', 'second order', 'gravity', 'dark energy')
    references = ('Friedmann 1922', 'Lemaître 1927', 'Satz 2022')
    until = 2.0
    dt = 0.02

    def __init__(
        self,
        a: float = 1.0,
        *,
        H0: float = 1.0,
        omega_matter: float = 0.3,
        omega_lambda: float = 0.7,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.a = a
        self.H0 = H0
        self.omega_matter = omega_matter
        self.omega_lambda = omega_lambda

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the scale factor of a homogeneous universe of matter and dark energy.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        square = self.H0**2 * (
            self.omega_matter / self.a + self.omega_lambda * self.a**2 + 1 - self.omega_matter - self.omega_lambda
        )
        if square < 0:
            error_msg = (
                f'The Friedmann equation gives a negative squared expansion rate {square} at a = {self.a}. Choose a, '
                'omega_matter, and omega_lambda for which it is positive.'
            )
            raise ValidationError(error_msg)
        universe = ScaleFactorElement('universe', a=self.a, values={"a'": math.sqrt(square)})
        acceleration = FriedmannAccelerationInteraction(
            'acceleration',
            target='universe',
            H0=self.H0,
            omega_matter=self.omega_matter,
            omega_lambda=self.omega_lambda,
        )
        self.system_ = System(self.label, elements=[universe], interactions=[acceleration])
