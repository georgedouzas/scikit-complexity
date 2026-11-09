"""Expansion systems.

The scale factor of a homogeneous universe evolves under matter and dark energy.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import GradientInteraction
from ....core.components.systems import BaseSystem, System
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ScaleFactorElement(Element):
    """Homogeneous universe whose size is the scale factor `a`.

    The scale factor is a state of order 2 with the expansion rate `a_d1`.

    Args:
        label:
            The label of the universe.

        visual:
            How the universe is drawn, or `None` for none.

        a:
            The initial scale factor.

        a_d1:
            The initial rate of change of the scale factor.

        H0:
            The Hubble constant.

        omega_matter:
            The density parameter of matter.

        omega_lambda:
            The density parameter of dark energy.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import ScaleFactorElement
        >>> ScaleFactorElement('universe', H0=2.0).H0
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[float, StateSpec(domain=RealSet.open(0, oo), order=2)] = 1.0,
        a_d1: float = 0.0,
        H0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        omega_matter: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        omega_lambda: Annotated[float, ParameterSpec()] = 0.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.a_d1 = a_d1
        self.H0 = H0
        self.omega_matter = omega_matter
        self.omega_lambda = omega_lambda


class FriedmannExpansionSystem(BaseSystem):
    """The scale factor of a homogeneous universe of matter and dark energy.

    The system follows Friedmann and Lemaître.

    The acceleration equation `a² a'' = -4 pi G M / 3 + Lambda a³ / 3` is written with the density parameters of matter
    `Ω_m = 8 pi G M / (3 H0²)` and of the cosmological constant `Ω_Λ = Lambda / (3 H0²)` in units of the Hubble constant
    `H0` at `a = 1`, so `a² a'' = -Ω_m H0² / 2 + Ω_Λ H0² a³`. The initial expansion rate follows from the Friedmann
    equation `a'² = H0² (Ω_m / a + Ω_Λ a² + 1 - Ω_m - Ω_Λ)`, which stays a first integral of the motion. Matter
    decelerates the expansion and dark energy accelerates it once `a` exceeds `(Ω_m / (2 Ω_Λ))^(1/3)`. The system has
    the scale factor `universe.a` of order 2 and its rate `universe.a_d1`.

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

    Raises:
        ValidationError:
            The universe does not expand at the initial scale factor.

    Examples:
        >>> from skcomplex.systems.physics.cosmology import FriedmannExpansionSystem
        >>> FriedmannExpansionSystem(omega_matter=1.0, omega_lambda=0.0).prepare().system_.elements[0].a_d1
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
    ) -> None:
        super().__init__(label)
        self.a = a
        self.H0 = H0
        self.omega_matter = omega_matter
        self.omega_lambda = omega_lambda

    @staticmethod
    def _get_expansion_potential(s: SimpleNamespace) -> Expression:
        """Return the potential of the acceleration equation.

        The potential is `-omega_matter H0² / (2 a) - omega_lambda H0² a² / 2`.
        """
        universe = s.target
        return (
            -universe.omega_matter * universe.H0**2 / (2 * universe.a)
            - universe.omega_lambda * universe.H0**2 * universe.a**2 / 2
        )

    def _build_system(self) -> System:
        """Return the scale factor of a homogeneous universe of matter and dark energy.

        The system follows Friedmann and Lemaître.
        """
        square = self.H0**2 * (
            self.omega_matter / self.a + self.omega_lambda * self.a**2 + 1 - self.omega_matter - self.omega_lambda
        )
        if square < 0:
            error_msg = (
                f'The Friedmann equation gives a negative squared expansion rate {square} at a = {self.a}. Choose a, '
                'omega_matter, and omega_lambda for which it is positive.'
            )
            raise ValidationError(error_msg)
        universe = ScaleFactorElement(
            'universe',
            a=self.a,
            a_d1=math.sqrt(square),
            H0=self.H0,
            omega_matter=self.omega_matter,
            omega_lambda=self.omega_lambda,
        )
        acceleration = GradientInteraction(
            'acceleration',
            target='universe',
            states='a',
            potential=self._get_expansion_potential,
        )
        return System(self.label, elements=[universe], interactions=[acceleration])
