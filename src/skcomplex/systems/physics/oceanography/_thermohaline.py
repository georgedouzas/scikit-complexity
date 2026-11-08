"""Thermohaline circulation between ocean boxes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import abs_symbolic

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class StommelBoxesElement(Element):
    """Ocean of Stommel's two boxes.

    The ocean holds the differences of temperature and salinity between the equator and the pole.

    Both differences relax to their forcing and are mixed by a flow proportional to the density difference
    `|T - S|`: `dT/dt = eta1 - T (1 + |T - S|)` and `dS/dt = eta2 - S (eta3 + |T - S|)`.

    Args:
        label:
            The label of the ocean.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        T:
            The initial temperature difference.

        S:
            The initial salinity difference.

        eta1:
            The thermal forcing.

        eta2:
            The freshwater forcing.

        eta3:
            The ratio of the relaxation times of temperature and salinity.

    Examples:
        >>> from skcomplex.systems.physics.oceanography import StommelBoxesElement
        >>> StommelBoxesElement('ocean').eta2
        1.2
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec()] = 3.0,
        S: Annotated[float, StateSpec()] = 0.0,
        eta1: Annotated[float, ParameterSpec()] = 3.0,
        eta2: Annotated[float, ParameterSpec()] = 1.2,
        eta3: Annotated[float, ParameterSpec()] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.S = S
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation of both differences to their forcing and their mixing.

        The circulation mixes the differences.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the temperature and salinity differences.
        """
        return [
            RateLaw(s.T, s.eta1 - s.T * (1 + abs_symbolic(s.T - s.S))),
            RateLaw(s.S, s.eta2 - s.S * (s.eta3 + abs_symbolic(s.T - s.S))),
        ]


class StommelSystem(BaseSystem):
    """Stommel's two-box thermohaline circulation.

    The circulation is bistable between a thermal and a haline mode.

    The dimensionless differences of temperature `T` and salinity `S` between an equatorial and a polar box relax to
    their forcing and are mixed by a flow proportional to the density difference `|T - S|`: `dT/dt = eta1 - T (1 + |T -
    S|)` and `dS/dt = eta2 - S (eta3 + |T - S|)`. For the default parameters a strong circulation driven by temperature
    and a weak one driven by salinity are both stable. The system has the differences `ocean.T` and `ocean.S`.

    Args:
        T:
            The initial temperature difference.

        S:
            The initial salinity difference.

        eta1:
            The thermal forcing.

        eta2:
            The freshwater forcing.

        eta3:
            The ratio of the relaxation times of temperature and salinity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.oceanography import StommelSystem
        >>> StommelSystem(eta2=1.0).prepare().system_.elements[0].eta2
        1.0
    """

    field = 'physics/oceanography'
    topics = ('ode', 'bistability', 'thermohaline circulation', 'tipping')
    references = ('Stommel 1961', 'Cessi 1994', 'Ghil and Lucarini 2020')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        T: float = 3.0,
        S: float = 0.0,
        *,
        eta1: float = 3.0,
        eta2: float = 1.2,
        eta3: float = 0.3,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.T = T
        self.S = S
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

    def _build_system(self) -> System:
        """Return Stommel's two-box thermohaline circulation.

        The circulation is bistable between a thermal and a haline mode.
        """
        return System(
            self.label,
            elements=[StommelBoxesElement('ocean', T=self.T, S=self.S, eta1=self.eta1, eta2=self.eta2, eta3=self.eta3)],
        )
