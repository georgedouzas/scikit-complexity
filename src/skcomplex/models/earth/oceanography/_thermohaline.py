"""Thermohaline circulation between ocean boxes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import abs_symbolic

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.system import System
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class StommelBoxesElement(Element):
    """Ocean of Stommel's two boxes, the differences of temperature and salinity between the equator and the pole.

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
        >>> from skcomplex.models.earth.oceanography import StommelBoxesElement
        >>> StommelBoxesElement('ocean').eta2
        1.2
    """

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
        """Return the relaxation of both differences to their forcing and their mixing by the circulation.

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


@register_model(
    field='earth/oceanography',
    tags=('ode', 'bistability', 'thermohaline circulation', 'tipping'),
    references=('Stommel 1961', 'Cessi 1994', 'Ghil and Lucarini 2020'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def stommel(
    T: float = 3.0,
    S: float = 0.0,
    *,
    eta1: float = 3.0,
    eta2: float = 1.2,
    eta3: float = 0.3,
) -> System:
    """Return Stommel's two-box thermohaline circulation, bistable between a thermal and a haline mode.

    The dimensionless differences of temperature `T` and salinity `S` between an equatorial and a polar box relax to
    their forcing and are mixed by a flow proportional to the density difference `|T - S|`:
    `dT/dt = eta1 - T (1 + |T - S|)` and `dS/dt = eta2 - S (eta3 + |T - S|)`. For the default parameters a strong
    circulation driven by temperature and a weak one driven by salinity are both stable.

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

    Returns:
        system:
            The model, with the differences `ocean.T` and `ocean.S`.

    Examples:
        >>> from skcomplex.models.earth.oceanography import stommel
        >>> stommel(eta2=1.0).elements[0].eta2
        1.0
    """
    return System(elements=[StommelBoxesElement('ocean', T=T, S=S, eta1=eta1, eta2=eta2, eta3=eta3)])
