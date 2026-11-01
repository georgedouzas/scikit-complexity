"""Ocean of Rothman's excitable carbon cycle, its carbonate ion concentration and dissolved inorganic carbon."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import hill
from ....specs import ParameterSpec, StateSpec, VisualSpec


class CarbonCycleElement(Element):
    """Ocean of Rothman's excitable carbon cycle, its carbonate ion concentration and dissolved inorganic carbon.

    The carbonate ion concentration `c` and the dissolved inorganic carbon `w`, in μmol kg⁻¹, follow
    `dc/dt = f(c) (mu (1 - b s(c, c_p) - theta (1 - s(c, c_x)) - nu) + w - w0)` and
    `dw/dt = mu (1 - b s(c, c_p) + theta (1 - s(c, c_x)) + nu) - w + w0`, with the Hill functions
    `s(c, c*) = c^gamma / (c^gamma + c*^gamma)` of burial and respiration and the buffer
    `f(c) = f0 c^beta / (c^beta + c_f^beta)`.

    Args:
        label:
            The label of the ocean.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        c:
            The initial carbonate ion concentration.

        w:
            The initial dissolved inorganic carbon.

        mu:
            The characteristic rate of the carbon fluxes.

        b:
            The maximal burial of organic carbon.

        theta:
            The maximal respiration of organic carbon.

        c_x:
            The crossover carbonate concentration of respiration, which sets the excitability.

        c_p:
            The crossover carbonate concentration of burial.

        nu:
            The rate of external carbon injection.

        w0:
            The input of dissolved inorganic carbon by weathering and volcanism.

        gamma:
            The Hill coefficient of burial and respiration.

        f0:
            The maximal buffer of the carbonate ion concentration.

        c_f:
            The carbonate concentration where the buffer is half its maximum.

        beta:
            The Hill coefficient of the buffer.

    Examples:
        >>> from skcomplex.models.earth.climate import CarbonCycleElement
        >>> CarbonCycleElement('ocean').c_x
        55.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 83.58,
        w: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 2050.0,
        mu: Annotated[float, ParameterSpec()] = 250.0,
        b: Annotated[float, ParameterSpec()] = 4.0,
        theta: Annotated[float, ParameterSpec()] = 5.0,
        c_x: Annotated[float, ParameterSpec()] = 55.0,
        c_p: Annotated[float, ParameterSpec()] = 110.0,
        nu: Annotated[float, ParameterSpec()] = 0.0,
        w0: Annotated[float, ParameterSpec()] = 2000.0,
        gamma: Annotated[float, ParameterSpec()] = 4.0,
        f0: Annotated[float, ParameterSpec()] = 0.694,
        c_f: Annotated[float, ParameterSpec()] = 43.9,
        beta: Annotated[float, ParameterSpec()] = 1.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.w = w
        self.mu = mu
        self.b = b
        self.theta = theta
        self.c_x = c_x
        self.c_p = c_p
        self.nu = nu
        self.w0 = w0
        self.gamma = gamma
        self.f0 = f0
        self.c_f = c_f
        self.beta = beta

    def _get_burial(self, s: SimpleNamespace) -> Expression:
        """Return the burial of organic carbon, a Hill function of the carbonate ion concentration.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            burial:
                The burial, relative to the characteristic rate.
        """
        return s.b * hill(s.c, s.c_p, s.gamma)

    def _get_respiration(self, s: SimpleNamespace) -> Expression:
        """Return the respiration of organic carbon, a decreasing Hill function of the carbonate ion concentration.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            respiration:
                The respiration, relative to the characteristic rate.
        """
        return s.theta * (1 - hill(s.c, s.c_x, s.gamma))

    def _get_buffer(self, s: SimpleNamespace) -> Expression:
        """Return the buffer factor of the carbonate ion concentration, a Hill function of it.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            buffer:
                The buffer factor.
        """
        return s.f0 * hill(s.c, s.c_f, s.beta)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return Rothman's balance of carbonate ion and dissolved inorganic carbon.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the carbonate ion concentration and the dissolved inorganic carbon.
        """
        burial = self._get_burial(s)
        respiration = self._get_respiration(s)
        buffer = self._get_buffer(s)
        return [
            RateLaw(s.c, buffer * (s.mu * (1 - burial - respiration - s.nu) + s.w - s.w0)),
            RateLaw(s.w, s.mu * (1 - burial + respiration + s.nu) - s.w + s.w0),
        ]
