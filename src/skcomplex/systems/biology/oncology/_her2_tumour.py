"""HER2+ breast cancer cells under trastuzumab and paclitaxel, with antibody binding and drug uptake."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class Her2TumourElement(Element):
    """HER2+ breast cancer cells under trastuzumab and paclitaxel, with antibody binding and drug uptake.

    The tumour grows logistically, `dA/dt = r (1 - DIm) (1 - A / (kc - DCh)) A`. The bound antibody `Ab` lowers the
    proliferation by `DIm = a1 Ab` from the time `t_a` on, and the internalized paclitaxel `Pin` lowers the carrying
    capacity by `DCh = sigma a2 exp(-gamma_p (t - t_p)) Pin` from the dose time `t_p` on. The free antibody and the
    free paclitaxel are drivers, held at the levels `antibody` and `paclitaxel` from their dose times on. The antibody
    binds the receptors, `dAb/dt = theta_ab (Af - Ab) (theta_r A - Ab)`, and the paclitaxel enters the cells,
    `dPin/dt = alpha_p (Pf - Pin) A`, while the free drugs are present. Cells are counted in millions, so the carrying
    capacity is 0.67, the antibody is in millions of molecules, and time is in days.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial number of cells.

        Ab:
            The initial bound antibody.

        Pin:
            The initial internalized paclitaxel.

        r:
            The growth rate.

        kc:
            The carrying capacity.

        a1:
            The inhibition of proliferation per unit of bound antibody.

        a2:
            The reduction of the carrying capacity per unit of internalized paclitaxel.

        gamma_p:
            The decay rate of the cytotoxicity of paclitaxel.

        sigma:
            The synergy of the two drugs, 1 for none.

        theta_ab:
            The binding rate of the antibody.

        theta_r:
            The number of receptors per cell.

        alpha_p:
            The uptake rate of paclitaxel.

        antibody:
            The level of free antibody from its dose on.

        paclitaxel:
            The level of free paclitaxel from its dose on, in pmol.

        t_antibody:
            The time of the antibody dose.

        t_a:
            The time from which the bound antibody inhibits proliferation.

        t_p:
            The time of the paclitaxel dose.

    Examples:
        >>> from skcomplex.systems.biology.oncology import Her2TumourElement
        >>> Her2TumourElement('tumour').kc
        0.67
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.035,
        Ab: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Pin: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.83,
        kc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.67,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.51,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 23.0,
        gamma_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 6.35,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        theta_ab: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 8.78,
        theta_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1940000.0,
        alpha_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.68,
        antibody: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        paclitaxel: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        t_antibody: float = 1.0,
        t_a: float = 2.0,
        t_p: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.Ab = Ab
        self.Pin = Pin
        self.r = r
        self.kc = kc
        self.a1 = a1
        self.a2 = a2
        self.gamma_p = gamma_p
        self.sigma = sigma
        self.theta_ab = theta_ab
        self.theta_r = theta_r
        self.alpha_p = alpha_p
        self.antibody = antibody
        self.paclitaxel = paclitaxel
        self.t_antibody = t_antibody
        self.t_a = t_a
        self.t_p = t_p

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth of the tumour, the binding of the antibody, and the uptake of paclitaxel.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        A, Ab, Pin, t = s.A, s.Ab, s.Pin, s.t
        free_antibody = where(t >= self.t_antibody, s.antibody, 0)
        free_paclitaxel = where(t >= self.t_p, s.paclitaxel, 0)
        inhibition = where(t > self.t_a, s.a1 * Ab, 0)
        cytotoxicity = where(t >= self.t_p, s.sigma * s.a2 * exp(-s.gamma_p * (t - self.t_p)) * Pin, 0)
        binding = s.theta_ab * (free_antibody - Ab) * (s.theta_r * A - Ab)
        return [
            RateLaw(A, s.r * (1 - inhibition) * (1 - A / (s.kc - cytotoxicity)) * A),
            RateLaw(Ab, where(free_antibody > 0, binding, 0)),
            RateLaw(Pin, where(free_paclitaxel > 0, s.alpha_p * (free_paclitaxel - Pin) * A, 0)),
        ]
