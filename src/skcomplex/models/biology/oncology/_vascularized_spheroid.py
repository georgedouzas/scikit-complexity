"""Tumour spheroid with a live core, a live periphery, a necrotic core, and the vasculature of core and periphery."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo, pi
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import logistic, pulse_train
from ....specs import ParameterSpec, StateSpec, VisualSpec

INHIBITOR_CLEARANCE = math.log(2) / 3.9


class VascularizedSpheroidElement(Element):
    """Tumour spheroid with a live core, a live periphery, a necrotic core, and the vasculature of core and periphery.

    The radius grows as `dAr/dt = r1 g(Ap / Vp)`, with `g` a sigmoid of the peripheral cell density. The core volume
    `Vc = 4/3 pi (dc Ar)^3` and the periphery `Vp = 4/3 pi Ar^3 - Vc` follow from the radius and the core-periphery
    ratio `dc(Ar)`. The increments of the radius, the core volume, and the tumour volume are read as rates, so
    `dAr = dAr/dt`, `dVc = 4/3 pi (dc (Ar + dAr))^3 - Vc` and `dVt = 4/3 pi ((Ar + dAr)^3 - Ar^3)` are per unit
    time. Peripheral cells pass to the core at the rate `dVc / Vp`, proliferate with the nutrient `Vbp / (Vp rv_ref)`
    and the density, and core cells die by necrosis when the core nutrient `Vbc / (Vc rv_ref)` is low. The vasculature
    grows by tumour angiogenic factor, which the inhibitor `U` lowers by `exp(-xi_e U)`, the periphery takes in
    external vasculature of density `Vx` as the tumour grows, and peripheral vessels pass to the core.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        Ar:
            The initial radius.

        Ac:
            The initial number of live core cells.

        Ap:
            The initial number of live peripheral cells.

        An:
            The initial number of necrotic cells.

        Vbc:
            The initial vascular volume of the core.

        Vbp:
            The initial vascular volume of the periphery.

        U:
            The initial concentration of the inhibitor.

        r1:
            The growth rate of the radius.

        r2:
            The proliferation rate of the peripheral cells.

        r3:
            The rate at which peripheral vessels pass to the core.

        m:
            The necrosis rate of the core cells.

        a1:
            The angiogenesis rate.

        p1:
            The peripheral nutrient at half proliferation.

        p2:
            The peripheral density at half proliferation.

        p3:
            The core nutrient at half necrosis.

        p4:
            The core nutrient at half angiogenic factor production.

        p_g:
            The peripheral density at half the radial growth.

        kappa1:
            The width of the nutrient switch of proliferation.

        kappa2:
            The width of the density switch of proliferation, negative so that crowding stops it.

        kappa3:
            The width of the necrosis switch, negative so that starvation causes it.

        kappa4:
            The width of the angiogenic factor switch, negative so that starvation causes it.

        kappa_g:
            The width of the density switch of the radial growth.

        theta1:
            The exponent of the necrosis switch.

        theta2:
            The exponent of the angiogenic factor switch.

        beta:
            The exponent of the radius in the passage of vessels to the core.

        xi_e:
            The efficiency of the inhibitor.

        rv_ref:
            The reference vascular fraction that supports tumour cells.

        Vx:
            The vascular density of the surrounding tissue.

        du:
            The clearance rate of the inhibitor.

        dose:
            The infusion rate during a dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

    Examples:
        >>> from skcomplex.models.biology.oncology import VascularizedSpheroidElement
        >>> VascularizedSpheroidElement('tumour').Ap
        495.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Ar: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 13.0,
        Ac: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        Ap: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 495.0,
        An: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Vbc: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Vbp: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.2,
        r2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 12.0,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.06,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        p1: Annotated[float, ParameterSpec()] = 1.0,
        p2: Annotated[float, ParameterSpec()] = 400000.0,
        p3: Annotated[float, ParameterSpec()] = 0.2,
        p4: Annotated[float, ParameterSpec()] = 0.45,
        p_g: Annotated[float, ParameterSpec()] = 400000.0,
        kappa1: Annotated[float, ParameterSpec()] = 0.17,
        kappa2: Annotated[float, ParameterSpec()] = -45000.0,
        kappa3: Annotated[float, ParameterSpec()] = -0.04,
        kappa4: Annotated[float, ParameterSpec()] = -0.06,
        kappa_g: Annotated[float, ParameterSpec()] = 55000.0,
        theta1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        theta2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.2,
        xi_e: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        rv_ref: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.005,
        Vx: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        du: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = INHIBITOR_CLEARANCE,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        period: float = 7.0,
        width: float = 1.0,
        start: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Ar = Ar
        self.Ac = Ac
        self.Ap = Ap
        self.An = An
        self.Vbc = Vbc
        self.Vbp = Vbp
        self.U = U
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.m = m
        self.a1 = a1
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p_g = p_g
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.kappa3 = kappa3
        self.kappa4 = kappa4
        self.kappa_g = kappa_g
        self.theta1 = theta1
        self.theta2 = theta2
        self.beta = beta
        self.xi_e = xi_e
        self.rv_ref = rv_ref
        self.Vx = Vx
        self.du = du
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth of the radius, the cell balances, the vascular balances, and the inhibitor kinetics.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        ratio = VascularizedSpheroidElement._core_ratio(s.Ar)
        core = VascularizedSpheroidElement._ball(ratio * s.Ar)
        periphery = VascularizedSpheroidElement._ball(s.Ar) - core
        density = s.Ap / periphery
        growth = s.r1 * logistic(density, s.p_g, s.kappa_g)
        core_increment = VascularizedSpheroidElement._ball(ratio * (s.Ar + growth)) - core
        volume_increment = VascularizedSpheroidElement._ball(s.Ar + growth) - VascularizedSpheroidElement._ball(s.Ar)
        transfer = core_increment / periphery
        core_nutrient = s.Vbc / (core * s.rv_ref)
        periphery_nutrient = s.Vbp / (periphery * s.rv_ref)
        necrosis = s.m * logistic(core_nutrient, s.p3, s.kappa3) ** s.theta1 * s.Ac
        proliferation = s.r2 * logistic(periphery_nutrient, s.p1, s.kappa1) * logistic(density, s.p2, s.kappa2) * s.Ap
        signal = logistic(core_nutrient, s.p4, s.kappa4) ** s.theta2
        angiogenesis = s.a1 * exp(-s.xi_e * s.U) * signal * s.Vbp
        internalization = s.r3 * s.Ar**s.beta * transfer * s.Vbp
        infusion = s.dose * pulse_train(s.t, self.period, self.width, start=self.start)
        return [
            RateLaw(s.Ar, growth),
            RateLaw(s.Ac, transfer * s.Ap - necrosis),
            RateLaw(s.Ap, -transfer * s.Ap + proliferation),
            RateLaw(s.An, necrosis),
            RateLaw(s.Vbc, internalization + angiogenesis),
            RateLaw(s.Vbp, s.Vx * volume_increment - internalization + angiogenesis),
            RateLaw(s.U, infusion - s.du * s.U),
        ]

    @staticmethod
    def _ball(radius: Expression) -> Expression:
        """Return the volume of a ball of a radius."""
        return 4 * pi * radius**3 / 3

    @staticmethod
    def _core_ratio(radius: Expression) -> Expression:
        """Return the core-periphery ratio `(1 / (1 + exp((0.5 - Ar) / 0.333)))^2.2 - 0.15` of a tumour radius."""
        return logistic(radius, 0.5, 0.333) ** 2.2 - 0.15
