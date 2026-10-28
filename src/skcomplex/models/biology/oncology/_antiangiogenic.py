"""Oncology models: tumours whose vasculature an angiogenesis inhibitor prunes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, log, oo, pi
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....core.laws.utils import pulse_train
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ..._catalog import register_model

BOOK_REFERENCE = 'Padmanabhan, Meskin, and Al Moustafa 2021'
INHIBITOR_CLEARANCE = math.log(2) / 3.9


def _switch(x: Expression, p: Expression, kappa: Expression) -> Expression:
    """Return the sigmoid `1 / (1 + exp((p - x) / kappa))`, which rises through `x = p` for a positive `kappa`."""
    return 1 / (1 + exp((p - x) / kappa))


def _ball(radius: Expression) -> Expression:
    """Return the volume of a ball of a radius."""
    return 4 * pi * radius**3 / 3


def _core_ratio(radius: Expression) -> Expression:
    """Return the core-periphery ratio `(1 / (1 + exp((0.5 - Ar) / 0.333)))^2.2 - 0.15` of a tumour radius."""
    return _switch(radius, 0.5, 0.333) ** 2.2 - 0.15


class AngiogenicTumourElement(Element):
    """Tumour volume, the vascular volume that sets its carrying capacity, and an angiogenesis inhibitor.

    The tumour follows `dA/dt = -lambda1 A log(A / Vb)`, the vasculature
    `dVb/dt = -lambda2 Vb + xi_s A - xi_i Vb A^(2/3) - a Vb U`, and the inhibitor `dU/dt = -du U + u(t)`, where the
    infusion `u` is `dose` for `width` time units every `period`, from `start` on. A `width` equal to the `period`
    is a continuous infusion.

    Args:
        label:
            The label of the tumour.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        **values:
            The values of the declared variables, by name:

            A:
                The initial tumour volume.

            Vb:
                The initial vascular volume.

            U:
                The initial concentration of the inhibitor.

            lambda1:
                The rate at which the tumour approaches its capacity.

            lambda2:
                The spontaneous loss rate of the vasculature.

            xi_s:
                The angiogenic stimulation by the tumour.

            xi_i:
                The angiogenic inhibition by the tumour.

            a:
                The vascular loss per unit of inhibitor.

            du:
                The clearance rate of the inhibitor.

            dose:
                The infusion rate during a dose.

    Examples:
        >>> from skcomplex.models.biology.oncology import AngiogenicTumourElement
        >>> AngiogenicTumourElement('tumour').xi_s
        5.85
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 17347.0,
        Vb: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 17347.0,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lambda1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.192,
        lambda2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        xi_s: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.85,
        xi_i: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.00873,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.66,
        du: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
        dose: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        period: float = 1.0,
        width: float = 1.0,
        start: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.Vb = Vb
        self.U = U
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.xi_s = xi_s
        self.xi_i = xi_i
        self.a = a
        self.du = du
        self.dose = dose
        self.period = period
        self.width = width
        self.start = start

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Gompertz growth of the tumour, the balance of the vasculature, and the inhibitor kinetics.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        infusion = s.dose * pulse_train(s.t, self.period, self.width, start=self.start)
        return [
            RateLaw(s.A, -s.lambda1 * s.A * log(s.A / s.Vb)),
            RateLaw(s.Vb, -s.lambda2 * s.Vb + s.xi_s * s.A - s.xi_i * s.Vb * s.A ** (2 / 3) - s.a * s.Vb * s.U),
            RateLaw(s.U, infusion - s.du * s.U),
        ]


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

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

        **values:
            The values of the declared variables, by name:

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
        ratio = _core_ratio(s.Ar)
        core = _ball(ratio * s.Ar)
        periphery = _ball(s.Ar) - core
        density = s.Ap / periphery
        growth = s.r1 * _switch(density, s.p_g, s.kappa_g)
        core_increment = _ball(ratio * (s.Ar + growth)) - core
        volume_increment = _ball(s.Ar + growth) - _ball(s.Ar)
        transfer = core_increment / periphery
        core_nutrient = s.Vbc / (core * s.rv_ref)
        periphery_nutrient = s.Vbp / (periphery * s.rv_ref)
        necrosis = s.m * _switch(core_nutrient, s.p3, s.kappa3) ** s.theta1 * s.Ac
        proliferation = s.r2 * _switch(periphery_nutrient, s.p1, s.kappa1) * _switch(density, s.p2, s.kappa2) * s.Ap
        signal = _switch(core_nutrient, s.p4, s.kappa4) ** s.theta2
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


@register_model(
    field='biology/oncology',
    tags=('anti-angiogenic therapy', 'ode', 'angiogenesis', 'Gompertz', 'dosing'),
    references=(BOOK_REFERENCE, 'Hahnfeldt et al. 1999'),
    defaults={'until': 30.0, 'dt': 0.1},
)
def hahnfeldt(
    A: float = 17347.0,
    Vb: float = 17347.0,
    *,
    lambda1: float = 0.192,
    lambda2: float = 0.0,
    xi_s: float = 5.85,
    xi_i: float = 0.00873,
    a: float = 0.66,
    du: float = 1.3,
    dose: float = 20.0,
    period: float = 1.0,
    width: float = 1.0,
    start: float = 0.0,
) -> System:
    """Return the Hahnfeldt model, a tumour whose carrying capacity is the vasculature it stimulates and inhibits.

    The tumour follows `dA/dt = -lambda1 A log(A / Vb)` and the vasculature
    `dVb/dt = -lambda2 Vb + xi_s A - xi_i Vb A^(2/3) - a Vb U`. The inhibitor follows `dU/dt = -du U + u(t)`, the
    differential form of the book's convolution. Untreated, tumour and vasculature tend to the volume
    `(xi_s / xi_i)^(3/2)`, about 17347 mm³, which is the initial state. The inhibitor parameters are those of endostatin
    in Lewis lung carcinoma, and the default schedule is the continuous infusion of 20 mg/kg/day of Hahnfeldt et al.,
    under which the vasculature and then the tumour regress. Volumes are in mm³, time in days, and the inhibitor in
    mg/kg.

    Args:
        A:
            The initial tumour volume.

        Vb:
            The initial vascular volume.

        lambda1:
            The rate at which the tumour approaches its capacity.

        lambda2:
            The spontaneous loss rate of the vasculature.

        xi_s:
            The angiogenic stimulation by the tumour.

        xi_i:
            The angiogenic inhibition by the tumour.

        a:
            The vascular loss per unit of inhibitor.

        du:
            The clearance rate of the inhibitor.

        dose:
            The infusion rate during a dose, 0 for no treatment.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose, equal to `period` for a continuous infusion.

        start:
            The time of the first dose.

    Returns:
        system:
            The model, with the volumes `tumour.A` and `tumour.Vb` and the inhibitor `tumour.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import hahnfeldt
        >>> hahnfeldt(dose=0.0).elements[0].dose
        0.0
    """
    tumour = AngiogenicTumourElement(
        'tumour',
        A=A,
        Vb=Vb,
        lambda1=lambda1,
        lambda2=lambda2,
        xi_s=xi_s,
        xi_i=xi_i,
        a=a,
        du=du,
        dose=dose,
        period=period,
        width=width,
        start=start,
    )
    return System(elements=[tumour])


@register_model(
    field='biology/oncology',
    tags=('anti-angiogenic therapy', 'ode', 'angiogenesis', 'necrosis', 'dosing'),
    references=(BOOK_REFERENCE, 'Csercsik et al. 2017'),
    defaults={'until': 60.0, 'dt': 0.2, 'solver': ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-8)},
)
def vascular_core_periphery(
    *,
    Vx: float = 0.005,
    xi_e: float = 0.8,
    du: float = INHIBITOR_CLEARANCE,
    dose: float = 5.0,
    period: float = 7.0,
    width: float = 1.0,
    start: float = 0.0,
) -> System:
    """Return a tumour with a live core, a live periphery, and a necrotic core, each fed by its own vasculature.

    The radius grows once the periphery is dense, cells pass from the periphery to the core as it grows, and core cells
    die when the core vasculature is too sparse. Starved cells release angiogenic factor, which grows both vascular
    compartments, and the inhibitor lowers the angiogenesis by `exp(-xi_e U)`. The book writes the radius, core
    volume, and tumour volume as increments. They are read as rates, so the increment of the radius is `dAr/dt` and
    the others follow from it per unit time, which is how the bi-compartmental model of Csercsik et al. advances. The
    book does not give the density `Vx` of the surrounding vasculature, which is set to the reference vascular fraction
    `rv_ref`, nor a schedule, which is a one-day infusion of 5 mg/kg/day every week. The other parameters and the
    initial state are the book's, the clearance `du` is `log(2) / 3.9` per day, lengths are in mm, volumes in mm³,
    time in days, and the inhibitor in mg/kg.

    Args:
        Vx:
            The vascular density of the surrounding tissue.

        xi_e:
            The efficiency of the inhibitor.

        du:
            The clearance rate of the inhibitor.

        dose:
            The infusion rate during a dose, 0 for no treatment.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

    Returns:
        system:
            The model, with the radius `tumour.Ar`, the cells `tumour.Ac`, `tumour.Ap`, and `tumour.An`, the vascular
            volumes `tumour.Vbc` and `tumour.Vbp`, and the inhibitor `tumour.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import vascular_core_periphery
        >>> vascular_core_periphery(dose=0.0).elements[0].dose
        0.0
    """
    tumour = VascularizedSpheroidElement(
        'tumour',
        Vx=Vx,
        xi_e=xi_e,
        du=du,
        dose=dose,
        period=period,
        width=width,
        start=start,
    )
    return System(elements=[tumour])
