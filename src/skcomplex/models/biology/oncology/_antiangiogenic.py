"""Oncology models: tumours whose vasculature an angiogenesis inhibitor prunes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from sage.all__sagemath_symbolics import exp, log, pi

from ....core.base import Domain, Expression, parameter, state
from ....core.model import Element, Law, Rate, Symbols, System, pulse_train
from ....core.runtime import register_model
from ....core.solvers import ScipyODE

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


class AngiogenicTumour(Element):
    """Tumour volume, the vascular volume that sets its carrying capacity, and an angiogenesis inhibitor.

    The tumour follows `dA/dt = -lambda1 A log(A / Vb)`, the vasculature
    `dVb/dt = -lambda2 Vb + xi_s A - xi_i Vb A^(2/3) - a Vb U`, and the inhibitor `dU/dt = -du U + u(t)`, where the
    infusion `u` is `dose` for `width` time units every `period`, from `start` on. A `width` equal to the `period`
    is a continuous infusion.

    Args:
        label:
            The label of the tumour.

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

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        start:
            The time of the first dose.

    Examples:
        >>> from skcomplex.models.biology.oncology import AngiogenicTumour
        >>> AngiogenicTumour('tumour').xi_s
        5.85
    """

    A: float = state(default=17347.0, domain=Domain.NONNEGATIVE)
    Vb: float = state(default=17347.0, domain=Domain.POSITIVE)
    U: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    lambda1: float = parameter(default=0.192, domain=Domain.POSITIVE)
    lambda2: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    xi_s: float = parameter(default=5.85, domain=Domain.POSITIVE)
    xi_i: float = parameter(default=0.00873, domain=Domain.POSITIVE)
    a: float = parameter(default=0.66, domain=Domain.NONNEGATIVE)
    du: float = parameter(default=1.3, domain=Domain.POSITIVE)
    dose: float = parameter(default=20.0, domain=Domain.NONNEGATIVE)
    period: float = 1.0
    width: float = 1.0
    start: float = 0.0

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(s.A, -s.lambda1 * s.A * log(s.A / s.Vb)),
            Rate(s.Vb, -s.lambda2 * s.Vb + s.xi_s * s.A - s.xi_i * s.Vb * s.A ** (2 / 3) - s.a * s.Vb * s.U),
            Rate(s.U, infusion - s.du * s.U),
        ]


class VascularizedSpheroid(Element):
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
        >>> from skcomplex.models.biology.oncology import VascularizedSpheroid
        >>> VascularizedSpheroid('tumour').Ap
        495.0
    """

    Ar: float = state(default=13.0, domain=Domain.POSITIVE)
    Ac: float = state(default=3.0, domain=Domain.NONNEGATIVE)
    Ap: float = state(default=495.0, domain=Domain.NONNEGATIVE)
    An: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    Vbc: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    Vbp: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    U: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=1.2, domain=Domain.POSITIVE)
    r2: float = parameter(default=12.0, domain=Domain.POSITIVE)
    r3: float = parameter(default=0.06, domain=Domain.NONNEGATIVE)
    m: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    a1: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    p1: float = parameter(default=1.0, domain=Domain.REAL)
    p2: float = parameter(default=4e5, domain=Domain.REAL)
    p3: float = parameter(default=0.2, domain=Domain.REAL)
    p4: float = parameter(default=0.45, domain=Domain.REAL)
    p_g: float = parameter(default=4e5, domain=Domain.REAL)
    kappa1: float = parameter(default=0.17, domain=Domain.REAL)
    kappa2: float = parameter(default=-45000.0, domain=Domain.REAL)
    kappa3: float = parameter(default=-0.04, domain=Domain.REAL)
    kappa4: float = parameter(default=-0.06, domain=Domain.REAL)
    kappa_g: float = parameter(default=55000.0, domain=Domain.REAL)
    theta1: float = parameter(default=0.3, domain=Domain.POSITIVE)
    theta2: float = parameter(default=0.3, domain=Domain.POSITIVE)
    beta: float = parameter(default=1.2, domain=Domain.POSITIVE)
    xi_e: float = parameter(default=0.8, domain=Domain.NONNEGATIVE)
    rv_ref: float = parameter(default=0.005, domain=Domain.POSITIVE)
    Vx: float = parameter(default=0.005, domain=Domain.NONNEGATIVE)
    du: float = parameter(default=INHIBITOR_CLEARANCE, domain=Domain.POSITIVE)
    dose: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    period: float = 7.0
    width: float = 1.0
    start: float = 0.0

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(s.Ar, growth),
            Rate(s.Ac, transfer * s.Ap - necrosis),
            Rate(s.Ap, -transfer * s.Ap + proliferation),
            Rate(s.An, necrosis),
            Rate(s.Vbc, internalization + angiogenesis),
            Rate(s.Vbp, s.Vx * volume_increment - internalization + angiogenesis),
            Rate(s.U, infusion - s.du * s.U),
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
    tumour = AngiogenicTumour(
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
    return System(components=[tumour])


@register_model(
    field='biology/oncology',
    tags=('anti-angiogenic therapy', 'ode', 'angiogenesis', 'necrosis', 'dosing'),
    references=(BOOK_REFERENCE, 'Csercsik et al. 2017'),
    defaults={'until': 60.0, 'dt': 0.2, 'solver': ScipyODE(method='LSODA', rtol=1e-8, atol=1e-8)},
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
    tumour = VascularizedSpheroid(
        'tumour',
        Vx=Vx,
        xi_e=xi_e,
        du=du,
        dose=dose,
        period=period,
        width=width,
        start=start,
    )
    return System(components=[tumour])
