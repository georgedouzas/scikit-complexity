"""Oncology models: chemotherapy combined with immunotherapy, viruses, antibodies, and antiangiogenic drugs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

from sage.all__sagemath_symbolics import exp, log, sin

from ....core.base import Domain, Expression, ParameterError, parameter, state
from ....core.components import Element
from ....core.laws import Law, Rate, Symbols, pulse, where
from ....core.runtime import register_model
from ....core.solvers import ScipyODE
from ....core.system import Observable, System

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'
INFUSIONS = ('constant', 'exponential', 'periodic')


def _doses(t: Expression, start: float, period: float, width: float, count: int) -> Expression:
    """Return a schedule of `count` rectangular doses of unit height, one every `period` from `start`."""
    return sum(pulse(t, start + k * period, width) for k in range(count))


def _toxicity(concentration: Expression, scale: Expression) -> Expression:
    """Return a toxicity linear in a low concentration and exponential in a high one, `exp(c / scale) - 1`."""
    return exp(concentration / scale) - 1


def _total_toxicity(s: Any) -> Expression:  # noqa: ANN401
    """Return the mean toxicity of the antiangiogenic and the immunotherapeutic drug."""
    tumour = s.tumour
    return (_toxicity(tumour.c_an, tumour.tox_an) + _toxicity(tumour.c_im, tumour.tox_im)) / 2


class ChemoimmunotherapyTumour(Element):
    """Tumour with NK cells, CD8 T cells, and circulating lymphocytes under chemotherapy, IL-2, and TIL injection.

    The tumour `A` grows logistically and NK cells `N` and CD8 T cells `E` kill it. CD8 T cells kill at the rate
    `D = a1 (E / A)^l / (g1 + (E / A)^l)`. The circulating lymphocytes `C` feed the NK cells. The chemotherapy `M`
    kills a fraction `1 - exp(-M)` of every population at its own rate, IL-2 `I` stimulates the CD8 T cells, and a TIL
    injection adds CD8 T cells. All three drugs follow the same schedule of doses at the rates `u_til`, `u_il2`, and
    `u_chemo`. Cells are counted, and time is in days.

    Args:
        label:
            The label of the tumour.

        A:
            The initial number of tumour cells.

        N:
            The initial number of NK cells.

        E:
            The initial number of CD8 T cells.

        C:
            The initial number of circulating lymphocytes.

        I:
            The initial amount of IL-2.

        M:
            The initial amount of chemotherapeutic drug.

        r1:
            The growth rate of the tumour.

        b1:
            The reciprocal carrying capacity of the tumour.

        c1:
            The rate at which NK cells kill tumour cells.

        a1:
            The highest rate at which CD8 T cells kill tumour cells.

        g1:
            The half-saturation of the CD8 T cell kill.

        l:
            The exponent of the CD8 T cell kill.

        a2:
            The chemotherapy kill rate of tumour cells.

        m1:
            The rate at which circulating lymphocytes become NK cells.

        d1:
            The death rate of NK cells.

        r2:
            The highest recruitment rate of NK cells by the tumour.

        g2:
            The half-saturation of the NK cell recruitment, in cells².

        psi1:
            The rate at which the tumour inactivates NK cells.

        a3:
            The chemotherapy kill rate of NK cells.

        r3:
            The highest recruitment rate of CD8 T cells by the tumour kill.

        g3:
            The half-saturation of the CD8 T cell recruitment, in cells².

        d2:
            The death rate of CD8 T cells.

        psi2:
            The rate at which the tumour inactivates CD8 T cells.

        psi3:
            The rate at which NK cells regulate CD8 T cells.

        r4:
            The rate at which the NK cell kill of tumour cells activates CD8 T cells.

        r5:
            The rate at which circulating lymphocytes meeting tumour cells activate CD8 T cells.

        a4:
            The chemotherapy kill rate of CD8 T cells.

        r6:
            The highest rate at which IL-2 stimulates CD8 T cells.

        g4:
            The half-saturation of the IL-2 stimulation.

        s1:
            The influx of circulating lymphocytes.

        d3:
            The death rate of circulating lymphocytes.

        a5:
            The chemotherapy kill rate of circulating lymphocytes.

        d_u1:
            The decay rate of IL-2.

        d_u2:
            The decay rate of the chemotherapeutic drug.

        u_til:
            The rate of TIL injection during a dose, in cells per day.

        u_il2:
            The rate of IL-2 injection during a dose.

        u_chemo:
            The rate of chemotherapy infusion during a dose.

        start:
            The time of the first dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.models.biology.oncology import ChemoimmunotherapyTumour
        >>> ChemoimmunotherapyTumour('tumour').r1
        0.431
    """

    A: float = state(default=1e6, domain=Domain.NONNEGATIVE)
    N: float = state(default=5e4, domain=Domain.NONNEGATIVE)
    E: float = state(default=100.0, domain=Domain.NONNEGATIVE)
    C: float = state(default=1.1e7, domain=Domain.NONNEGATIVE)
    I: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    M: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.431, domain=Domain.NONNEGATIVE)
    b1: float = parameter(default=2.17e-8, domain=Domain.POSITIVE)
    c1: float = parameter(default=7.13e-10, domain=Domain.NONNEGATIVE)
    a1: float = parameter(default=8.17, domain=Domain.NONNEGATIVE)
    g1: float = parameter(default=0.618, domain=Domain.POSITIVE)
    l: float = parameter(default=0.657, domain=Domain.POSITIVE)
    a2: float = parameter(default=0.9, domain=Domain.NONNEGATIVE)
    m1: float = parameter(default=1.29e-3, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=4.12e-2, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.4198, domain=Domain.NONNEGATIVE)
    g2: float = parameter(default=2.02e7, domain=Domain.POSITIVE)
    psi1: float = parameter(default=1e-7, domain=Domain.NONNEGATIVE)
    a3: float = parameter(default=0.6, domain=Domain.NONNEGATIVE)
    r3: float = parameter(default=0.996, domain=Domain.NONNEGATIVE)
    g3: float = parameter(default=3.03e5, domain=Domain.POSITIVE)
    d2: float = parameter(default=2e-2, domain=Domain.NONNEGATIVE)
    psi2: float = parameter(default=3.42e-10, domain=Domain.NONNEGATIVE)
    psi3: float = parameter(default=1.8e-8, domain=Domain.NONNEGATIVE)
    r4: float = parameter(default=1.1e-7, domain=Domain.NONNEGATIVE)
    r5: float = parameter(default=3e-11, domain=Domain.NONNEGATIVE)
    a4: float = parameter(default=0.6, domain=Domain.NONNEGATIVE)
    r6: float = parameter(default=0.125, domain=Domain.NONNEGATIVE)
    g4: float = parameter(default=2e7, domain=Domain.POSITIVE)
    s1: float = parameter(default=1.21e5, domain=Domain.NONNEGATIVE)
    d3: float = parameter(default=1.2e-2, domain=Domain.NONNEGATIVE)
    a5: float = parameter(default=0.6, domain=Domain.NONNEGATIVE)
    d_u1: float = parameter(default=10.0, domain=Domain.POSITIVE)
    d_u2: float = parameter(default=0.9, domain=Domain.POSITIVE)
    u_til: float = parameter(default=1e7, domain=Domain.NONNEGATIVE)
    u_il2: float = parameter(default=1e7, domain=Domain.NONNEGATIVE)
    u_chemo: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    start: float = 7.0
    period: float = 7.0
    width: float = 1.0
    doses: int = 3

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of the tumour, the immune cells, and the drugs.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        A, N, E, C, I, M = s.A, s.N, s.E, s.C, s.I, s.M
        schedule = _doses(s.t, self.start, self.period, self.width, self.doses)
        kill = 1 - exp(-M)
        ratio = abs(E) ** s.l
        D = s.a1 * ratio / (s.g1 * abs(A) ** s.l + ratio)
        tumour = s.r1 * A * (1 - s.b1 * A) - s.c1 * N * A - D * A - s.a2 * kill * A
        nk = s.m1 * C - s.d1 * N + s.r2 * A**2 / (s.g2 + A**2) * N - s.psi1 * N * A - s.a3 * kill * N
        recruitment = s.r3 * (D * A) ** 2 / (s.g3 + (D * A) ** 2) * E
        inhibition = s.d2 * E + s.psi2 * E * A + s.psi3 * N * E**2 + s.a4 * kill * E
        activation = (s.r4 * N + s.r5 * C) * A + s.r6 * E * I / (s.g4 + I) + s.u_til * schedule
        return [
            Rate(A, tumour),
            Rate(N, nk),
            Rate(E, recruitment - inhibition + activation),
            Rate(C, s.s1 - s.d3 * C - s.a5 * kill * C),
            Rate(I, -s.d_u1 * I + s.u_il2 * schedule),
            Rate(M, -s.d_u2 * M + s.u_chemo * schedule),
        ]


class ClonalTumour(Element):
    """Two competing tumour clones and an immune response under chemotherapy and a dendritic-cell vaccine.

    Each clone grows logistically, competes with the other clone and with the immune cells `E`, and dies at the
    chemotherapy rate `u_chemo1` or `u_chemo2` and at the rate `b UIm E` of the vaccine-recruited immune cells. The
    immune cells grow logistically to `theta` and are recruited by both clones. Cells are counted, and time is in days.

    Args:
        label:
            The label of the tumour.

        A1:
            The initial number of cells of the first clone.

        A2:
            The initial number of cells of the second clone.

        E:
            The initial number of immune cells.

        r1:
            The growth rate of the first clone.

        r2:
            The growth rate of the second clone.

        b1:
            The reciprocal carrying capacity of the first clone.

        b2:
            The reciprocal carrying capacity of the second clone.

        c1:
            The competition of the second clone on the first.

        c2:
            The competition of the immune cells on the first clone.

        c3:
            The competition of the first clone on the second.

        c4:
            The competition of the immune cells on the second clone.

        r3:
            The growth rate of the immune cells.

        theta:
            The immune threshold.

        s1:
            The rate at which the first clone recruits immune cells.

        s2:
            The rate at which the second clone recruits immune cells.

        u_chemo1:
            The chemotherapy kill rate of the first clone.

        u_chemo2:
            The chemotherapy kill rate of the second clone.

        u_vaccine:
            The level of the immunotherapeutic agent.

    Examples:
        >>> from skcomplex.models.biology.oncology import ClonalTumour
        >>> ClonalTumour('tumour').theta
        300000.0
    """

    A1: float = state(default=9e4, domain=Domain.NONNEGATIVE)
    A2: float = state(default=4e4, domain=Domain.NONNEGATIVE)
    E: float = state(default=500.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.15, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.11, domain=Domain.NONNEGATIVE)
    b1: float = parameter(default=2e-8, domain=Domain.POSITIVE)
    b2: float = parameter(default=2e-8, domain=Domain.POSITIVE)
    c1: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    c2: float = parameter(default=5.5, domain=Domain.NONNEGATIVE)
    c3: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    c4: float = parameter(default=5.5, domain=Domain.NONNEGATIVE)
    r3: float = parameter(default=0.8, domain=Domain.NONNEGATIVE)
    theta: float = parameter(default=3e5, domain=Domain.POSITIVE)
    s1: float = parameter(default=1.8e-3, domain=Domain.NONNEGATIVE)
    s2: float = parameter(default=1.8e-3, domain=Domain.NONNEGATIVE)
    u_chemo1: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    u_chemo2: float = parameter(default=0.02, domain=Domain.NONNEGATIVE)
    u_vaccine: float = parameter(default=20.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the growth and the kill of both clones and the growth of the immune response.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        A1, A2, E = s.A1, s.A2, s.E
        first = s.r1 * A1 * (1 - s.b1 * A1) - (s.c1 * A2 + s.c2 * E) * s.b1 * A1
        second = s.r2 * A2 * (1 - s.b2 * A2) - (s.c3 * A1 + s.c4 * E) * s.b2 * A2
        return [
            Rate(A1, first - s.u_chemo1 * A1 - s.b1 * s.u_vaccine * A1 * E),
            Rate(A2, second - s.u_chemo2 * A2 - s.b2 * s.u_vaccine * A2 * E),
            Rate(E, s.r3 * E * (1 - E / s.theta) + (s.s1 * A1 + s.s2 * A2) * E / s.theta),
        ]


class ChemovirotherapyTumour(Element):
    """Uninfected and infected tumour cells, an oncolytic virus, and a chemotherapeutic drug.

    The uninfected cells `AU` grow logistically, the virus `V` infects them, and infected cells `AV` burst at the rate
    `a2`, releasing `beta_v` virions each. The drug `U` kills both kinds of cells and is infused at the rate `u1`,
    constantly, as `u1 exp(-d_u t)`, or as `u1 sin(d_u t)^2`. One unit of cells is a million cells, and one unit of time
    is two days.

    Args:
        label:
            The label of the tumour.

        AU:
            The initial number of uninfected cells.

        AV:
            The initial number of infected cells.

        V:
            The initial number of virions.

        U:
            The initial amount of drug.

        r1:
            The growth rate of uninfected cells.

        b:
            The reciprocal carrying capacity.

        alpha:
            The infection rate.

        a1:
            The drug kill rate of uninfected cells.

        a2:
            The rate at which the virus kills infected cells.

        a3:
            The drug kill rate of infected cells.

        beta_v:
            The burst size.

        d1:
            The decay rate of the virus.

        d_u:
            The decay rate of the drug.

        u1:
            The infusion rate of the drug.

        infusion:
            The form of the infusion, `'constant'`, `'exponential'`, or `'periodic'`.

    Examples:
        >>> from skcomplex.models.biology.oncology import ChemovirotherapyTumour
        >>> ChemovirotherapyTumour('tumour', infusion='periodic').infusion
        'periodic'
    """

    AU: float = state(default=1.0, domain=Domain.NONNEGATIVE)
    AV: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    V: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    U: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.206, domain=Domain.NONNEGATIVE)
    b: float = parameter(default=1e-6, domain=Domain.POSITIVE)
    alpha: float = parameter(default=0.001, domain=Domain.NONNEGATIVE)
    a1: float = parameter(default=0.005, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=0.5115, domain=Domain.NONNEGATIVE)
    a3: float = parameter(default=0.006, domain=Domain.NONNEGATIVE)
    beta_v: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.001, domain=Domain.NONNEGATIVE)
    d_u: float = parameter(default=4.16, domain=Domain.POSITIVE)
    u1: float = parameter(default=10.0, domain=Domain.NONNEGATIVE)
    infusion: str = 'constant'

    def laws(self, s: Symbols) -> list[Law]:
        """Return the growth, the infection, and the drug kill of the cells, the virus balance, and the drug balance.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.

        Raises:
            ParameterError:
                The infusion is not one of the known forms.
        """
        if self.infusion not in INFUSIONS:
            error_msg = f'The infusion must be one of {INFUSIONS}, got {self.infusion!r}.'
            raise ParameterError(error_msg)
        AU, AV, V, U = s.AU, s.AV, s.V, s.U
        shape = {'constant': 1, 'exponential': exp(-s.d_u * s.t), 'periodic': sin(s.d_u * s.t) ** 2}[self.infusion]
        infection = s.alpha * AU * V
        return [
            Rate(AU, s.r1 * AU * (1 - s.b * (AU + AV)) - infection - s.a1 * AU * U),
            Rate(AV, infection - s.a2 * AV - s.a3 * AV * U),
            Rate(V, s.beta_v * s.a2 * AV - infection - s.d1 * V),
            Rate(U, -s.d_u * U + s.u1 * shape),
        ]


class GliomaBrain(Element):
    """Glia, neurons, and a glioma under temozolomide and an oncolytic virus.

    Normal glia `G` and the glioma compete. Neurons `N` die in proportion to the loss of glia,
    `psi dG/dt H(-dG/dt) N`. The drug `U` kills glia, neurons, and glioma with saturating kills. The virus `V` infects
    only glioma cells, and the infected cells `AV` burst. Cells and virions are counted, the drug is in mg per m², and
    time is in days.

    Args:
        label:
            The label of the brain.

        G:
            The initial number of glial cells.

        N:
            The initial number of neurons.

        AU:
            The initial number of uninfected glioma cells.

        AV:
            The initial number of infected glioma cells.

        V:
            The initial number of virions.

        U:
            The initial amount of drug.

        r1:
            The growth rate of glia.

        r2:
            The growth rate of the glioma.

        b:
            The reciprocal carrying capacity of glia and glioma.

        c1:
            The competition of the glioma on glia.

        c2:
            The competition of glia on the glioma.

        a1:
            The drug kill rate of glia.

        a2:
            The drug kill rate of neurons.

        a3:
            The drug kill rate of the glioma.

        g1:
            The half-saturation of the drug kill of glia.

        g2:
            The half-saturation of the drug kill of neurons.

        g3:
            The half-saturation of the drug kill of the glioma.

        psi:
            The loss of neurons per glial cell lost.

        alpha:
            The infection rate.

        a4:
            The rate at which the virus kills infected cells.

        beta_v:
            The burst size.

        d_u1:
            The decay rate of the virus.

        d_u2:
            The clearance rate of the drug.

        u1:
            The infusion rate of the drug, in mg per m² per day.

    Examples:
        >>> from skcomplex.models.biology.oncology import GliomaBrain
        >>> GliomaBrain('brain').u1
        400.0
    """

    G: float = state(default=5.94e12, domain=Domain.NONNEGATIVE)
    N: float = state(default=5.94e12, domain=Domain.NONNEGATIVE)
    AU: float = state(default=6e10, domain=Domain.NONNEGATIVE)
    AV: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    V: float = state(default=1e9, domain=Domain.NONNEGATIVE)
    U: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=0.0068, domain=Domain.NONNEGATIVE)
    r2: float = parameter(default=0.012, domain=Domain.NONNEGATIVE)
    b: float = parameter(default=0.167e-12, domain=Domain.POSITIVE)
    c1: float = parameter(default=3.0769e-15, domain=Domain.NONNEGATIVE)
    c2: float = parameter(default=3.0769e-16, domain=Domain.NONNEGATIVE)
    a1: float = parameter(default=2.808e5, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=2.808e5, domain=Domain.NONNEGATIVE)
    a3: float = parameter(default=2.808e8, domain=Domain.NONNEGATIVE)
    g1: float = parameter(default=6e12, domain=Domain.POSITIVE)
    g2: float = parameter(default=6e12, domain=Domain.POSITIVE)
    g3: float = parameter(default=6e12, domain=Domain.POSITIVE)
    psi: float = parameter(default=3.418e-13, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=1.2e-10, domain=Domain.NONNEGATIVE)
    a4: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    beta_v: float = parameter(default=1000.0, domain=Domain.NONNEGATIVE)
    d_u1: float = parameter(default=24.0, domain=Domain.NONNEGATIVE)
    d_u2: float = parameter(default=0.2, domain=Domain.POSITIVE)
    u1: float = parameter(default=400.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the balances of glia, neurons, glioma, virus, and drug.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        G, N, AU, AV, V, U = s.G, s.N, s.AU, s.AV, s.V, s.U
        glia = s.r1 * G * (1 - s.b * G) - s.c1 * G * AU - s.a1 * G * U / (s.g1 + G)
        infection = s.alpha * AU * V
        glioma = s.r2 * AU * (1 - s.b * (AU + AV)) - s.c2 * G * AU - s.a3 * AU * U / (s.g3 + AU) - infection
        return [
            Rate(G, glia),
            Rate(N, s.psi * where(glia < 0, glia, 0) * N - s.a2 * N * U / (s.g2 + N)),
            Rate(AU, glioma),
            Rate(AV, infection - s.a4 * AV),
            Rate(V, s.beta_v * s.a4 * AV - infection - s.d_u1 * V),
            Rate(U, -s.d_u2 * U + s.u1),
        ]


class Her2Tumour(Element):
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
        >>> from skcomplex.models.biology.oncology import Her2Tumour
        >>> Her2Tumour('tumour').kc
        0.67
    """

    A: float = state(default=0.035, domain=Domain.NONNEGATIVE)
    Ab: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    Pin: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    r: float = parameter(default=0.83, domain=Domain.NONNEGATIVE)
    kc: float = parameter(default=0.67, domain=Domain.POSITIVE)
    a1: float = parameter(default=2.51, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=23.0, domain=Domain.NONNEGATIVE)
    gamma_p: float = parameter(default=6.35, domain=Domain.NONNEGATIVE)
    sigma: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    theta_ab: float = parameter(default=8.78, domain=Domain.NONNEGATIVE)
    theta_r: float = parameter(default=1.94e6, domain=Domain.NONNEGATIVE)
    alpha_p: float = parameter(default=0.68, domain=Domain.NONNEGATIVE)
    antibody: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    paclitaxel: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    t_antibody: float = 1.0
    t_a: float = 2.0
    t_p: float = 1.0

    def laws(self, s: Symbols) -> list[Law]:
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
            Rate(A, s.r * (1 - inhibition) * (1 - A / (s.kc - cytotoxicity)) * A),
            Rate(Ab, where(free_antibody > 0, binding, 0)),
            Rate(Pin, where(free_paclitaxel > 0, s.alpha_p * (free_paclitaxel - Pin) * A, 0)),
        ]


class AngiogenesisImmunotherapyTumour(Element):
    """Tumour with its vasculature and effector cells under bevacizumab and atezolizumab, with resistance.

    The tumour `A` grows by the Hahnfeldt law toward its vascular capacity `Vb` and effector cells `E` kill it. The
    capacity is stimulated and inhibited by the tumour and destroyed by the antiangiogenic effect `UAn`, and the
    immunotherapy effect `UIm` boosts the effector cells. Each drug follows a two-compartment model with central amount
    `c` and peripheral amount `q`, in mg per kg, infused for `width` days every `period` days. Its effect is
    `c^g / ((f U50)^g + c^g)`, where `f = 1 + delta_r R` and `R` integrates `max(0, u_theta - c)`, so the drug loses
    potency while its concentration stays low. Volumes are in mm³, and time is in days.

    Args:
        label:
            The label of the tumour.

        A:
            The initial tumour volume.

        Vb:
            The initial vascular capacity.

        E:
            The initial effector cells, dimensionless.

        c_an:
            The initial central amount of the antiangiogenic drug.

        q_an:
            The initial peripheral amount of the antiangiogenic drug.

        R_an:
            The initial resistance integral of the antiangiogenic drug.

        c_im:
            The initial central amount of the immunotherapeutic drug.

        q_im:
            The initial peripheral amount of the immunotherapeutic drug.

        R_im:
            The initial resistance integral of the immunotherapeutic drug.

        v1:
            The growth constant of the tumour.

        v2:
            The spontaneous loss of vasculature.

        a1:
            The kill rate of the tumour by effector cells.

        xi_s:
            The stimulation of the vasculature by the tumour.

        xi_i:
            The inhibition of the vasculature by the tumour.

        a2:
            The highest destruction rate of the vasculature by the antiangiogenic drug.

        r1:
            The proliferation rate of effector cells.

        c1:
            The inverse of the tumour volume above which the tumour suppresses the effector cells.

        d1:
            The death rate of effector cells.

        s_e:
            The influx of effector cells.

        a3:
            The highest boost of effector cells by the immunotherapeutic drug.

        alpha_an:
            The distribution rate of the antiangiogenic drug.

        beta_an:
            The elimination rate of the antiangiogenic drug.

        k12_an:
            The rate from the central to the peripheral compartment of the antiangiogenic drug.

        k21_an:
            The rate from the peripheral to the central compartment of the antiangiogenic drug.

        u50_an:
            The amount of antiangiogenic drug of half effect before resistance.

        alpha_im:
            The distribution rate of the immunotherapeutic drug.

        beta_im:
            The elimination rate of the immunotherapeutic drug.

        k12_im:
            The rate from the central to the peripheral compartment of the immunotherapeutic drug.

        k21_im:
            The rate from the peripheral to the central compartment of the immunotherapeutic drug.

        u50_im:
            The amount of immunotherapeutic drug of half effect before resistance.

        g:
            The steepness of the drug effects.

        delta_r:
            The capacity of the tumour to resist the drugs.

        u_theta:
            The amount below which resistance develops.

        tox_an:
            The amount of antiangiogenic drug at which its toxicity turns exponential.

        tox_im:
            The amount of immunotherapeutic drug at which its toxicity turns exponential.

        dose_an:
            The dose of the antiangiogenic drug, in mg per kg.

        dose_im:
            The dose of the immunotherapeutic drug, in mg per kg.

        period:
            The time between the starts of doses.

        width:
            The duration of an infusion.

        doses:
            The number of doses.

    Examples:
        >>> from skcomplex.models.biology.oncology import AngiogenesisImmunotherapyTumour
        >>> AngiogenesisImmunotherapyTumour('tumour').Vb
        625.0
    """

    A: float = state(default=5000.0, domain=Domain.POSITIVE)
    Vb: float = state(default=625.0, domain=Domain.POSITIVE)
    E: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    c_an: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    q_an: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    R_an: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    c_im: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    q_im: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    R_im: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    v1: float = parameter(default=0.025, domain=Domain.NONNEGATIVE)
    v2: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    a1: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    xi_s: float = parameter(default=8.7, domain=Domain.NONNEGATIVE)
    xi_i: float = parameter(default=8.73e-3, domain=Domain.NONNEGATIVE)
    a2: float = parameter(default=1.35, domain=Domain.NONNEGATIVE)
    r1: float = parameter(default=3e-5, domain=Domain.NONNEGATIVE)
    c1: float = parameter(default=5e-4, domain=Domain.NONNEGATIVE)
    d1: float = parameter(default=0.37451, domain=Domain.NONNEGATIVE)
    s_e: float = parameter(default=0.15, domain=Domain.NONNEGATIVE)
    a3: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    alpha_an: float = parameter(default=0.4811, domain=Domain.POSITIVE)
    beta_an: float = parameter(default=0.0348, domain=Domain.POSITIVE)
    k12_an: float = parameter(default=0.223, domain=Domain.NONNEGATIVE)
    k21_an: float = parameter(default=0.215, domain=Domain.POSITIVE)
    u50_an: float = parameter(default=11.4274, domain=Domain.POSITIVE)
    alpha_im: float = parameter(default=0.5827, domain=Domain.POSITIVE)
    beta_im: float = parameter(default=0.0271, domain=Domain.POSITIVE)
    k12_im: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    k21_im: float = parameter(default=0.2455, domain=Domain.POSITIVE)
    u50_im: float = parameter(default=7.1903, domain=Domain.POSITIVE)
    g: float = parameter(default=1.0, domain=Domain.POSITIVE)
    delta_r: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    u_theta: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    tox_an: float = parameter(default=10.0, domain=Domain.POSITIVE)
    tox_im: float = parameter(default=10.0, domain=Domain.POSITIVE)
    dose_an: float = parameter(default=15.0, domain=Domain.NONNEGATIVE)
    dose_im: float = parameter(default=20.0, domain=Domain.NONNEGATIVE)
    period: float = 21.0
    width: float = 1 / 24
    doses: int = 6

    def _drug_laws(self, s: Symbols, drug: str, schedule: Expression) -> tuple[list[Law], Expression]:
        """Return the laws of the compartments and the resistance of a drug, and its effect."""
        c, q, R = getattr(s, f'c_{drug}'), getattr(s, f'q_{drug}'), getattr(s, f'R_{drug}')
        alpha, beta = getattr(s, f'alpha_{drug}'), getattr(s, f'beta_{drug}')
        k12, k21 = getattr(s, f'k12_{drug}'), getattr(s, f'k21_{drug}')
        dose, u50 = getattr(s, f'dose_{drug}'), getattr(s, f'u50_{drug}')
        k10 = alpha * beta / k21
        laws: list[Law] = [
            Rate(c, -(k10 + k12) * c + k21 * q + dose / self.width * schedule),
            Rate(q, k12 * c - k21 * q),
            Rate(R, where(c < s.u_theta, s.u_theta - c, 0)),
        ]
        effect = c**s.g / (((1 + s.delta_r * R) * u50) ** s.g + c**s.g)
        return laws, effect

    def laws(self, s: Symbols) -> list[Law]:
        """Return the tumour, vasculature, and effector balances, and the kinetics and resistance of both drugs.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        A, Vb, E = s.A, s.Vb, s.E
        schedule = _doses(s.t, 0.0, self.period, self.width, self.doses)
        antiangiogenic, UAn = self._drug_laws(s, 'an', schedule)
        immunotherapy, UIm = self._drug_laws(s, 'im', schedule)
        return [
            Rate(A, -s.v1 * A * log(A / Vb) - s.a1 * A * E),
            Rate(Vb, -s.v2 * Vb + s.xi_s * A - s.xi_i * Vb * A ** (2 / 3) - s.a2 * Vb * UAn),
            Rate(E, s.r1 * (A - s.c1 * A**2) * E - s.d1 * E + s.s_e + s.a3 * E * UIm),
            *antiangiogenic,
            *immunotherapy,
        ]


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'immunotherapy', 'combination therapy', 'ode', 'dosing'),
    references=(BOOK, 'de Pillis, Gu, and Radunskaya 2006'),
    defaults={'until': 60.0, 'dt': 0.2, 'solver': ScipyODE(method='LSODA', rtol=1e-6, atol=1e-6)},
)
def chemoimmunotherapy(
    A: float = 1e6,
    N: float = 5e4,
    E: float = 100.0,
    C: float = 1.1e7,
    *,
    u_til: float = 1e7,
    u_il2: float = 1e7,
    u_chemo: float = 2.0,
    start: float = 7.0,
    period: float = 7.0,
    width: float = 1.0,
    doses: int = 3,
) -> System:
    """Return a tumour with NK cells, CD8 T cells, and lymphocytes under chemotherapy, IL-2, and TIL injection.

    The tumour, implanted in a mouse, grows toward a high equilibrium that the immune cells cannot hold back. The
    doses of chemotherapy shrink it, and the injected tumour-infiltrating lymphocytes clear what the drug leaves. The
    parameters are the table of the book, except the decay rates of the drugs, which the book's table swaps. The
    chemotherapeutic drug decays at 0.9 per day and IL-2 at 10 per day, as in de Pillis et al. 2006. The schedule of
    three weekly one-day doses from day 7 is a choice, since the book gives none. Cells are counted, and time is in
    days.

    Args:
        A:
            The initial number of tumour cells.

        N:
            The initial number of NK cells.

        E:
            The initial number of CD8 T cells.

        C:
            The initial number of circulating lymphocytes.

        u_til:
            The rate of TIL injection during a dose, in cells per day.

        u_il2:
            The rate of IL-2 injection during a dose.

        u_chemo:
            The rate of chemotherapy infusion during a dose.

        start:
            The time of the first dose.

        period:
            The time between the starts of doses.

        width:
            The duration of a dose.

        doses:
            The number of doses.

    Returns:
        system:
            The model, with the states `tumour.A`, `tumour.N`, `tumour.E`, `tumour.C`, `tumour.I`, and `tumour.M`.

    Examples:
        >>> from skcomplex.models.biology.oncology import chemoimmunotherapy
        >>> chemoimmunotherapy(u_til=0.0).elements[0].u_til
        0.0
    """
    tumour = ChemoimmunotherapyTumour(
        'tumour',
        A=A,
        N=N,
        E=E,
        C=C,
        u_til=u_til,
        u_il2=u_il2,
        u_chemo=u_chemo,
        start=start,
        period=period,
        width=width,
        doses=doses,
    )
    return System(components=[tumour])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'immunotherapy', 'combination therapy', 'heterogeneity', 'resistance', 'ode'),
    references=(BOOK, 'Piretto, Delitala, and Ferraro 2018'),
    defaults={'until': 300.0, 'dt': 1.0},
)
def clonal_chemoimmunotherapy(
    A1: float = 9e4,
    A2: float = 4e4,
    E: float = 500.0,
    *,
    c1: float = 0.1,
    c3: float = 0.1,
    u_chemo1: float = 0.2,
    u_chemo2: float = 0.02,
    u_vaccine: float = 20.0,
) -> System:
    """Return two competing tumour clones, sensitive and resistant to chemotherapy, under a drug and a vaccine.

    The chemotherapy kills the sensitive first clone ten times faster than the resistant second clone. Alone, it
    removes the sensitive clone and frees the resistant clone from competition, so the tumour relapses resistant. With
    the dendritic-cell vaccine, the recruited immune cells clear both clones. The book does not model the drugs, so
    their effects are constant rates, and their values are a choice. The competition rates take a value in the book's
    range of 0.01 to 1. Cells are counted, and time is in days.

    Args:
        A1:
            The initial number of cells of the sensitive clone.

        A2:
            The initial number of cells of the resistant clone.

        E:
            The initial number of immune cells.

        c1:
            The competition of the second clone on the first.

        c3:
            The competition of the first clone on the second.

        u_chemo1:
            The chemotherapy kill rate of the first clone.

        u_chemo2:
            The chemotherapy kill rate of the second clone.

        u_vaccine:
            The level of the immunotherapeutic agent.

    Returns:
        system:
            The model, with the clones `tumour.A1` and `tumour.A2` and the immune cells `tumour.E`.

    Examples:
        >>> from skcomplex.models.biology.oncology import clonal_chemoimmunotherapy
        >>> clonal_chemoimmunotherapy(u_vaccine=0.0).elements[0].u_vaccine
        0.0
    """
    tumour = ClonalTumour(
        'tumour',
        A1=A1,
        A2=A2,
        E=E,
        c1=c1,
        c3=c3,
        u_chemo1=u_chemo1,
        u_chemo2=u_chemo2,
        u_vaccine=u_vaccine,
    )
    return System(components=[tumour])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'virotherapy', 'combination therapy', 'oncolytic virus', 'ode'),
    references=(BOOK, 'Malinzi 2019'),
    defaults={'until': 32.0, 'dt': 0.1},
)
def chemovirotherapy(
    AU: float = 1.0,
    V: float = 0.1,
    U: float = 0.1,
    *,
    u1: float = 10.0,
    infusion: str = 'constant',
) -> System:
    """Return uninfected and infected tumour cells, an oncolytic virus, and a chemotherapeutic drug.

    The virus infects the growing tumour, and the drug kills both uninfected and infected cells. Over the default run
    of 64 days, before the virus bursts through the tumour, the combination leaves the smallest tumour. Later the virus
    alone clears the tumour slightly faster, because the drug also kills the infected cells that make the virus. The
    infusion rate is a choice, since the book gives none. One unit of cells is a million cells, and one unit of time is
    two days.

    Args:
        AU:
            The initial number of uninfected cells.

        V:
            The initial number of virions, 0 for chemotherapy alone.

        U:
            The initial amount of drug.

        u1:
            The infusion rate of the drug, 0 for virotherapy alone.

        infusion:
            The form of the infusion, `'constant'`, `'exponential'`, or `'periodic'`.

    Returns:
        system:
            The model, with the cells `tumour.AU` and `tumour.AV`, the virus `tumour.V`, and the drug `tumour.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import chemovirotherapy
        >>> chemovirotherapy(infusion='exponential').elements[0].infusion
        'exponential'
    """
    return System(components=[ChemovirotherapyTumour('tumour', AU=AU, V=V, U=U, u1=u1, infusion=infusion)])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'virotherapy', 'combination therapy', 'glioma', 'neurons', 'ode'),
    references=(BOOK, 'Urenda-Cázares, Gallegos, and Macías-Díaz 2020', 'Iarosz et al. 2015'),
    defaults={'until': 300.0, 'dt': 1.0},
)
def glioma_chemovirotherapy(
    AU: float = 6e10,
    V: float = 1e9,
    *,
    u1: float = 400.0,
) -> System:
    """Return glia, neurons, and a glioma under a constant temozolomide infusion and an oncolytic virus.

    Untreated, the glioma grows and crowds out the glia, and the neurons die with the glia. The virus or the drug clears
    the glioma, and the neurons survive. The drug also kills some neurons. The book gives no initial virus or drug, so
    the run starts with no drug and with a billion virions. Cells and virions are counted, the drug is in mg per m², and
    time is in days.

    Args:
        AU:
            The initial number of glioma cells.

        V:
            The initial number of virions, 0 for chemotherapy alone.

        u1:
            The infusion rate of temozolomide, in mg per m² per day, 0 for virotherapy alone.

    Returns:
        system:
            The model, with the glia `brain.G`, the neurons `brain.N`, the glioma `brain.AU` and `brain.AV`, the virus
            `brain.V`, and the drug `brain.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import glioma_chemovirotherapy
        >>> glioma_chemovirotherapy(u1=0.0).elements[0].u1
        0.0
    """
    return System(components=[GliomaBrain('brain', AU=AU, V=V, u1=u1)])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'targeted therapy', 'combination therapy', 'HER2', 'antibody', 'ode'),
    references=(BOOK, 'Jarrett et al. 2019'),
    defaults={'until': 10.0, 'dt': 0.02, 'solver': ScipyODE(method='LSODA', rtol=1e-8, atol=1e-10)},
)
def her2_targeted_therapy(
    A: float = 0.035,
    *,
    antibody: float = 0.2,
    paclitaxel: float = 2.0,
    t_antibody: float = 1.0,
    t_a: float = 2.0,
    t_p: float = 1.0,
    sigma: float = 1.0,
) -> System:
    """Return HER2+ breast cancer cells under trastuzumab and paclitaxel, with antibody binding and drug uptake.

    The free antibody and the free paclitaxel are drivers held at their levels from their doses on. The antibody binds
    the receptors, which are in excess, so the bound antibody rises to the free antibody, and from the time `t_a` on it
    lowers the growth rate to `r (1 - a1 Ab)`. The paclitaxel enters the cells and briefly lowers the carrying capacity.
    The cytotoxicity falls from one at the dose, so `t_p` is the dose time. The book's units cannot hold 35,000 cells
    with a capacity of 0.67, so cells are counted in millions and the antibody in millions of molecules. The dose levels
    and times are a choice. Time is in days.

    Args:
        A:
            The initial number of cells, in millions.

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

        sigma:
            The synergy of the two drugs, 1 for none.

    Returns:
        system:
            The model, with the cells `tumour.A`, the bound antibody `tumour.Ab`, and the internalized paclitaxel
            `tumour.Pin`.

    Examples:
        >>> from skcomplex.models.biology.oncology import her2_targeted_therapy
        >>> her2_targeted_therapy(sigma=1.5).elements[0].sigma
        1.5
    """
    tumour = Her2Tumour(
        'tumour',
        A=A,
        antibody=antibody,
        paclitaxel=paclitaxel,
        t_antibody=t_antibody,
        t_a=t_a,
        t_p=t_p,
        sigma=sigma,
    )
    return System(components=[tumour])


@register_model(
    field='biology/oncology',
    tags=('antiangiogenic therapy', 'immunotherapy', 'combination therapy', 'pharmacokinetics', 'resistance', 'ode'),
    references=(BOOK, 'Teles and Lemos 2019', 'Hahnfeldt et al. 1999'),
    defaults={'until': 150.0, 'dt': 0.25},
    observables=lambda system: [Observable('toxicity', _total_toxicity)],
)
def angiogenesis_immunotherapy(
    A: float = 5000.0,
    Vb: float = 625.0,
    E: float = 0.1,
    *,
    dose_an: float = 15.0,
    dose_im: float = 20.0,
    a2: float = 1.35,
    a3: float = 0.1,
    period: float = 21.0,
    doses: int = 6,
) -> System:
    """Return the Hahnfeldt tumour with effector cells under bevacizumab and atezolizumab every three weeks.

    Each drug alone holds the tumour or slows it, and together they shrink it. The drug amounts follow the
    two-compartment model whose bolus solution is the book's closed form, with the elimination rate
    `k10 = alpha beta / k21`. The effector cells grow as `r1 (A - c1 A^2) E`, so `1 / c1` is the volume above which the
    tumour suppresses them. The drug effects in the vasculature and effector balances are the saturating effects of
    the book's pharmacodynamics. The resistance capacity `delta_r = 0.01` and its threshold `u_theta = 5` mg per kg
    are a choice, since the book gives none. The toxicity of each drug is `exp(c / tox) - 1`, linear for a low amount
    and exponential for a high one, and the observable `toxicity` is the mean of both. Volumes are in mm³, drug amounts
    in mg per kg, and time is in days.

    Args:
        A:
            The initial tumour volume.

        Vb:
            The initial vascular capacity.

        E:
            The initial effector cells, dimensionless.

        dose_an:
            The dose of bevacizumab, in mg per kg, 0 for immunotherapy alone.

        dose_im:
            The dose of atezolizumab, in mg per kg, 0 for antiangiogenic therapy alone.

        a2:
            The highest destruction rate of the vasculature by bevacizumab, in the book's range of 1.35 to 11.25.

        a3:
            The highest boost of effector cells by atezolizumab, in the book's range of 0.1 to 0.5.

        period:
            The time between doses.

        doses:
            The number of doses.

    Returns:
        system:
            The model, with the tumour `tumour.A`, the capacity `tumour.Vb`, the effector cells `tumour.E`, the drug
            amounts `tumour.c_an` and `tumour.c_im`, and the observable `toxicity`.

    Examples:
        >>> from skcomplex.models.biology.oncology import angiogenesis_immunotherapy
        >>> angiogenesis_immunotherapy(dose_an=0.0).elements[0].dose_an
        0.0
    """
    tumour = AngiogenesisImmunotherapyTumour(
        'tumour',
        A=A,
        Vb=Vb,
        E=E,
        dose_an=dose_an,
        dose_im=dose_im,
        a2=a2,
        a3=a3,
        period=period,
        doses=doses,
    )
    return System(components=[tumour])
