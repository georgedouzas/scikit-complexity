"""Normal epithelium and prostate cancer cells driven by activated androgen receptors, with tissue and serum PSA."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class AndrogenReceptorProstateElement(Element):
    """Normal epithelium and prostate cancer cells driven by activated androgen receptors, with tissue and serum PSA.

    The normal cells follow `dN/dt = r1 N (1 - (N + c1 (AD + AI)) / b1) - d1 N`, the dependent cells
    `dAD/dt = r2 AD (1 - m1)(1 - L) - d2 AD`, and the independent cells `dAI/dt = r3 AI (1 - (c1 N + AD + AI) / b3)
    - d3 AI + m1 r2 AD (1 - L)`, where `L = (c1 N + AD + AI) / b2`. The rates are
    `r_i = psi1_i UD^2 / (psi2^2 + UD^2) + phi1_i UT^2 / (phi2^2 + UT^2)` and `d_i = psi3_i / (psi4 + UD + UT)` for
    the receptors activated by DHT, `UD`, and by testosterone, `UT`. The tissue PSA follows
    `dUIP/dt = -du1 UIP Nf - gamma_p UIP (AD + AI)^2 / (KP + AD + AI) - du2 UIP + r4`, with the production
    `r4 = sum_i beta_i (UT + UD + sigma) X_i` over the cell types, and the serum PSA gains the leaks and is cleared
    at `du3`. The capacity of the normal cells `b1` and their share `Nf = N / b1` follow the prostate volume
    `v1 a^phi / (v2^phi + a^phi) + v3` at the age `a`, divided by the volume `Vc` of a billion cells.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `prostate_androgen_receptor` read its parameters.

    Args:
        label:
            The label of the prostate.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial normal epithelial cells.

        AD:
            The initial androgen-dependent cells.

        AI:
            The initial androgen-independent cells.

        UIP:
            The initial tissue PSA.

        UP:
            The initial serum PSA.

        UT:
            The concentration of receptors activated by testosterone.

        UD:
            The concentration of receptors activated by dihydrotestosterone.

        m1:
            The fraction of the divisions of dependent cells that yields independent cells.

        b2:
            The carrying capacity of the dependent cells.

        b3:
            The carrying capacity of the independent cells.

        c1:
            The interaction between normal and cancer cells.

        psi1_n:
            The maximum growth of the normal cells driven by DHT.

        psi1_d:
            The maximum growth of the dependent cells driven by DHT.

        psi1_i:
            The maximum growth of the independent cells driven by DHT.

        phi1_n:
            The maximum growth of the normal cells driven by testosterone.

        phi1_d:
            The maximum growth of the dependent cells driven by testosterone.

        phi1_i:
            The maximum growth of the independent cells driven by testosterone.

        psi2:
            The DHT-activated receptors at which the growth is half way.

        phi2:
            The testosterone-activated receptors at which the growth is half way.

        psi3_n:
            The death of the normal cells.

        psi3_d:
            The death of the dependent cells.

        psi3_i:
            The death of the independent cells.

        psi4:
            The activated receptors at which the death is halved.

        Vc:
            The volume of a billion cells.

        v1:
            The prostate volume that ageing adds.

        v2:
            The age at which half of that volume is added.

        v3:
            The prostate volume of a young man.

        phi:
            The steepness of the growth of the prostate with age.

        age:
            The age of the patient at the start.

        du1:
            The rate at which tissue PSA leaks through normal epithelium.

        du2:
            The decay rate of tissue PSA.

        du3:
            The clearance rate of serum PSA.

        gamma_p:
            The rate at which tissue PSA leaks through cancer cells.

        KP:
            The cancer cells at which the leak per cell is half its maximum.

        beta_n:
            The PSA production of the normal cells.

        beta_d:
            The PSA production of the dependent cells.

        beta_i:
            The PSA production of the independent cells.

        sigma:
            The PSA production independent of the receptors.

    Examples:
        >>> from skcomplex.systems.biology.oncology import AndrogenReceptorProstateElement
        >>> AndrogenReceptorProstateElement('prostate').KP
        800.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        AI: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        UIP: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        UP: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        UT: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
        UD: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        b2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 22.4,
        b3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 22.4,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        psi1_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001202,
        psi1_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001048,
        psi1_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 9.8e-05,
        phi1_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001281,
        phi1_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00111,
        phi1_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.000279,
        psi2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 12.355,
        phi2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 49.587,
        psi3_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.008,
        psi3_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        psi3_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        psi4: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 62.162,
        Vc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.56,
        v1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 18.397,
        v2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 63.753,
        v3: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 28.4,
        phi: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 12.0,
        age: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 60.0,
        du1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001515,
        du2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05635,
        du3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.776,
        gamma_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3125,
        KP: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 800.0,
        beta_n: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0022622,
        beta_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.002,
        beta_i: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.003,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.AD = AD
        self.AI = AI
        self.UIP = UIP
        self.UP = UP
        self.UT = UT
        self.UD = UD
        self.m1 = m1
        self.b2 = b2
        self.b3 = b3
        self.c1 = c1
        self.psi1_n = psi1_n
        self.psi1_d = psi1_d
        self.psi1_i = psi1_i
        self.phi1_n = phi1_n
        self.phi1_d = phi1_d
        self.phi1_i = phi1_i
        self.psi2 = psi2
        self.phi2 = phi2
        self.psi3_n = psi3_n
        self.psi3_d = psi3_d
        self.psi3_i = psi3_i
        self.psi4 = psi4
        self.Vc = Vc
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.phi = phi
        self.age = age
        self.du1 = du1
        self.du2 = du2
        self.du3 = du3
        self.gamma_p = gamma_p
        self.KP = KP
        self.beta_n = beta_n
        self.beta_d = beta_d
        self.beta_i = beta_i
        self.sigma = sigma
