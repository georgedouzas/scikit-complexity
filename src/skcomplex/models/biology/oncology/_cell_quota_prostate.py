"""Prostate cancer cells whose growth and mutation follow their androgen cell quota, with the PSA they release."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class CellQuotaProstateElement(Element):
    """Prostate cancer cells whose growth and mutation follow their androgen cell quota, with the PSA they release.

    The cells grow at the Droop rate `rm (1 - q_i / Q_i)`, die at `d_i`, and switch from dependent to independent at
    `mx1 m501^n / (Q1^n + m501^n)` and back at `mx2 Q2^n / (Q2^n + m502^n)`. The quotas follow
    `dQ_i/dt = vm (qm - Q_i) / (qm - q_i) U / (U + v50) - rm (Q_i - q_i) - d3 Q_i`, and the serum PSA
    `dP/dt = r0 (AD + AI) + r1 AD Q1^theta / (Q1^theta + rho1^theta) + r2 AI Q2^theta / (Q2^theta + rho2^theta)
    - du P`. The serum androgen relaxes at the rate `du1` to `Umin` during the `width` days of therapy that open
    every `period` days and to `U0` otherwise.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `prostate_cell_quota` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AD:
            The initial number of androgen-dependent cells.

        AI:
            The initial number of androgen-independent cells.

        Q1:
            The initial cell quota of the dependent cells.

        Q2:
            The initial cell quota of the independent cells.

        P:
            The initial serum PSA.

        U:
            The initial serum androgen.

        rm:
            The maximum proliferation rate.

        q1:
            The minimum quota of the dependent cells.

        q2:
            The minimum quota of the independent cells.

        d1:
            The death rate of the dependent cells.

        d2:
            The death rate of the independent cells.

        d3:
            The degradation rate of the quota.

        qm:
            The maximum quota.

        vm:
            The maximum uptake rate of androgen.

        v50:
            The serum androgen at which the uptake is half way.

        mx1:
            The maximum mutation rate from dependent to independent.

        mx2:
            The maximum mutation rate from independent to dependent.

        m501:
            The quota at which the mutation to independence is half way.

        m502:
            The quota at which the mutation back to dependence is half way.

        n:
            The steepness of the mutation rates.

        r0:
            The baseline PSA production per cell.

        r1:
            The androgen-driven PSA production per dependent cell.

        r2:
            The androgen-driven PSA production per independent cell.

        rho1:
            The quota at which the PSA production of the dependent cells is half way.

        rho2:
            The quota at which the PSA production of the independent cells is half way.

        theta:
            The steepness of the PSA production.

        du:
            The clearance rate of the PSA.

        U0:
            The normal serum androgen.

        Umin:
            The serum androgen under therapy.

        du1:
            The rate at which the serum androgen relaxes.

        period:
            The time between the starts of therapy periods.

        width:
            The duration of a therapy period.

    Examples:
        >>> from skcomplex.models.biology.oncology import CellQuotaProstateElement
        >>> CellQuotaProstateElement('tumour').qm
        5.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        AI: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Q1: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        Q2: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 3.4,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        rm: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.025,
        q1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        q2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.15,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.015,
        d2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.015,
        d3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.09,
        qm: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        vm: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.275,
        v50: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        mx1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00015,
        mx2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        m501: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.08,
        m502: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.7,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        r0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        rho1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.3,
        rho2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.1,
        theta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        U0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        Umin: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        du1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        period: float = 500.0,
        width: float = 250.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AD = AD
        self.AI = AI
        self.Q1 = Q1
        self.Q2 = Q2
        self.P = P
        self.U = U
        self.rm = rm
        self.q1 = q1
        self.q2 = q2
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.qm = qm
        self.vm = vm
        self.v50 = v50
        self.mx1 = mx1
        self.mx2 = mx2
        self.m501 = m501
        self.m502 = m502
        self.n = n
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2
        self.rho1 = rho1
        self.rho2 = rho2
        self.theta = theta
        self.du = du
        self.U0 = U0
        self.Umin = Umin
        self.du1 = du1
        self.period = period
        self.width = width
