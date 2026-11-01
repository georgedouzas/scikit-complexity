"""Androgen-dependent and androgen-independent prostate cancer cells under continuous androgen suppression."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class AndrogenSuppressionElement(Element):
    """Androgen-dependent and androgen-independent prostate cancer cells under continuous androgen suppression.

    The dependent cells follow `dAD/dt = r1 U (1 - b (AD + c1 AI)) AD - (d1 + m1)(1 - U / U0) AD`, the independent
    cells `dAI/dt = r2 (1 - b (AI + c2 AD)) AI + m1 (1 - U / U0) AD`, and the serum androgen
    `du/dt = -du (U - U0) - U0 du a`, which the drug efficacy `a` lowers to `U0 (1 - a)`.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `continuous_androgen_suppression` read its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AD:
            The initial number of androgen-dependent cells.

        AI:
            The initial number of androgen-independent cells.

        U:
            The initial serum androgen concentration.

        r1:
            The growth rate of the dependent cells per unit of androgen.

        r2:
            The growth rate of the independent cells.

        b:
            The reciprocal carrying capacity.

        c1:
            The competition of the independent cells on the dependent ones.

        c2:
            The competition of the dependent cells on the independent ones.

        d1:
            The death rate of the dependent cells under androgen deprivation.

        m1:
            The rate at which androgen deprivation turns dependent cells independent.

        U0:
            The normal androgen concentration.

        du:
            The elimination rate of the androgen.

        a:
            The efficacy of the suppression.

    Examples:
        >>> from skcomplex.models.biology.oncology import AndrogenSuppressionElement
        >>> AndrogenSuppressionElement('tumour', a=0.6).a
        0.6
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AD: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        AI: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.14,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.025,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.006,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 9.1e-11,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.064,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5e-05,
        U0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        du: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.9,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AD = AD
        self.AI = AI
        self.U = U
        self.r1 = r1
        self.r2 = r2
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.d1 = d1
        self.m1 = m1
        self.U0 = U0
        self.du = du
        self.a = a
