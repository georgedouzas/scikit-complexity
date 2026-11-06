"""Effector cells and tumour cells under a gene therapy that makes the T cells express a tumour receptor."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class TcrGeneTherapyElement(Element):
    """Effector cells and tumour cells under a gene therapy that makes the T cells express a tumour receptor.

    The effector cells follow `dE/dt = s + alpha A - d1 E + r1 E / (E + g1)` and the tumour cells
    `dA/dt = r2 A (1 - b A) - a E A / (A + g2)`, a predator-prey system of the immune attack.

    The element holds the states and parameters and has no laws of its own: the interactions of `tcr_gene_therapy` read
    its parameters.

    Args:
        label:
            The label of the tumour site.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        E:
            The initial number of effector cells.

        A:
            The initial number of tumour cells.

        s:
            The influx rate of effector cells.

        alpha:
            The antigenicity of the tumour.

        d1:
            The death rate of effector cells.

        r1:
            The proliferation rate of effector cells.

        g1:
            The half-saturation constant of the proliferation of effector cells.

        r2:
            The growth rate of tumour cells.

        b:
            The reciprocal carrying capacity of tumour cells.

        a:
            The kill rate that the therapy gives the effector cells.

        g2:
            The half-saturation constant of the kill.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TcrGeneTherapyElement
        >>> TcrGeneTherapyElement('site').r2
        0.18
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1000.0,
        s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1245,
        g1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.001,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.18,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-09,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        g2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100000.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.A = A
        self.s = s
        self.alpha = alpha
        self.d1 = d1
        self.r1 = r1
        self.g1 = g1
        self.r2 = r2
        self.b = b
        self.a = a
        self.g2 = g2
