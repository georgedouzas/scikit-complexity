"""Immune, tumour, and normal cells under a chemotherapy whose infusion rate is a bounded control input."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....specs import ParameterSpec, VisualSpec
from ._tumour_immune_chemotherapy import TumourImmuneChemotherapyElement

MAX_DOSE = 1.0


class OptimalChemotherapyElement(TumourImmuneChemotherapyElement):
    """Immune, tumour, and normal cells under a chemotherapy whose infusion rate is a bounded control input.

    The cells and the drug follow the model of de Pillis and Radunskaya, `TumourImmuneChemotherapyElement`, with the
    infusion rate `u` between 0 and its largest dose. The oncologist minimises the tumour burden and the toxicity of
    the drug, `∫ (A + w u^2 / 2) dt + A(T)`, where the terminal term weighs the tumour left at the end of the
    treatment.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        E:
            The initial immune cells.

        A:
            The initial tumour cells.

        N:
            The initial normal cells.

        U:
            The initial drug concentration.

        s:
            The influx of immune cells.

        rho:
            The immune response rate.

        alpha:
            The immune threshold.

        c1:
            The rate at which the tumour inactivates immune cells.

        c2:
            The rate at which immune cells kill the tumour.

        c3:
            The rate at which normal cells compete with the tumour.

        c4:
            The rate at which the tumour competes with normal cells.

        d1:
            The death rate of immune cells.

        r1:
            The growth rate of the tumour.

        r2:
            The growth rate of normal cells.

        b1:
            The reciprocal carrying capacity of the tumour.

        b2:
            The reciprocal carrying capacity of normal cells.

        a1:
            The fraction of immune cells the drug kills.

        a2:
            The fraction of tumour cells the drug kills.

        a3:
            The fraction of normal cells the drug kills.

        k:
            The decay rate of the drug, `dU` in the book.

        u:
            The infusion rate of the drug, the control input.

        w:
            The weight of the squared infusion rate in the cost.

    Examples:
        >>> from skcomplex.models.biology.oncology import OptimalChemotherapyElement
        >>> OptimalChemotherapyElement('patient', w=0.5).w
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: float = 0.15,
        A: float = 0.25,
        N: float = 1.0,
        U: float = 0.0,
        s: float = 0.33,
        rho: float = 0.01,
        alpha: float = 0.3,
        c1: float = 1.0,
        c2: float = 0.5,
        c3: float = 1.0,
        c4: float = 1.0,
        d1: float = 0.2,
        r1: float = 1.5,
        r2: float = 1.0,
        b1: float = 1.0,
        b2: float = 1.0,
        a1: float = 0.2,
        a2: float = 0.3,
        a3: float = 0.1,
        k: float = 1.0,
        u: Annotated[float, ParameterSpec(adaptive=True, bounds=(0.0, MAX_DOSE))] = 0.0,
        w: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(
            label,
            visual=visual,
            E=E,
            A=A,
            N=N,
            U=U,
            s=s,
            rho=rho,
            alpha=alpha,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            d1=d1,
            r1=r1,
            r2=r2,
            b1=b1,
            b2=b2,
            a1=a1,
            a2=a2,
            a3=a3,
            k=k,
            u=u,
        )
        self.w = w
