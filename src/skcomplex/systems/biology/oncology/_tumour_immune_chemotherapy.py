"""Immune, tumour, and normal cells that compete and that a drug kills, after de Pillis and Radunskaya."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class TumourImmuneChemotherapyElement(Element):
    """Immune, tumour, and normal cells that compete and that a drug kills, after de Pillis and Radunskaya.

    The immune cells follow `dE/dt = s + rho E A / (alpha + A) - d1 E - c1 E A - a1 (1 - exp(-U)) E`, the tumour
    `dA/dt = r1 A (1 - b1 A) - c2 E A - c3 N A - a2 (1 - exp(-U)) A`, the normal cells
    `dN/dt = r2 N (1 - b2 N) - c4 N A - a3 (1 - exp(-U)) N`, and the drug `dU/dt = u - k U` under a constant infusion
    `u`.

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
            The constant infusion rate of the drug.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TumourImmuneChemotherapyElement
        >>> TumourImmuneChemotherapyElement('patient', u=1.0).u
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        E: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        U: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.33,
        rho: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        c3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        c4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        d1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.E = E
        self.A = A
        self.N = N
        self.U = U
        self.s = s
        self.rho = rho
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.d1 = d1
        self.r1 = r1
        self.r2 = r2
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.k = k
        self.u = u

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the balances of the immune, tumour, and normal cells and of the drug.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        E, A, N, U = s.E, s.A, s.N, s.U
        kill = 1 - exp(-U)
        return [
            RateLaw(E, s.s + s.rho * E * A / (s.alpha + A) - s.d1 * E - s.c1 * E * A - s.a1 * kill * E),
            RateLaw(A, s.r1 * A * (1 - s.b1 * A) - s.c2 * E * A - s.c3 * N * A - s.a2 * kill * A),
            RateLaw(N, s.r2 * N * (1 - s.b2 * N) - s.c4 * N * A - s.a3 * kill * N),
            RateLaw(U, s.u - s.k * U),
        ]
