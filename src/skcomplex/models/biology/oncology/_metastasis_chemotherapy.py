"""Normal and tumour cells at a primary and a secondary site, each with its drug, and a delayed migration."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class MetastasisChemotherapyElement(Element):
    """Normal and tumour cells at a primary and a secondary site, each with its drug, and a delayed migration.

    At each site the normal and tumour cells grow logistically, compete, and a drug kills them with a saturating
    response, `a N U / (bc + N)`. The drug at a site follows `dU/dt = u - (k + g1 N / (bc + N) + g2 A / (bc + A)) U`.
    Tumour cells leave the primary site at the rate `delta`, and a fraction `epsilon` of them reaches the secondary site
    after the delay `tau`, `dAs/dt = ... + epsilon delta Ap(t - tau)`. Before the start the primary tumour keeps its
    initial value.

    The element holds the states and parameters and has no laws of its own: the interactions of `delayed_metastasis`
    read its parameters.

    Args:
        label:
            The label of the patient.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        Np:
            The initial normal cells at the primary site.

        Ap:
            The initial tumour cells at the primary site.

        Up:
            The initial drug concentration at the primary site.

        Ns:
            The initial normal cells at the secondary site.

        As:
            The initial tumour cells at the secondary site.

        Us:
            The initial drug concentration at the secondary site.

        r1:
            The growth rate of normal cells at the primary site.

        r2:
            The growth rate of tumour cells at the primary site.

        r3:
            The growth rate of normal cells at the secondary site.

        r4:
            The growth rate of tumour cells at the secondary site.

        b1:
            The reciprocal carrying capacity of normal cells at the primary site.

        b2:
            The reciprocal carrying capacity of tumour cells at the primary site.

        b3:
            The reciprocal carrying capacity of normal cells at the secondary site.

        b4:
            The reciprocal carrying capacity of tumour cells at the secondary site.

        c1:
            The rate at which the tumour competes with normal cells at the primary site.

        c2:
            The rate at which normal cells compete with the tumour at the primary site.

        c3:
            The rate at which the tumour competes with normal cells at the secondary site.

        c4:
            The rate at which normal cells compete with the tumour at the secondary site.

        a1:
            The kill rate of the drug on normal cells at the primary site.

        a2:
            The kill rate of the drug on tumour cells at the primary site.

        a3:
            The kill rate of the drug on normal cells at the secondary site.

        a4:
            The kill rate of the drug on tumour cells at the secondary site.

        g1:
            The rate at which the drug combines with normal cells at the primary site.

        g2:
            The rate at which the drug combines with tumour cells at the primary site.

        g3:
            The rate at which the drug combines with normal cells at the secondary site.

        g4:
            The rate at which the drug combines with tumour cells at the secondary site.

        bc:
            The half saturation of the kill and of the binding.

        k:
            The elimination rate of the drug at both sites, `dU1` and `dU2` in the book.

        delta:
            The rate at which tumour cells leave the primary site.

        epsilon:
            The fraction of leaving cells that reaches the secondary site.

        tau:
            The delay of the migration.

        up:
            The infusion rate of the drug at the primary site.

        us:
            The infusion rate of the drug at the secondary site.

    Examples:
        >>> from skcomplex.models.biology.oncology import MetastasisChemotherapyElement
        >>> MetastasisChemotherapyElement('patient').epsilon
        0.0001
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        Np: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1460.0,
        Ap: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 165.0,
        Up: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Ns: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1750.0,
        As: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Us: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        r4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 12.0,
        b1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00068,
        b2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00047,
        b3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00057,
        b4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.00043,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0075,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        c3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.002,
        c4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        a1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0008,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        a3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0006,
        a4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        g1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0024,
        g2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.24,
        g3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0018,
        g4: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        bc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        up: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        us: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.Np = Np
        self.Ap = Ap
        self.Up = Up
        self.Ns = Ns
        self.As = As
        self.Us = Us
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.g4 = g4
        self.bc = bc
        self.k = k
        self.delta = delta
        self.epsilon = epsilon
        self.tau = tau
        self.up = up
        self.us = us
