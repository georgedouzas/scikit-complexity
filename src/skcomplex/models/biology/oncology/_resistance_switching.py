"""Sensitive cells and cells resistant by mutation or by a reversible phenotype switch, under a constant dose."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw
from ....specs import ParameterSpec, StateSpec, VisualSpec


class ResistanceSwitchingElement(Element):
    """Sensitive cells and cells resistant by mutation or by a reversible phenotype switch, under a constant dose.

    With the total `V = S + Rm + Rp`, the sensitive cells follow
    `dS/dt = r1 (1 - b V) S - (m1 + m2) S - (alpha_m + alpha_p) u S - a u S + m5 Rp`, the mutated cells
    `dRm/dt = r2 (1 - b V) Rm + m1 S + alpha_m u S - am u Rm`, and the switched cells
    `dRp/dt = r3 (1 - b V) Rp + m2 S + alpha_p u S - m5 Rp - ap u Rp`. The rates `m1` and `m2` are the spontaneous
    (acquired) resistance, `alpha_m` and `alpha_p` the resistance the drug induces, and `m5` the resensitization of
    switched cells.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        S:
            The initial sensitive cells.

        Rm:
            The initial cells resistant by mutation.

        Rp:
            The initial cells resistant by phenotype switching.

        r1:
            The growth rate of sensitive cells.

        r2:
            The growth rate of mutated cells.

        r3:
            The growth rate of switched cells.

        b:
            The reciprocal carrying capacity of the tumour.

        m1:
            The rate of spontaneous resistance by mutation.

        m2:
            The rate of spontaneous resistance by phenotype switching.

        alpha_m:
            The rate per unit of dose of resistance induced by mutation.

        alpha_p:
            The rate per unit of dose of resistance induced by phenotype switching.

        m5:
            The rate at which switched cells become sensitive again.

        a:
            The kill rate of sensitive cells per unit of dose.

        am:
            The kill rate of mutated cells per unit of dose.

        ap:
            The kill rate of switched cells per unit of dose.

        u:
            The effective dose.

    Examples:
        >>> from skcomplex.models.biology.oncology import ResistanceSwitchingElement
        >>> ResistanceSwitchingElement('tumour').alpha_m
        0.01
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        S: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        Rm: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        Rp: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        r3: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        m1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1e-06,
        m2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha_m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        alpha_p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        m5: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        am: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ap: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        u: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.S = S
        self.Rm = Rm
        self.Rp = Rp
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.b = b
        self.m1 = m1
        self.m2 = m2
        self.alpha_m = alpha_m
        self.alpha_p = alpha_p
        self.m5 = m5
        self.a = a
        self.am = am
        self.ap = ap
        self.u = u

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth, transitions, and kill of the sensitive, mutated, and switched cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        S, Rm, Rp, u = s.S, s.Rm, s.Rp, s.u
        room = 1 - s.b * (S + Rm + Rp)
        return [
            RateLaw(
                S,
                s.r1 * room * S - (s.m1 + s.m2) * S - (s.alpha_m + s.alpha_p) * u * S - s.a * u * S + s.m5 * Rp,
            ),
            RateLaw(Rm, s.r2 * room * Rm + s.m1 * S + s.alpha_m * u * S - s.am * u * Rm),
            RateLaw(Rp, s.r3 * room * Rp + s.m2 * S + s.alpha_p * u * S - s.m5 * Rp - s.ap * u * Rp),
        ]
