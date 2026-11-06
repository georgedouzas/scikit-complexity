"""Infected density of the nodes of one degree of a network, in heterogeneous mean field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class DegreeClassElement(Element):
    """Infected density of the nodes of one degree of a network, in heterogeneous mean field.

    A node of degree `k` recovers at rate 1 and is infected at rate `lam k Theta`, where
    `Theta = sum_k k P(k) rho_k / <k>` is the probability that an edge points to an infected node, a total over the
    degree classes, so `d rho_k / dt = -rho_k + lam k (1 - rho_k) Theta`. The classes are the members of a group, and
    the recovery and the infection are interactions of the group, which `heterogeneous_mean_field_sis` adds.

    Args:
        label:
            The label of the class or the group of classes.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        rho:
            The initial infected density.

        lam:
            The spreading rate, the ratio of the infection and recovery rates.

        k:
            The degree of the class.

        P:
            The probability `P(k)` of the degree.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.epidemiology import DegreeClassElement
        >>> GroupElement(
        ...     'network', element=DegreeClassElement, size=2, values={'k': [1.0, 2.0], 'P': [0.5, 0.5]}
        ... ).prepare().size_
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        P: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.lam = lam
        self.k = k
        self.P = P
