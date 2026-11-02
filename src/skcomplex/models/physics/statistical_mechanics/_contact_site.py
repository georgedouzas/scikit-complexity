"""Site of the contact process, which infects a neighbour at the rate `rate / 4` and recovers at rate 1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, LocalRuleLaw
from ....core.spaces import VonNeumannNeighborhood
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ....types import Array


class ContactSiteElement(Element):
    """Site of the contact process, which infects a neighbour at the rate `rate / 4` and recovers at rate 1.

    One step is a time slice `tau`, in which an infected site recovers with probability `1 - exp(-tau)` and a
    healthy site is infected with probability `1 - exp(-rate tau f)`, where `f` is the fraction of its infected
    neighbours.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        infected:
            The initial state, 1 for infected and 0 for healthy.

        rate:
            The infection rate.

        tau:
            The time slice of one step.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ContactSiteElement
        >>> ContactSiteElement('site').rate
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        infected: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.infected = infected
        self.rate = rate
        self.tau = tau

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the stochastic infection and recovery.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            LocalRuleLaw(s.infected, VonNeumannNeighborhood(center=True), ContactSiteElement._spread, stochastic=True),
        ]

    @staticmethod
    def _spread(infected: Array, *around: Array, rate: float | Array, tau: float | Array, random: Array) -> Array:
        """Return the next infections of the contact process over a time slice `tau`."""
        share = sum(around) / len(around)
        infects = random < -np.expm1(-rate * tau * share)
        recovers = random < -np.expm1(-tau)
        return np.where(infected == 1, np.where(recovers, 0.0, 1.0), np.where(infects, 1.0, 0.0))
