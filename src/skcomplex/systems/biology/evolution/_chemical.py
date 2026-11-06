"""Molecular species of a pool whose relative population grows with the populations of its catalysts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import both, minimum, mod, neighbors, total, uniform, where
from ....core.specs import ParameterSpec, StateSpec, TopologyRequestSpec, VisualSpec


class ChemicalElement(Element):
    """Molecular species of a pool whose relative population grows with the populations of its catalysts.

    The populations follow the catalytic network: every species gains `h` times the populations of the species that
    catalyse it, and the populations are normalised, so they approach the Perron eigenvector of the network. Every
    `period` steps the least populated species, with ties broken at random, is replaced by a new species with `k`
    random catalysts, and every species catalyses the new one with probability `p`.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The relative population.

        tie:
            The random key that breaks ties between the least populated species.

        fresh:
            Whether the species appeared at the previous graph update.

        h:
            The growth of a step from the catalysts.

        p:
            The probability of every catalytic link of a new species.

        period:
            The number of steps between two graph updates.

        k:
            The number of catalysts of a new species.

    Examples:
        >>> from skcomplex.systems.biology.evolution import ChemicalElement
        >>> ChemicalElement('species').period
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        tie: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        fresh: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1.0,
        h: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        k: int = 1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.tie = tie
        self.fresh = fresh
        self.h = h
        self.p = p
        self.period = period
        self.k = k

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the catalytic growth, and the replacement of the least populated species at every graph update.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        grown = s.x + s.h * neighbors(s.other.x) + 1e-6
        key = s.x + 1e-9 * s.tie
        update = mod(s.t, s.period) > s.period - 3 / 2
        least = where(both(key <= minimum(key), update), 1, 0)
        newcomer = where(total(s.fresh) > 1 / 2, 1, 0) * (1 - s.fresh)

        def fresh(node: SimpleNamespace) -> object:
            """Return the weight of a candidate catalysed species, 1 for the new species and 0 otherwise."""
            return node.fresh

        return [
            UpdateLaw(s.x, grown / total(grown)),
            UpdateLaw(s.tie, uniform('tie')),
            UpdateLaw(s.fresh, 0),
            UpdateLaw(TopologyRequestSpec('add-node', edges=self.k), least),
            UpdateLaw(TopologyRequestSpec('remove-node'), least),
            UpdateLaw(TopologyRequestSpec('add-edge', toward=fresh), s.p * newcomer),
        ]
