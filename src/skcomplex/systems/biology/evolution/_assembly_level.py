"""Objects of one assembly index, discovered from the objects one assembly step below them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import shift_member
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class AssemblyLevelElement(Element):
    """Objects of one assembly index, discovered from the objects one assembly step below them.

    In a group ordered by assembly index, the objects of index `a + 1` grow as `dN_{a+1}/dt = k (N_a)^alpha`, Eq. (3)
    of Sharma et al. (2022), where the first member reads the constant pool `N_0` of building blocks. The selectivity
    `alpha = 1` reuses every object built before, and `alpha < 1` only a subset, the signature of selection.

    Args:
        label:
            The label of the level or of the group of levels.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial number of objects.

        k:
            The rate of discovery.

        alpha:
            The selectivity of reuse.

        N0:
            The constant number of building blocks below the first level.

    Examples:
        >>> from skcomplex.systems.biology.evolution import AssemblyLevelElement
        >>> AssemblyLevelElement('levels', alpha=0.5).alpha
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        N0: float = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.k = k
        self.alpha = alpha
        self.N0 = N0

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the discovery of the objects of this level from the level below.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        below = shift_member(s.N, -1, boundary='fixed', fill=self.N0)
        return [RateLaw(s.N, s.k * below**s.alpha)]
