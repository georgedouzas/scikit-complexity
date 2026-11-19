"""Fractal distribution networks and the metabolic scaling they imply."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import ConstraintLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role


class BranchingVesselElement(Element):
    """Vessels of a space-filling branching network, with their sizes and levels.

    Args:
        label:
            The label of the vessel or of the group on the edges.

        r:
            The radius.

        l:
            The length.

        volume:
            The volume.

        terminal:
            Whether the vessel is a capillary.

        level:
            The level of the vessel, 1 for the aorta's daughters and `depth` for the capillaries.

        depth:
            The level of the capillaries.

        n:
            The branching ratio.

        r_c:
            The radius of a capillary.

        l_c:
            The length of a capillary.

        space:
            The space whose sites hold one member each, or `None` for one member.

        on:
            `'edges'` for one member per edge of a network space, or `None` for one per site.

    Examples:
        >>> from skcomplex.systems.biology.physiology import BranchingVesselElement
        >>> BranchingVesselElement('vessel', level=3.0).level
        3.0
    """

    def __init__(
        self,
        label: str,
        *,
        r: Annotated[float | Array, RealSet.closed_open(0, oo)] = 1.0,
        l: Annotated[float | Array, RealSet.closed_open(0, oo)] = 1.0,
        volume: Annotated[float | Array, RealSet.closed_open(0, oo)] = 1.0,
        terminal: Annotated[float | Array, RealSet.closed(0, 1)] = 0.0,
        level: Annotated[float | Array, RealSet.closed_open(0, oo)] = 1.0,
        depth: Annotated[float | Array, RealSet.closed_open(0, oo)] = 1.0,
        n: Annotated[float | Array, RealSet.open(1, oo)] = 2.0,
        r_c: Annotated[float | Array, RealSet.open(0, oo)] = 1.0,
        l_c: Annotated[float | Array, RealSet.open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
        on: str | None = None,
    ) -> None:
        super().__init__(label, space=space, on=on)
        self.r = r
        self.l = l
        self.volume = volume
        self.terminal = terminal
        self.level = level
        self.depth = depth
        self.n = n
        self.r_c = r_c
        self.l_c = l_c


class VesselSizingInteraction(Interaction):
    """Sizes of the vessels of a space-filling branching network, set by their levels.

    The sizes make the network minimize transport energy.

    Every vessel branches into `n` daughters. Area-preserving branching gives the radii the ratio `n^(-1/2)` between
    levels and space filling the lengths the ratio `n^(-1/3)`, and the capillaries at the last level `depth` have the
    same radius `r_c` and length `l_c` in every organism, so `r = r_c n^((depth - level) / 2)`, `l = l_c n^((depth -
    level) / 3)`, and the volume of the vessel is `pi r^2 l`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the vessels, whose values are `r`, `l`, `volume`, `terminal`, `level`, `depth`, `n`, `r_c`, and
            `l_c`.

    Examples:
        >>> from skcomplex.systems.biology.physiology import VesselSizingInteraction
        >>> VesselSizingInteraction('sizing', target='vessels').target
        'vessels'
    """

    law_classes = (ConstraintLaw,)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the radius, the length, the volume, and the capillary flag of the vessel.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            height = s.depth - s.level
            self.laws_ = [
                ConstraintLaw(s.r == s.r_c * s.n ** (height / 2), target=s.r),
                ConstraintLaw(s.l == s.l_c * s.n ** (height / 3), target=s.l),
                ConstraintLaw(s.volume == pi * s.r**2 * s.l, target=s.volume),
                ConstraintLaw(s.terminal == where(s.level > s.depth - 1 / 2, 1, 0), target=s.terminal),
            ]


class WestBrownEnquistSystem(BaseSystem):
    """The distribution network of West, Brown, and Enquist.

    It is a space-filling tree of invariant capillaries.

    The tree branches `n` ways at every level down to the capillaries at level `depth`. The metabolic rate is
    proportional to the number of capillaries, `n^depth`, and the body mass to the blood volume, which the
    area-preserving and space-filling vessels make `V = pi r_c^2 l_c n^depth (n^(depth / 3) - 1) / (n^(1 / 3) - 1)`,
    which grows as `n^(4 depth / 3)` for deep trees. So the metabolic rate scales as the mass to the power 3/4,
    Kleiber's law, which the exponent between consecutive depths approaches from below. A `sweep` over `depth`
    fits it. The system has the radii `vessels.r`, lengths `vessels.l`, and volumes `vessels.volume` on the edges of the
    tree and the observables `blood_volume` and `capillaries` of the catalog, which a run passes in `observables`.

    Args:
        depth:
            The level of the capillaries, the number of branchings.

        n:
            The branching ratio.

        r_c:
            The radius of a capillary.

        l_c:
            The length of a capillary.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.physiology import WestBrownEnquistSystem
        >>> WestBrownEnquistSystem(3).prepare().system_.elements[0].level.size
        14
    """

    field = 'biology/physiology'
    topics = ('allometry', 'kleiber law', 'fractal network', 'static', 'metabolic scaling')
    references = ('West, Brown, and Enquist 1997', 'Ribeiro and Netto 2024')
    observables = (
        Observable('blood_volume', group='vessels', variable='volume', reduce='sum'),
        Observable('capillaries', group='vessels', variable='terminal', reduce='sum'),
    )

    def __init__(
        self,
        depth: int = 6,
        *,
        n: int = 2,
        r_c: float = 1.0,
        l_c: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.depth = depth
        self.n = n
        self.r_c = r_c
        self.l_c = l_c

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the distribution network of West.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        graph = nx.balanced_tree(self.n, self.depth)
        space = NetworkSpace(graph).prepare()
        _, children = space.edges_[None]
        levels = nx.single_source_shortest_path_length(graph, 0)
        vessels = BranchingVesselElement(
            'vessels',
            level=np.array([levels[int(child)] for child in children], dtype=float),
            depth=float(self.depth),
            n=float(self.n),
            r_c=self.r_c,
            l_c=self.l_c,
            space=space,
            on='edges',
        )
        sizing = VesselSizingInteraction('sizing', target='vessels')
        self.system_ = System(self.label, elements=[vessels], interactions=[sizing], space=space)
