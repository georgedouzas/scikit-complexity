"""Cell cycle and invasion of melanoma cells on a triangular lattice, after Carr et al. (2021)."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.laws.utils import shift_member
from ....core.components.system import BaseSystem, System
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec

BINARY = FiniteEnumeratedSet([0, 1])
NEIGHBOURS = 6
PHASES = ('g1', 'es', 'sg2m')
MOVES = ('Mr', 'My', 'Mg')
STEADY_SHARES = (25.0, 5.9, 12.5)


class CellSiteElement(Element):
    """Site of the triangular lattice of the cell invasion model of Carr et al. (2021).

    A site is vacant or holds one cell in the phase G1 (red), early S (yellow), or S/G2/M (green) of the fluorescent
    cell cycle indicator, with the rates of the cell cycle and of moves that `CellCycleInvasionInteraction` reads, and
    `open_k`, which is 0 where its neighbour `k` lies outside the lattice.

    Args:
        label:
            The label of the site or the group.

        g1:
            1 for a cell in G1.

        es:
            1 for a cell in early S.

        sg2m:
            1 for a cell in S/G2/M.

        Rr:
            The rate per hour from G1 to early S.

        Ry:
            The rate per hour from early S to S/G2/M.

        Rg:
            The rate per hour of division.

        Mr:
            The rate per hour of moves of a cell in G1.

        My:
            The rate per hour of moves of a cell in early S.

        Mg:
            The rate per hour of moves of a cell in S/G2/M.

        open_0:
            1 when the neighbour at the first offset lies inside the lattice, and likewise for `open_1` to `open_5`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import CellSiteElement
        >>> CellSiteElement('site', g1=1).g1
        1
    """

    def __init__(
        self,
        label: str,
        *,
        g1: Annotated[float, StateSpec(domain=BINARY)] = 0,
        es: Annotated[float, StateSpec(domain=BINARY)] = 0,
        sg2m: Annotated[float, StateSpec(domain=BINARY)] = 0,
        Rr: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.04,
        Ry: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.17,
        Rg: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.08,
        Mr: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        My: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        Mg: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        open_0: Annotated[float, ParameterSpec(domain=BINARY)] = 1,
        open_1: Annotated[float, ParameterSpec(domain=BINARY)] = 1,
        open_2: Annotated[float, ParameterSpec(domain=BINARY)] = 1,
        open_3: Annotated[float, ParameterSpec(domain=BINARY)] = 1,
        open_4: Annotated[float, ParameterSpec(domain=BINARY)] = 1,
        open_5: Annotated[float, ParameterSpec(domain=BINARY)] = 1,
    ) -> None:
        super().__init__(label)
        self.g1 = g1
        self.es = es
        self.sg2m = sg2m
        self.Rr = Rr
        self.Ry = Ry
        self.Rg = Rg
        self.Mr = Mr
        self.My = My
        self.Mg = Mg
        self.open_0 = open_0
        self.open_1 = open_1
        self.open_2 = open_2
        self.open_3 = open_3
        self.open_4 = open_4
        self.open_5 = open_5


class CellCycleInvasionInteraction(Interaction):
    """Cell cycle, divisions, and moves of the cells on the sites of a triangular lattice.

    A cell in G1 enters early S at the rate `Rr` while a neighbouring site is vacant, a cell in early S enters S/G2/M at
    the rate `Ry`, and a cell in S/G2/M divides at the rate `Rg` into two cells in G1, the daughter on a neighbouring
    site chosen at random, which aborts when that site is occupied. A cell of each phase moves to a neighbouring site
    chosen at random at the rates `Mr`, `My`, and `Mg`, which aborts when it is occupied. Every site reads its six
    neighbours at member offsets along the group order, the steps of a triangular lattice in axial coordinates.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of sites.

        offsets:
            The six member offsets of the neighbours along the group order.

    Examples:
        >>> from skcomplex.systems.biology.oncology import CellCycleInvasionInteraction
        >>> CellCycleInvasionInteraction('cycle', target='cells', offsets=(1, -1, 8, -8, -7, 7)).offsets[2]
        8
    """

    law_classes = (JumpLaw,)

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], offsets: tuple[int, ...]) -> None:
        super().__init__(label)
        self.target = target
        self.offsets = offsets

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the jumps of the cell cycle, the divisions, and the moves of every site."""
        s = namespace.target
        cells = (s.g1, s.es, s.sg2m)
        opens = (s.open_0, s.open_1, s.open_2, s.open_3, s.open_4, s.open_5)
        free = [
            opened * (1 - sum(shift_member(cell, offset, boundary='fixed') for cell in cells))
            for opened, offset in zip(opens, self.offsets, strict=True)
        ]
        blocked: Expression = np.prod([1 - item for item in free])
        laws: list[BaseLaw] = [
            JumpLaw(s.Rr * s.g1 * (1 - blocked), {s.g1: -1, s.es: 1}),
            JumpLaw(s.Ry * s.es, {s.es: -1, s.sg2m: 1}),
        ]
        for offset, vacant in zip(self.offsets, free, strict=True):
            daughter = shift_member(s.g1, offset, boundary='fixed')
            laws.append(JumpLaw(s.Rg / NEIGHBOURS * s.sg2m * vacant, {s.sg2m: -1, s.g1: 1, daughter: 1}))
            for cell, rate in zip(cells, (s.Mr, s.My, s.Mg), strict=True):
                moved = shift_member(cell, offset, boundary='fixed')
                laws.append(JumpLaw(rate / NEIGHBOURS * cell * vacant, {cell: -1, moved: 1}))
        return laws


class CellCycleInvasionSystem(BaseSystem):
    """Cell invasion with the fluorescent cell cycle, after Carr et al. (2021).

    Melanoma cells on a triangular lattice pass through G1, early S, and S/G2/M, divide into two cells in G1 next to a
    vacancy, and move to vacant neighbouring sites, run by the Gillespie algorithm, as Wang et al. (2024, arXiv
    2409.19675, section 2.1.2) use it. A cell leaves G1 only while a neighbouring site is vacant, so a full lattice
    keeps its cells in G1. The lattice is a rhombus of `height` rows of `width` sites in axial coordinates, filled at a
    density, with a vacant scratch of the columns between two fractions of the width. The system has the group `cells`
    of sites and the observables `g1_cells`, `es_cells`, and `sg2m_cells`, the number of cells in every phase, the
    summaries that ABC-SMC compares.

    Args:
        width:
            The number of sites per row.

        height:
            The number of rows.

        Rr:
            The rate per hour from G1 to early S.

        Ry:
            The rate per hour from early S to S/G2/M.

        Rg:
            The rate per hour of division.

        Mr:
            The rate per hour of moves of a cell in G1.

        My:
            The rate per hour of moves of a cell in early S.

        Mg:
            The rate per hour of moves of a cell in S/G2/M.

        density:
            The share of the sites outside the scratch that hold a cell at the start.

        scratch:
            The fractions of the width between which the columns start vacant, or `None` for no scratch.

        random_state:
            The seed or generator of the initial cells and their phases.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import CellCycleInvasionSystem
        >>> CellCycleInvasionSystem(8, 4).prepare().system_.elements[0].prepare().size_
        32
    """

    field = 'biology/oncology'
    topics = ('cell cycle', 'invasion', 'lattice', 'stochastic', 'scratch assay')
    references = ('Carr et al. 2021', 'Simpson et al. 2018', 'Wang et al. 2024')
    until = 48.0
    dt = 1.0
    observables = tuple(
        ObservableSpec(f'{phase}_cells', group='cells', variable=phase, reduce='sum') for phase in PHASES
    )

    def __init__(
        self,
        width: int = 24,
        height: int = 12,
        *,
        Rr: float = 0.04,
        Ry: float = 0.17,
        Rg: float = 0.08,
        Mr: float = 0.25,
        My: float = 0.25,
        Mg: float = 0.25,
        density: float = 1.0,
        scratch: tuple[float, float] | None = (0.375, 0.625),
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.Rr = Rr
        self.Ry = Ry
        self.Rg = Rg
        self.Mr = Mr
        self.My = My
        self.Mg = Mg
        self.density = density
        self.scratch = scratch
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the sites of the lattice with the initial monolayer and its scratch."""
        width, height = self.width, self.height
        rows, columns = np.divmod(np.arange(width * height), width)
        steps = ((0, 1), (0, -1), (1, 0), (-1, 0), (-1, 1), (1, -1))
        opens = {
            f'open_{index}': (
                (rows + row >= 0) & (rows + row < height) & (columns + column >= 0) & (columns + column < width)
            ).astype(float)
            for index, (row, column) in enumerate(steps)
        }
        rng = np.random.default_rng(self.random_state)
        occupied = rng.random(width * height) < self.density
        if self.scratch is not None:
            low, high = self.scratch
            occupied &= ~((columns >= low * width) & (columns < high * width))
        shares = np.asarray(STEADY_SHARES) / sum(STEADY_SHARES)
        phases = rng.choice(len(PHASES), size=width * height, p=shares)
        values: dict[str, object] = {
            phase: (occupied & (phases == index)).astype(float) for index, phase in enumerate(PHASES)
        }
        values.update(opens)
        values.update({'Rr': self.Rr, 'Ry': self.Ry, 'Rg': self.Rg, 'Mr': self.Mr, 'My': self.My, 'Mg': self.Mg})
        offsets = tuple(row * width + column for row, column in steps)
        cells = GroupElement('cells', element=CellSiteElement, size=width * height, values=values)
        cycle = CellCycleInvasionInteraction('cycle', target='cells', offsets=offsets)
        return System(self.label, elements=[cells], interactions=[cycle])
