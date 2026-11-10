"""Off-lattice tumour growth of cells joined by springs, after Wang et al. (2024)."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet
from scipy.spatial import cKDTree

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import ContinuousSpace, DelaunayNeighborhood, DiscreteTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec

BINARY = FiniteEnumeratedSet([0, 1])
HOURS_PER_DAY = 24.0
HEXAGON_AREA = math.sqrt(3.0) / 2.0
ROW_SPACING = math.sqrt(3.0) / 2.0
DELAUNAY = DelaunayNeighborhood()
DAUGHTER_DISTANCE = 0.8
SPRING_CUTOFF = 1.5


class SpringCellView(AgentView):
    """View of a cell of the off-lattice tumour, with its states and parameters as attributes.

    `SpringCellElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    cancer: float
    age: float
    p0: float
    dmax: float
    g_age1: float
    g_age2: float
    tau: float
    p_psc: float
    lam: float
    s: float


class SpringCellElement(Element):
    """Cell of the off-lattice tumour of Wang et al. (2024), a healthy or a cancer cell with its age.

    Args:
        label:
            The label of the cell or the group.

        cancer:
            1 for a cancer cell, 0 for a healthy cell.

        age:
            The hours since the cell was born or last divided.

        p0:
            The probability per hour that a cancer cell on the edge of the tumour divides.

        dmax:
            The largest distance from the edge of the tumour at which a cancer cell divides.

        g_age1:
            The age in hours a cancer cell needs to divide before the switch.

        g_age2:
            The age in hours a cancer cell needs to divide after the switch.

        tau:
            The day of the switch of the growth phase.

        p_psc:
            The probability per hour that a cancer cell invades a healthy neighbour.

        lam:
            The mechanical interaction coefficient of the springs.

        s:
            The natural length of the springs, the diameter of a cell.

    Examples:
        >>> from skcomplex.systems.biology.oncology import SpringCellElement
        >>> SpringCellElement('cell', cancer=1).cancer
        1
    """

    def __init__(
        self,
        label: str,
        *,
        cancer: Annotated[float, StateSpec(domain=BINARY)] = 0,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        p0: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.03,
        dmax: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        g_age1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        g_age2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        p_psc: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.001,
        lam: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.2,
        s: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.cancer = cancer
        self.age = age
        self.p0 = p0
        self.dmax = dmax
        self.g_age1 = g_age1
        self.g_age2 = g_age2
        self.tau = tau
        self.p_psc = p_psc
        self.lam = lam
        self.s = s

    @classmethod
    def _get_view(cls) -> type[SpringCellView]:
        """Return the view of a cell, `SpringCellView`."""
        return SpringCellView


class TumourGrowthInteraction(Interaction):
    """Springs, divisions, and invasions of the cells of an off-lattice tumour, every hour.

    Every cell in turn ages by one hour and moves by Eq. 2.2 of Wang et al. (2024, arXiv 2409.19675), `r_i + lam sum_j
    r_ij / |r_ij| (s - |r_ij|)` over its neighbours `j` of the Delaunay triangulation, with `r_ij = r_i - r_j`, and a
    cell pushed out of the square leaves the tissue it observes. Springs longer than one and a half natural lengths,
    such as the long edges of the hull, are cut, as in Meineke et al. (2001). A cancer cell older than `g_age1` before
    the day `tau` and `g_age2` after it divides with the probability `p0 (1 - d / dmax)`, where `d` is its distance to
    the nearest healthy cell, the edge of the tumour, and splits into two cancer cells 0.8 natural spring lengths apart
    along a drawn direction. A cancer cell then invades a drawn healthy neighbour with the probability `p_psc`, which a
    young cancer cell replaces in its position. The division, the direction, the invasion, and the invaded neighbour
    are named draws, so runs record and replay them.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the variable group of cells.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TumourGrowthInteraction
        >>> TumourGrowthInteraction('growth', target='cells').target
        'cells'
    """

    law_classes = (RuleLaw,)
    draws = True

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label)
        self.target = target

    def _get_laws(self, namespace: object) -> list[BaseLaw]:
        """Return the rule that every cell runs."""
        return [RuleLaw(self._act, members='target')]

    def _act(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Age a cell, divide and invade if it is a cancer cell, and move it by its springs."""
        cell = cast('SpringCellView', agent)
        cell.age += 1.0
        neighbours = [cast('SpringCellView', item) for item in cell.get_neighbors(over=DELAUNAY)]
        if cell.cancer:
            self._divide(cell, system)
            self._invade(cell, neighbours, system)
        self._relax(cell, neighbours, system)

    @staticmethod
    def _get_edge_distance(cell: SpringCellView, system: SystemViewProtocol) -> float:
        """Return the distance of a cell to the nearest healthy cell, from a tree kept for the step."""
        storage = system.storage
        if storage.get('time') != system.time:
            cells = [cast('SpringCellView', item) for item in system.get_agents('target')]
            healthy = [item.location for item in cells if not item.cancer]
            storage['time'] = system.time
            storage['tree'] = cKDTree(np.asarray(healthy, dtype=float)) if healthy else None
        tree = storage['tree']
        if not isinstance(tree, cKDTree):
            return math.inf
        return float(tree.query(np.asarray(cell.location, dtype=float))[0])

    def _divide(self, cell: SpringCellView, system: SystemViewProtocol) -> None:
        """Divide a cancer cell old enough with the probability of its distance to the edge of the tumour."""
        needed = cell.g_age1 if system.time < cell.tau * HOURS_PER_DAY else cell.g_age2
        if cell.age < needed:
            return
        probability = max(0.0, cell.p0 * (1.0 - self._get_edge_distance(cell, system) / cell.dmax))
        if not system.draw('divide', 'bernoulli', p=probability):
            return
        angle = system.draw('direction', 'uniform', low=0.0, high=2.0 * math.pi)
        offset = DAUGHTER_DISTANCE / 2 * cell.s * np.array([math.cos(angle), math.sin(angle)])
        centre = np.asarray(cell.location, dtype=float)
        cell.age = 0.0
        cell.move_to(self._get_inside(centre - offset, system))
        cell.create(location=self._get_inside(centre + offset, system), cancer=1, age=0.0)

    @staticmethod
    def _invade(cell: SpringCellView, neighbours: list[SpringCellView], system: SystemViewProtocol) -> None:
        """Turn a drawn healthy neighbour into a young cancer cell with the probability `p_psc`."""
        healthy = [item for item in neighbours if not item.cancer]
        if not healthy or not system.draw('invade', 'bernoulli', p=cell.p_psc):
            return
        chosen = healthy[min(int(system.draw('invaded', 'uniform', low=0.0, high=len(healthy))), len(healthy) - 1)]
        chosen.cancer = 1
        chosen.age = 0.0

    def _relax(self, cell: SpringCellView, neighbours: list[SpringCellView], system: SystemViewProtocol) -> None:
        """Move a cell by the forces of the springs to its neighbours, Eq. 2.2."""
        if not neighbours:
            return
        position = np.asarray(cell.location, dtype=float)
        differences = position - np.asarray([item.location for item in neighbours], dtype=float)
        lengths = np.maximum(np.linalg.norm(differences, axis=1), 1e-9)
        pulls = np.where(lengths < SPRING_CUTOFF * cell.s, cell.s - lengths, 0.0)
        force = (differences / lengths[:, None] * pulls[:, None]).sum(axis=0)
        moved = position + cell.lam * force
        if np.any(moved < 0) or np.any(moved > np.asarray(system.dimensions, dtype=float)):
            cell.remove()
            return
        cell.move_to(tuple(float(item) for item in moved))

    @staticmethod
    def _get_inside(position: np.ndarray, system: SystemViewProtocol) -> tuple[float, ...]:
        """Return a position clipped into the square, a small margin inside its edges."""
        sides = np.asarray(system.dimensions, dtype=float)
        return tuple(float(item) for item in np.clip(position, 1e-6, sides - 1e-6))


def _get_tumour_area(cancer: np.ndarray) -> float:
    """Return the tumour area, the number of cancer cells times the area of a hexagonal cell."""
    return float(np.sum(cancer)) * HEXAGON_AREA


class OffLatticeTumourGrowthSystem(BaseSystem):
    """Biphasic off-lattice tumour growth of cells joined by springs, after Wang et al. (2024).

    The biphasic Voronoi cell-based model of Wang et al. (2024, arXiv 2409.19675, section 2.1.1): healthy cells start on
    a hexagonal layout of a square of `side` cell diameters with a cluster of cancer cells at its centre. Cell centres
    are joined by springs along the edges of their Delaunay triangulation, which relax by Eq. 2.2, and cancer cells
    divide and invade healthy neighbours by `TumourGrowthInteraction`. The age a cancer cell needs to divide switches
    from `g_age1` to `g_age2` at the day `tau`, so the tumour grows in two phases. Time runs in hours. The system has
    the variable group `cells` and the observable `tumour_area`, in squared cell diameters, the summary that ABC-SMC
    compares.

    Args:
        side:
            The side of the square in cell diameters.

        g_age1:
            The age in hours a cancer cell needs to divide before the switch.

        g_age2:
            The age in hours a cancer cell needs to divide after the switch.

        tau:
            The day of the switch.

        p0:
            The probability per hour that a cancer cell on the edge of the tumour divides.

        dmax:
            The largest distance from the edge of the tumour at which a cancer cell divides.

        p_psc:
            The probability per hour that a cancer cell invades a healthy neighbour.

        lam:
            The mechanical interaction coefficient of the springs.

        initial_radius:
            The radius in cell diameters of the cluster of cancer cells at the start.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OffLatticeTumourGrowthSystem
        >>> OffLatticeTumourGrowthSystem(side=6).prepare().system_.space.extent
        (6.0, 6.0)
    """

    field = 'biology/oncology'
    topics = ('tumour growth', 'agents', 'off-lattice', 'springs', 'delaunay', 'stochastic')
    references = ('Wang et al. 2024', 'Jenner et al. 2020', 'Meineke et al. 2001')
    until = 240.0
    observables = (ObservableSpec('tumour_area', group='cells', variable='cancer', reduce=_get_tumour_area),)

    def __init__(
        self,
        side: float = 12.0,
        *,
        g_age1: float = 20.0,
        g_age2: float = 5.0,
        tau: float = 10.0,
        p0: float = 0.03,
        dmax: float = 2.0,
        p_psc: float = 0.001,
        lam: float = 0.2,
        initial_radius: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.side = side
        self.g_age1 = g_age1
        self.g_age2 = g_age2
        self.tau = tau
        self.p0 = p0
        self.dmax = dmax
        self.p_psc = p_psc
        self.lam = lam
        self.initial_radius = initial_radius

    def _build_system(self) -> CompositeSystem:
        """Return the cells on a hexagonal layout with a cancer cluster at the centre."""
        side = float(self.side)
        rows, columns = np.arange(0.5, side, ROW_SPACING), np.arange(0.5, side - 0.5)
        positions = np.asarray(
            [(column + 0.5 * (index % 2), row) for index, row in enumerate(rows) for column in columns],
        )
        cancer = (np.linalg.norm(positions - side / 2, axis=1) <= self.initial_radius).astype(float)
        parameters = {
            'p0': self.p0,
            'dmax': self.dmax,
            'g_age1': self.g_age1,
            'g_age2': self.g_age2,
            'tau': self.tau,
            'p_psc': self.p_psc,
            'lam': self.lam,
        }
        cells = GroupElement(
            'cells',
            element=SpringCellElement,
            size=len(positions),
            values={'cancer': cancer, **parameters},
            location=[tuple(item) for item in positions],
            variable=True,
        )
        return CompositeSystem(
            self.label,
            elements=[cells],
            interactions=[TumourGrowthInteraction('growth', target='cells')],
            space=ContinuousSpace(extent=(side, side), boundary='open'),
            time=DiscreteTime(),
        )
