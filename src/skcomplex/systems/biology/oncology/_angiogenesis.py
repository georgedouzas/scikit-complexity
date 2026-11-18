"""A tumour that grows on oxygen and recruits blood vessels.

The tumour recruits the vessels by secreting growth factor.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw, UpdateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ...general.agents import BirthDeathInteraction, MovementInteraction

OXYGEN = 'tissue/tissue.oxygen'
VEGF = 'tissue/tissue.vegf'
VESSELS = 'cells/vessels'
TUMOUR = 'cells/tumour'
SPROUTS = 'cells/sprouts'
TISSUE = 'tissue/tissue'


class TissueElement(Element):
    """Tissue in which oxygen and the growth factor VEGF diffuse and decay.

    VEGF is the vascular endothelial growth factor.

    Both follow `u' = D Δu - lambda u`, and cells add or remove amounts at their sites: vessels supply oxygen, tumour
    cells consume it, and hypoxic tumour cells secrete VEGF. The system adds the diffusion and the decay as
    interactions.

    Args:
        label:
            The label of the tissue or the lattice field.

        oxygen:
            The initial oxygen.

        vegf:
            The initial VEGF.

        D_oxygen:
            The diffusion of oxygen.

        decay_oxygen:
            The decay rate of oxygen.

        D_vegf:
            The diffusion of VEGF.

        decay_vegf:
            The decay rate of VEGF.

        space:
            The lattice with one site of tissue per cell, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TissueElement
        >>> TissueElement('tissue').D_vegf
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        oxygen: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        vegf: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        D_oxygen: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        decay_oxygen: Annotated[float, RealSet.closed_open(0, oo)] = 0.05,
        D_vegf: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        decay_vegf: Annotated[float, RealSet.closed_open(0, oo)] = 0.005,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.oxygen = oxygen
        self.vegf = vegf
        self.D_oxygen = D_oxygen
        self.decay_oxygen = decay_oxygen
        self.D_vegf = D_vegf
        self.decay_vegf = decay_vegf


class SproutView(AgentView):
    """View of a sprout tip, with its parameters as attributes.

    `SproutElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    chemotaxis: float
    branching: float
    activation: float


class SproutElement(Element):
    """Endothelial tip cell that climbs the VEGF gradient.

    It leaves a new vessel segment behind it.

    Every step the tip moves to a neighbouring cell with a probability that grows with `exp(chemotaxis * VEGF /
    VEGF_here)`, so it senses the relative gradient, and it leaves a vessel segment where there was none. With
    probability `branching` the tip splits in two, the new tip standing where the tip was. Below the VEGF level
    `activation` the tip stays put. The system adds the vessel segment, the branching, and the step as member
    interactions.

    Args:
        label:
            The label of the agent or group.

        chemotaxis:
            How strongly VEGF draws the tip.

        branching:
            The probability per step of a new tip.

        activation:
            The VEGF above which the tip moves.

        values:
            Further values by name, such as the cells `x` and `y` of the members.

        size:
            The number of members, or `None` for one.

        space:
            The grid the members live on, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import SproutElement
        >>> SproutElement('sprouts', size=2).size
        2
    """

    def __init__(
        self,
        label: str,
        *,
        chemotaxis: Annotated[float, RealSet.closed_open(0, oo)] = 30.0,
        branching: Annotated[float, RealSet.closed_open(0, oo)] = 0.02,
        activation: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.chemotaxis = chemotaxis
        self.branching = branching
        self.activation = activation

    @classmethod
    def _get_view(cls) -> type[SproutView]:
        """Return the view of a sprout tip, `SproutView`."""
        return SproutView


class TumourCellView(AgentView):
    """View of a tumour cell, with its states and parameters as attributes.

    `TumourCellElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    hypoxic: float
    consumption: float
    proliferation: float
    division: float
    hypoxia: float
    secretion: float
    necrosis: float


class TumourCellElement(Element):
    """Tumour cell that consumes oxygen and divides when supplied.

    It secretes VEGF when hypoxic and dies without oxygen.

    Every step the cell takes up `consumption` oxygen at its site, or what is left. Above `proliferation` it divides
    with probability `division` into a free neighbouring cell. Below `hypoxia` it secretes `secretion` VEGF, and below
    `necrosis` it dies. The system adds the uptake, the hypoxia, the secretion, and the life cycle as member
    interactions.

    Args:
        label:
            The label of the agent or group.

        hypoxic:
            Whether the cell is hypoxic, `1`, or not, `0`.

        consumption:
            The oxygen a cell takes up per step.

        proliferation:
            The oxygen above which a cell divides.

        division:
            The probability per step of a division.

        hypoxia:
            The oxygen below which a cell secretes VEGF.

        secretion:
            The VEGF a hypoxic cell secretes per step.

        necrosis:
            The oxygen below which a cell dies.

        values:
            Further values by name, such as the cells `x` and `y` of the members.

        size:
            The number of members, or `None` for one.

        space:
            The grid the members live on, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TumourCellElement
        >>> TumourCellElement('tumour', size=3).size
        3
    """

    visual = VisualSpec('agents', {'color_by': 'hypoxic', 'color_range': (0.0, 1.0)})

    def __init__(
        self,
        label: str,
        *,
        hypoxic: Annotated[float, FiniteEnumeratedSet([0, 1])] = 0.0,
        consumption: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        proliferation: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        division: Annotated[float, RealSet.closed_open(0, oo)] = 0.2,
        hypoxia: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        secretion: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        necrosis: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.hypoxic = hypoxic
        self.consumption = consumption
        self.proliferation = proliferation
        self.division = division
        self.hypoxia = hypoxia
        self.secretion = secretion
        self.necrosis = necrosis

    @classmethod
    def _get_view(cls) -> type[TumourCellView]:
        """Return the view of a tumour cell, `TumourCellView`."""
        return TumourCellView


class VesselView(AgentView):
    """View of a vessel, with its parameters as attributes.

    `VesselElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    supply: float


class VesselElement(Element):
    """Segment of a blood vessel that supplies oxygen to its site every step.

    The supply is a member transfer of the system.

    Args:
        label:
            The label of the agent or group.

        supply:
            The oxygen supplied per step.

        values:
            Further values by name, such as the cells `x` and `y` of the members.

        size:
            The number of members, or `None` for one.

        space:
            The grid the members live on, or `None`.

    Examples:
        >>> from skcomplex.systems.biology.oncology import VesselElement
        >>> VesselElement('vessels', size=2).size
        2
    """

    def __init__(
        self,
        label: str,
        *,
        supply: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.supply = supply

    @classmethod
    def _get_view(cls) -> type[VesselView]:
        """Return the view of a vessel, `VesselView`."""
        return VesselView


class TumourAngiogenesisSystem(BaseSystem):
    """A tumour that outgrows its oxygen supply and recruits new blood vessels.

    The new vessels grow from a parent vessel.

    The scales are coupled: tumour cells and vessel tips are agents on a grid, and oxygen and VEGF are fields on a
    lattice of the same shape. The tumour starts far from the parent vessel along the left edge, so its cells become
    hypoxic and secrete VEGF, which draws tips out of the vessel. The new vessels bring oxygen and the tumour grows. The
    system has the agents `cells/tumour`, `cells/vessels`, and `cells/sprouts`, and the fields `tissue/tissue.oxygen`
    and `tissue/tissue.vegf`.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        n_sprouts:
            The number of tips on the parent vessel.

        radius:
            The radius of the initial tumour, centred three quarters of the way to the right.

        chemotaxis:
            How strongly VEGF draws the tips.

        random_state:
            The seed or generator of the positions of the tips.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TumourAngiogenesisSystem
        >>> [group.label for group in TumourAngiogenesisSystem().prepare().system_.systems[0].elements]
        ['tumour', 'vessels', 'sprouts']
    """

    field = 'biology/oncology'
    topics = ('agents', 'field', 'multiscale', 'angiogenesis', 'co-simulation')
    references = ('Carmichael and Hadžikadić 2019', 'Anderson and Chaplain 1998')
    until = 120
    dt = 1

    def __init__(
        self,
        width: int = 40,
        height: int = 25,
        *,
        n_sprouts: int = 5,
        radius: int = 2,
        chemotaxis: float = 30.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.width = width
        self.height = height
        self.n_sprouts = n_sprouts
        self.radius = radius
        self.chemotaxis = chemotaxis
        self.random_state = random_state

    @staticmethod
    def _get_oxygen_decay(namespace: SimpleNamespace) -> Expression:
        """Return the decay `-decay_oxygen oxygen` of oxygen."""
        rate: Expression = -namespace.target.decay_oxygen * namespace.target.oxygen
        return rate

    @staticmethod
    def _get_oxygen_diffusion(namespace: SimpleNamespace) -> Expression:
        """Return the diffusion coefficient `D_oxygen` of oxygen."""
        coefficient: Expression = namespace.target.D_oxygen
        return coefficient

    @staticmethod
    def _get_vegf_decay(namespace: SimpleNamespace) -> Expression:
        """Return the decay `-decay_vegf vegf` of VEGF."""
        rate: Expression = -namespace.target.decay_vegf * namespace.target.vegf
        return rate

    @staticmethod
    def _get_vegf_diffusion(namespace: SimpleNamespace) -> Expression:
        """Return the diffusion coefficient `D_vegf` of VEGF."""
        coefficient: Expression = namespace.target.D_vegf
        return coefficient

    @staticmethod
    def _get_uptake(cell: AgentViewProtocol, site: object, system: SystemViewProtocol) -> float:
        """Return minus the oxygen a tumour cell takes up at its site.

        The uptake is `consumption` or what is left.
        """
        if not isinstance(cell, TumourCellView):
            return 0.0
        return -min(cell.consumption, system.get_field(OXYGEN, site))

    @staticmethod
    def _get_hypoxic(cell: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return whether the oxygen at the site of a tumour cell is below `hypoxia`."""
        if not isinstance(cell, TumourCellView):
            return 0.0
        return float(system.get_field(OXYGEN, cell.location) < cell.hypoxia)

    @staticmethod
    def _check_hypoxia(cell: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Mark whether the oxygen at the site of a tumour cell is below `hypoxia`."""
        if isinstance(cell, TumourCellView):
            cell.hypoxic = TumourAngiogenesisSystem._get_hypoxic(cell, system)

    @staticmethod
    def _secretes(cell: AgentViewProtocol, site: object, system: SystemViewProtocol) -> bool:
        """Return whether a living tumour cell is hypoxic.

        A hypoxic cell secretes VEGF.
        """
        if not isinstance(cell, TumourCellView):
            return False
        return bool(cell.hypoxic) and system.get_field(OXYGEN, site) >= cell.necrosis

    @staticmethod
    def _get_secretion(cell: AgentViewProtocol, site: object, system: SystemViewProtocol) -> float:
        """Return the VEGF `secretion` of a hypoxic tumour cell."""
        del site, system
        return float(cell.secretion) if isinstance(cell, TumourCellView) else 0.0

    @staticmethod
    def _dies(cell: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether the oxygen at the site of a tumour cell is below `necrosis`.

        Below it the cell dies.
        """
        return isinstance(cell, TumourCellView) and system.get_field(OXYGEN, cell.location) < cell.necrosis

    @staticmethod
    def _get_division(cell: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability `division` that a supplied tumour cell divides.

        A cell is supplied above `proliferation` oxygen.
        """
        if not isinstance(cell, TumourCellView) or cell.hypoxic:
            return 0.0
        return float(cell.division) if system.get_field(OXYGEN, cell.location) > cell.proliferation else 0.0

    @staticmethod
    def _get_supply(vessel: AgentViewProtocol, site: object, system: SystemViewProtocol) -> float:
        """Return the oxygen `supply` of a vessel segment."""
        del site, system
        return float(vessel.supply) if isinstance(vessel, VesselView) else 0.0

    @staticmethod
    def _is_active(tip: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether the VEGF at a tip reaches `activation`.

        An activated tip moves.
        """
        return isinstance(tip, SproutView) and system.get_field(VEGF, tip.location) >= tip.activation

    def _lays(self, tip: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether an active tip stands where there is no vessel segment.

        There the tip leaves one.
        """
        return self._is_active(tip, system) and not tip.get_agents_here(VESSELS)

    def _get_branching(self, tip: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability `branching` that an active tip splits in two."""
        return float(tip.branching) if isinstance(tip, SproutView) and self._is_active(tip, system) else 0.0

    @staticmethod
    def _get_chemotaxis(tip: AgentViewProtocol, system: SystemViewProtocol) -> object:
        """Return a neighbouring cell drawn by the VEGF gradient.

        The weight of a cell grows with `exp(chemotaxis * VEGF / VEGF_here)`.
        """
        if not isinstance(tip, SproutView):
            return None
        level = system.get_field(VEGF, tip.location)
        cells = cast('list[tuple[int, int]]', list(tip.get_neighbor_cells()))
        levels = np.array([system.get_field(VEGF, cell) for cell in cells])
        weights = np.exp(tip.chemotaxis * (levels - levels.max()) / level)
        return cells[int(system.rng.choice(len(cells), p=weights / weights.sum()))]

    def _get_tissue(self) -> list[Interaction]:
        """Return the diffusion and the decay of oxygen and VEGF in the tissue."""
        return [
            Interaction(
                'oxygen_diffusion',
                roles={'target': 'tissue'},
                laws=lambda s: [RateLaw(s.target.oxygen, self._get_oxygen_diffusion(s) * laplacian(s.target.oxygen))],
            ),
            Interaction(
                'oxygen_decay',
                roles={'target': 'tissue'},
                laws=lambda s: [RateLaw(s.target.oxygen, self._get_oxygen_decay(s))],
            ),
            Interaction(
                'vegf_diffusion',
                roles={'target': 'tissue'},
                laws=lambda s: [RateLaw(s.target.vegf, self._get_vegf_diffusion(s) * laplacian(s.target.vegf))],
            ),
            Interaction(
                'vegf_decay',
                roles={'target': 'tissue'},
                laws=lambda s: [RateLaw(s.target.vegf, self._get_vegf_decay(s))],
            ),
        ]

    def _take_up(self, cell: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Let a tumour cell take up the oxygen of its site."""
        if cell.alive:
            site = cell.location_key
            system.deposit(OXYGEN, site, self._get_uptake(cell, site, system))

    def _secrete(self, cell: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Let a hypoxic tumour cell secrete VEGF into its site."""
        site = cell.location_key
        if cell.alive and self._secretes(cell, site, system):
            system.deposit(VEGF, site, self._get_secretion(cell, site, system))

    def _supply(self, vessel: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Let a vessel segment supply oxygen to its site."""
        if vessel.alive:
            site = vessel.location_key
            system.deposit(OXYGEN, site, self._get_supply(vessel, site, system))

    @staticmethod
    def _get_rule(label: str, group: str, rule: Callable[[AgentViewProtocol, SystemViewProtocol], None]) -> Interaction:
        """Return the interaction that runs a rule on every member of a group, in its activation."""
        return Interaction(label, roles={'target': group}, laws=lambda s: [UpdateLaw(s.target, rule)])

    def _get_cells(self) -> list[Interaction]:
        """Return the member interactions of the tumour cells, vessel segments, and tips."""
        return [
            self._get_rule('uptake', TUMOUR, self._take_up),
            self._get_rule('hypoxia', TUMOUR, self._check_hypoxia),
            self._get_rule('secretion', TUMOUR, self._secrete),
            BirthDeathInteraction(
                'tumour_life',
                target=TUMOUR,
                death=self._dies,
                birth=self._get_division,
                placement='neighbor',
                schedule='member',
            ),
            self._get_rule('supply', VESSELS, self._supply),
            BirthDeathInteraction(
                'laying',
                target=SPROUTS,
                group=VESSELS,
                birth=self._lays,
                schedule='member',
            ),
            BirthDeathInteraction('branching', target=SPROUTS, birth=self._get_branching, schedule='member'),
            MovementInteraction(
                'chemotaxis',
                target=SPROUTS,
                step=self._get_chemotaxis,
                condition=self._is_active,
                schedule='member',
            ),
        ]

    @staticmethod
    def _get_places(cells: Sequence[tuple[int, int]]) -> dict[str, object]:
        """Return the cells of the members of a group as their values `x` and `y`."""
        return {'x': np.array([cell[0] for cell in cells]), 'y': np.array([cell[1] for cell in cells])}

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a tumour.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        center = (3 * self.width // 4, self.height // 2)
        tumour = [
            (x, y)
            for x in range(center[0] - self.radius, center[0] + self.radius + 1)
            for y in range(center[1] - self.radius, center[1] + self.radius + 1)
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= self.radius**2
        ]
        tips = sorted(int(row) for row in rng.choice(self.height, size=self.n_sprouts, replace=False))
        grid = LatticeSpace((self.width, self.height), boundary='fixed')
        groups = [
            TumourCellElement('tumour', values=self._get_places(tumour), size=len(tumour), space=grid),
            VesselElement(
                'vessels',
                values=self._get_places([(0, row) for row in range(self.height)]),
                size=self.height,
                space=grid,
            ),
            SproutElement(
                'sprouts',
                chemotaxis=self.chemotaxis,
                values=self._get_places([(0, row) for row in tips]),
                size=self.n_sprouts,
                space=grid,
            ),
        ]
        tissue = TissueElement('tissue', space=LatticeSpace((self.width, self.height), boundary='reflecting'))
        self.system_ = System(
            self.label,
            systems=[
                System('cells', elements=groups, space=grid),
                System('tissue', elements=[tissue], interactions=self._get_tissue()),
            ],
            interactions=self._get_cells(),
        )
