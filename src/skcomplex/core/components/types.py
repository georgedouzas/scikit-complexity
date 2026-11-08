"""Define the types of the core.

They are the targets of laws, the pairings and rules of matches, and the protocols.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, MutableMapping, Sequence
from types import SimpleNamespace
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement
from shapely import Polygon

from ..specs import (
    BatchMatchingSpec,
    ContinuousMatchingSpec,
    IterativeMatchingSpec,
    PartnerSpec,
    PartnerSwitchSpec,
    QueueSpec,
    RandomSearchSpec,
    StableMatchingSpec,
)
from ..types import Array
from .spaces import LatticeNeighborhood, MemoryNeighborhood

type Target = Expression | FreeModuleElement
type Process = str | tuple[str, ...] | Expression | FreeModuleElement | None
type Feasibility = Callable[[AgentViewProtocol, AgentViewProtocol], bool]
type Pairing = Callable[[Sequence[object], Sequence[object], np.random.Generator, int], list[tuple[object, object]]]
type Reach = LatticeNeighborhood | MemoryNeighborhood
type MatchingRule = (
    RandomSearchSpec
    | PartnerSpec
    | PartnerSwitchSpec
    | QueueSpec
    | IterativeMatchingSpec
    | ContinuousMatchingSpec
    | BatchMatchingSpec
    | StableMatchingSpec
)
type NamespaceFunction = Callable[[SimpleNamespace], Target]
type PairPotential = Callable[[SimpleNamespace, Expression], Expression]
type Combination = Callable[[SimpleNamespace, Target], Target]
type AgentCombination = Callable[[AgentViewProtocol, SystemViewProtocol, Array | None, int | None], npt.ArrayLike]


@runtime_checkable
class AgentViewProtocol(Protocol):
    """Agent as agent rules read it, such as `AgentView` of the agent stepper.

    It exposes its id, location, moves, and life.
    """

    @property
    def id(self) -> int:
        """Unique id of the agent."""

    @property
    def group(self) -> str:
        """Path of the group of the agent."""

    @property
    def group_label(self) -> str:
        """Label of the agent's group, the last part of its path."""

    @property
    def location(self) -> object:
        """Grid coordinates, network node, or position of the agent."""

    @property
    def location_key(self) -> Hashable:
        """Location of the agent as a key.

        It equals the coordinates `neighbor_cells` gives for its cell or node.
        """

    @property
    def alive(self) -> bool:
        """Whether the agent has not been removed."""

    def neighbor_cells(self) -> list[object]:
        """Return the coordinates of the cells or nodes around the agent's cell or node.

        They are hashable as location keys.
        """

    def agents_here(self, group: str | None = None) -> Sequence[AgentViewProtocol]:
        """Return the other agents of the agent's cell or node.

        They are of every group or of the group of a path.
        """

    def displacement_to(self, other: AgentViewProtocol | object) -> npt.NDArray[np.float64]:
        """Return the vector from the agent to another agent or a location.

        On a torus it is the shortest one.
        """

    def distance_to(self, other: AgentViewProtocol | object) -> float:
        """Return the distance to another agent or a location.

        On a network it is the number of hops.
        """

    def neighbors(
        self,
        radius: float | None = None,
        layer: str | None = None,
        group: str | None = None,
    ) -> Sequence[AgentViewProtocol]:
        """Return the other agents in the neighbouring cells or nodes.

        They are of every group or of the group of a path.
        """

    def regions_here(self, layer: str) -> list[int]:
        """Return the indices of the regions of a continuous space layer with the agent."""

    def move_to(self, location: object) -> None:
        """Move the agent to a cell, a node, or a position.

        It can also move to a uniform point inside a polygon.
        """

    def remove(self) -> None:
        """Remove the agent from the system."""

    def create(self, group: str | None = None, location: object = None, **values: object) -> AgentViewProtocol:
        """Create an agent of a group.

        By default it is the agent's own group at the agent's location.
        """

    def referenced(self, state: str) -> AgentViewProtocol | None:
        """Return the member of another group that a reference state of the agent names.

        The result is `None` when there is none.
        """

    def lags(self, state: str, window: int | None = None) -> npt.NDArray[np.float64]:
        """Return the past values of a state of the agent at the ends of the last steps.

        The last one comes last.
        """


@runtime_checkable
class SystemViewProtocol(Protocol):
    """System as agent rules read it, such as `SystemView` of the agent stepper.

    It exposes its agents, storage, and randomness.
    """

    @property
    def rng(self) -> np.random.Generator:
        """Random number generator of the system."""

    @property
    def storage(self) -> MutableMapping[str, object]:
        """Mutable mapping private to the owner of the rule.

        It persists across the steps of one run.
        """

    def agents(self, name: str) -> Sequence[AgentViewProtocol]:
        """Return the agents of a group bound to a role of the owner or of a group by path."""

    @property
    def dimensions(self) -> tuple[float, ...] | None:
        """Extent of the agent space per axis.

        It counts cells of a lattice or lengths of a continuous space, and is `None` otherwise.
        """

    @property
    def torus(self) -> bool:
        """Whether the agent space wraps around at its edges."""

    def groups(self, name: str) -> list[str]:
        """Return the paths of the groups a role of the owner binds, or of a group by path."""

    def regions(self, layer: str) -> int:
        """Return the number of regions of a layer of the continuous space."""

    def region(self, layer: str, index: int) -> Polygon:
        """Return the polygon of a region of a layer of the continuous space."""

    def parameter(self, name: str) -> float:
        """Return the current value of a parameter of the owner."""

    @property
    def parameters(self) -> dict[str, float]:
        """Current values of every parameter by dotted name."""

    def set_parameter(self, name: str, value: float) -> None:
        """Set a parameter of any group or interaction during the run.

        The change is recorded in the trajectory.
        """

    @property
    def inputs(self) -> Mapping[str, object]:
        """Values of the states of other partitions at the last synchronisation point.

        They are keyed by dotted name.
        """

    @property
    def round(self) -> int:
        """Index of the current round of a repeated phase, `0` outside repeated phases."""

    def count(self, name: str) -> int:
        """Return the number of agents of a group, a role of the owner or a path."""

    def random_empty_cell(self) -> object:
        """Return the coordinates of a random empty cell of the grid or network."""

    def observable(self, name: str) -> object:
        """Return the current value of an observable of the system.

        An example is an aggregate of a group.
        """

    def field(self, name: str, cell: object) -> float:
        """Return the value of a field state at a cell or node.

        The state is given by path or as `<role>.<state>`.
        """

    def deposit(self, name: str, cell: object, amount: float) -> None:
        """Add an amount to a field state at a cell or node.

        The state is given by path or as `<role>.<state>`.
        """

    def create(self, group: str, location: object = None, **values: object) -> AgentViewProtocol:
        """Create an agent of a group, a role of the owner or a path.

        It is created at a location, or by default at a random place.
        """

    @property
    def time(self) -> float:
        """Time of the system.

        It includes the fraction of the step that the visits of a phase reached.
        """

    def history(self, name: str, window: int | None = None) -> npt.NDArray[np.float64]:
        """Return the past values of a reporter or of an observable.

        The values of an observable are those at the end of every earlier step.
        """

    def composition(self, name: str, state: str) -> dict[Hashable, int]:
        """Return the number of agents by value of a state.

        The agents are those of a group, a role of the owner or a path.
        """

    def occupancy(self, name: str) -> dict[Hashable, int]:
        """Return the number of agents by the key of their location.

        The agents are those of a group, a role of the owner or a path.
        """

    def report(self, name: str, value: float) -> None:
        """Publish a value of the owner as the reporter `<owner>.<name>`."""
