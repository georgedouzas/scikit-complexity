"""Define the types of the core: the targets of laws, the pairings and rules of matches, and the protocols."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Callable, Hashable, MutableMapping, Sequence
from types import SimpleNamespace
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression
from sage.modules.free_module_element import FreeModuleElement
from sage.rings.polynomial.laurent_polynomial import LaurentPolynomial
from sage.structure.element import Matrix

from ..specs import (
    BatchClearingSpec,
    ContinuousClearingSpec,
    GreedyMatchingSpec,
    IterativeClearingSpec,
    PartnerSwitchSpec,
    QueueSpec,
    RandomSearchSpec,
    StableMatchingSpec,
)
from ..types import Array, Indices, IntArray
from .spaces import BaseSpace, LatticeNeighborhood, MemoryNeighborhood

type Target = Expression | FreeModuleElement
type Process = str | tuple[str, ...] | Expression | FreeModuleElement | None
type Pairing = Callable[[Sequence[object], Sequence[object], np.random.Generator, int], list[tuple[object, object]]]
type Reach = LatticeNeighborhood | MemoryNeighborhood
type MatchingRule = (
    RandomSearchSpec
    | PartnerSwitchSpec
    | QueueSpec
    | IterativeClearingSpec
    | ContinuousClearingSpec
    | BatchClearingSpec
    | StableMatchingSpec
    | GreedyMatchingSpec
)
type NamespaceFunction = Callable[[SimpleNamespace], Target]
type PairPotential = Callable[[SimpleNamespace, Expression], Expression]
type Combination = Callable[[SimpleNamespace, Target], Target]
type AgentCombination = Callable[[AgentViewProtocol, ModelViewProtocol, Array | None, int | None], npt.ArrayLike]


@runtime_checkable
class AutomatonProtocol(Protocol):
    """Cellular automaton the analysis reads by its methods, such as `Automaton` of the automata models."""

    space: BaseSpace
    neighborhood: LatticeNeighborhood | MemoryNeighborhood
    rule: object
    states: int

    def __init__(
        self,
        space: BaseSpace,
        neighborhood: LatticeNeighborhood | MemoryNeighborhood,
        rule: object,
        states: int,
    ) -> None:
        """Store the space, the neighbourhood, the rule, and the number of states."""

    @property
    def reads(self) -> LatticeNeighborhood | MemoryNeighborhood:
        """Neighbourhood the rule reads."""

    @property
    def size(self) -> int:
        """Number of sites."""

    def table(self) -> Indices:
        """Return the site of every cell the rule reads at every site."""

    def local_map(self) -> IntArray:
        """Return the next value of every pattern of the cells the rule reads."""

    def step(self, configuration: npt.ArrayLike) -> IntArray:
        """Return the image of a configuration."""

    def run(self, configuration: npt.ArrayLike, steps: int) -> IntArray:
        """Return the configurations of a number of steps."""


@runtime_checkable
class LinearAutomatonProtocol(Protocol):
    """Linear cellular automaton the analysis reads by its matrix and its Laurent polynomial."""

    def matrix(self) -> Matrix:
        """Return the Sage matrix of the automaton on its space."""

    def laurent_polynomial(self) -> LaurentPolynomial:
        """Return the Sage Laurent polynomial of the automaton on `Z^d`."""


@runtime_checkable
class SubshiftProtocol(Protocol):
    """Subshift of finite type the analysis reads by its count of admissible configurations and its violations."""

    def count(self, space: BaseSpace) -> int:
        """Return the number of admissible configurations of a space."""

    def violations(self, configuration: npt.ArrayLike, space: BaseSpace) -> npt.NDArray[np.bool_]:
        """Return the sites at which a forbidden pattern appears."""


@runtime_checkable
class AgentViewProtocol(Protocol):
    """Agent as agent rules read it, such as `AgentView` of the agent stepper: its id, location, moves, and life."""

    @property
    def id(self) -> int:
        """Unique id of the agent."""

    @property
    def group(self) -> str:
        """Path of the group of the agent."""

    @property
    def location(self) -> object:
        """Grid coordinates, network node, or position of the agent."""

    @property
    def location_key(self) -> Hashable:
        """Location of the agent as a key, equal to the coordinates `neighbor_cells` gives for its cell or node."""

    @property
    def alive(self) -> bool:
        """Whether the agent has not been removed."""

    def neighbor_cells(self) -> list[object]:
        """Return the coordinates of the cells or nodes around the agent's cell or node, hashable as location keys."""

    def agents_here(self, group: str | None = None) -> Sequence[AgentViewProtocol]:
        """Return the other agents of the agent's cell or node, of every group or of the group of a path."""

    def distance_to(self, other: AgentViewProtocol | object) -> float:
        """Return the distance to another agent or a location, the number of hops on a network."""

    def neighbors(self, radius: float | None = None) -> Sequence[AgentViewProtocol]:
        """Return the other agents in the neighbouring cells or nodes, or within a radius in a continuous space."""

    def move_to(self, location: object) -> None:
        """Move the agent to a cell, a node, or a position."""

    def remove(self) -> None:
        """Remove the agent from the model."""

    def create(self, group: str | None = None, location: object = None, **values: object) -> AgentViewProtocol:
        """Create an agent of a group, by default the agent's own group at the agent's location."""

    def referenced(self, state: str) -> AgentViewProtocol | None:
        """Return the member of another group that a reference state of the agent names, or `None`."""


@runtime_checkable
class ModelViewProtocol(Protocol):
    """Model as agent rules read it, such as `ModelView` of the agent stepper: its agents, storage, and randomness."""

    @property
    def rng(self) -> np.random.Generator:
        """Random number generator of the model."""

    @property
    def storage(self) -> MutableMapping[str, object]:
        """Mutable mapping private to the owner of the rule that persists across the steps of one run."""

    def agents(self, name: str) -> Sequence[AgentViewProtocol]:
        """Return the agents of a group bound to a role of the owner, or of a group by path."""

    @property
    def dimensions(self) -> tuple[float, ...] | None:
        """Extent of the agent space per axis, cells of a lattice or lengths of a continuous space, `None` otherwise."""

    @property
    def torus(self) -> bool:
        """Whether the agent space wraps around at its edges."""

    def groups(self, name: str) -> list[str]:
        """Return the paths of the groups a role of the owner binds, or of a group by path."""

    def parameter(self, name: str) -> float:
        """Return the current value of a parameter of the owner."""

    def field(self, name: str, cell: object) -> float:
        """Return the value of a field state, by path or as `<role>.<state>`, at a cell or node."""

    def deposit(self, name: str, cell: object, amount: float) -> None:
        """Add an amount to a field state, by path or as `<role>.<state>`, at a cell or node."""

    def create(self, group: str, location: object = None, **values: object) -> AgentViewProtocol:
        """Create an agent of a group, a role of the owner or a path, at a location or by default a random place."""

    @property
    def time(self) -> float:
        """Time of the model, with the fraction of the step that the visits of a phase reached."""

    def history(self, name: str, window: int | None = None) -> npt.NDArray[np.float64]:
        """Return the past values of a reporter, or of an observable at the end of every earlier step."""

    def composition(self, name: str, state: str) -> dict[float, int]:
        """Return the number of agents of a group, a role of the owner or a path, by value of a state."""

    def occupancy(self, name: str) -> dict[Hashable, int]:
        """Return the number of agents of a group, a role of the owner or a path, by the key of their location."""

    def report(self, name: str, value: float) -> None:
        """Publish a value of the owner as the reporter `<owner>.<name>`."""
