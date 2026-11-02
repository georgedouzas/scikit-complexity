"""Define the types of the analyses: inputs that only the analyses read, and the ReservoirPy node they drive."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from typing import Protocol

import networkx as nx
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..core.system import System
from ..engine.runtime import Trajectory
from ..specs import NetworkSnapshotSpec
from ..types import AnyArray, Array

type Point = Mapping[str, Expression | float]
type Partition = npt.ArrayLike | Callable[[AnyArray], npt.ArrayLike]
type Source = Trajectory | System | nx.Graph | NetworkSnapshotSpec | npt.ArrayLike
type Run = Trajectory | npt.ArrayLike
type Runs = Source | list[Run]
type Item = str | tuple[int, ...]


class ReservoirNodeProtocol(Protocol):
    """Node of ReservoirPy, imported on first use, as the reservoir step reads it: a reservoir or a ridge readout."""

    initialized: bool
    store: Array
    state: dict[str, Array]

    def initialize(self, x: Array) -> None:
        """Initialize the node for inputs of the shape of an example."""

    def run(self, x: Array) -> npt.ArrayLike:
        """Return the outputs of the node driven by a series of inputs."""

    def step(self, x: Array) -> npt.ArrayLike:
        """Return the output of the node after one input."""

    def fit(self, x: Array, y: Array) -> object:
        """Fit the node to inputs and targets."""
