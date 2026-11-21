"""Define the types that only the analyses read."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from typing import Protocol

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..engine.problems import NetworkSnapshotSpec
from ..engine.runtime import Trajectory
from ..systems import BaseSystem
from ..types import AnyArray, Array

type Point = Mapping[str, Expression | float]
type Partition = npt.ArrayLike | Callable[[AnyArray], npt.ArrayLike]
type DataSource = Trajectory | BaseSystem | nx.Graph | NetworkSnapshotSpec | npt.ArrayLike
type Run = Trajectory | npt.ArrayLike
type Runs = DataSource | list[Run]


class ReservoirNodeProtocol(Protocol):
    """A protocol of a node of ReservoirPy, as the reservoir step reads it.

    It names what `reservoir` uses of a node of ReservoirPy, a reservoir or a ridge readout, which is imported on first
    use: its state, its weights, and the methods that initialize, run, step, and fit it, such as the fitted readout
    that turns the features of a sine wave into its next change. It types the nodes of a reservoir model of a chaotic
    flow, a power demand, or a speech signal.

    Read more in the user guide, under Analyse.

    See Also:
        reservoir: Fits the nodes this protocol describes.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import reservoir
        >>> reservoir(np.sin(0.1 * np.arange(400.0)), variables=['x']).step.readout.initialized
        True
    """

    initialized: bool
    state: dict[str, Array]
    Wout: Array
    bias: Array
    _monomial_idx: npt.NDArray[np.int_]

    def initialize(self, x: Array) -> None:
        """Initialize the node for inputs of the shape of an example."""

    def run(self, x: Array) -> npt.ArrayLike:
        """Return the outputs of the node driven by a series of inputs."""

    def step(self, x: Array) -> npt.ArrayLike:
        """Return the output of the node after one input."""

    def fit(self, x: Array, y: Array) -> object:
        """Fit the node to inputs and targets."""
