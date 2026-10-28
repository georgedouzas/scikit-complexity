"""Define the base of every analysis, which stores its settings, reads its source, and keeps its results."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from ..core.errors import AnalysisError, NotCheckedError
from ..core.system import System
from ..engine.runtime import Trajectory
from ._adapters import JumpPath, Series, Snapshots, fields_of, networks_of, path_of, series_of, snapshots_of
from ._kinds import DataKind

type Source = Trajectory | System | nx.Graph | npt.ArrayLike


class BaseAnalysis(BaseEstimator, ABC):
    """Base of every analysis: it stores its settings, checks them, prepares on a source, and runs.

    A child stores its settings in `__init__` and implements four hooks that the verbs call:

    - `_check()` validates the settings.
    - `_get_data_kind()` returns the kind of data the analysis reads.
    - `_prepare(source)` reads the data from the source with the readers of the base and sets the attributes that end
      in `_`.
    - `_run()` computes the results and sets them as attributes that end in `_`.

    The readers `_read_series`, `_read_path`, `_read_fields`, `_read_snapshots`, and `_read_networks` take a trajectory,
    and pass an array or a graph through.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import BaseAnalysis, DataKind
        >>> class MeanAnalysis(BaseAnalysis):
        ...     def __init__(self, variables=None):
        ...         self.variables = variables
        ...
        ...     def _get_data_kind(self):
        ...         return DataKind.SERIES
        ...
        ...     def _prepare(self, source):
        ...         self.series_ = self._read_series(source, self.variables)
        ...         return self
        ...
        ...     def _run(self):
        ...         self.mean_ = self.series_.values.mean(axis=0)
        ...         return self
        >>> MeanAnalysis().check().prepare(np.array([1.0, 2.0, 3.0])).run().mean_.tolist()
        [2.0]
    """

    def _check(self) -> Self:
        """Validate the settings of the analysis, raising a named error for one it cannot take."""
        return self

    @abstractmethod
    def _get_data_kind(self) -> DataKind:
        """Return the kind of data the analysis reads."""

    @abstractmethod
    def _prepare(self, source: Source) -> Self:
        """Read the data of the analysis from a source and set the attributes that end in `_`."""

    @abstractmethod
    def _run(self) -> Self:
        """Compute the results of the analysis and set them as attributes that end in `_`."""

    def _read_series(self, source: Source, names: Sequence[str] | None = None) -> Series:
        """Return named series of a trajectory, or the columns of an array, one series per column.

        Args:
            source:
                A trajectory, a series, or an array with one row per time and one column per series.

            names:
                The variables of a trajectory, or `None` for every scalar observable and reporter.

        Returns:
            series:
                The times, the columns, and their labels.
        """
        if isinstance(source, Trajectory):
            return series_of(source, names)
        array = np.asarray(source)
        values = array.reshape(-1, 1) if array.ndim <= 1 else array.reshape(array.shape[0], -1)
        labels = list(names) if names is not None else [str(index) for index in range(values.shape[1])]
        return Series(np.arange(values.shape[0], dtype=float), values, labels)

    def _read_path(self, source: Source, name: str | None = None) -> JumpPath:
        """Return the jump path of a discrete state of a trajectory, or of a series of states."""
        if isinstance(source, Trajectory):
            return path_of(source, name)
        values = np.asarray(source)
        changed = np.concatenate([[True], values[1:] != values[:-1]]) if values.ndim == 1 else None
        if changed is None:
            error_msg = f'A jump path is one series of states, but the array has shape {values.shape}.'
            raise AnalysisError(error_msg)
        return JumpPath(np.flatnonzero(changed).astype(float), values[changed], name or 'state')

    def _read_fields(self, source: Source, name: str | None = None) -> npt.NDArray[np.float64]:
        """Return the frames of a lattice or grid state of a trajectory, or an array of frames as it is."""
        if isinstance(source, Trajectory):
            return fields_of(source, name)
        return np.asarray(source, dtype=float)

    def _read_snapshots(self, source: Source, group: str | None = None) -> Snapshots:
        """Return the agents of a group of a trajectory at every frame."""
        if not isinstance(source, Trajectory):
            error_msg = f'Agent snapshots are read from a trajectory, not from {type(source).__name__}.'
            raise AnalysisError(error_msg)
        return snapshots_of(source, group)

    def _read_networks(self, source: Source) -> list[nx.Graph]:
        """Return the network of every frame of a trajectory, or a graph as the only frame."""
        if isinstance(source, nx.Graph):
            return [source]
        if not isinstance(source, Trajectory):
            error_msg = f'Networks are read from a trajectory or a graph, not from {type(source).__name__}.'
            raise AnalysisError(error_msg)
        return networks_of(source)

    def check(self) -> Self:
        """Validate the settings of the analysis and mark it as checked.

        Returns:
            analysis:
                The analysis itself.

        Raises:
            AnalysisError:
                A setting is invalid, named in the message.
        """
        self._check()
        self._checked = True
        return self

    def prepare(self, source: Source) -> Self:
        """Read the data of the checked analysis from a source.

        Args:
            source:
                The trajectory, array, graph, or system the analysis measures.

        Returns:
            analysis:
                The analysis itself.

        Raises:
            NotCheckedError:
                The analysis was not checked with `check`.
        """
        if not getattr(self, '_checked', False):
            raise NotCheckedError()
        self._prepare(source)
        self._prepared = True
        return self

    def run(self) -> Self:
        """Compute the results of the prepared analysis.

        Returns:
            analysis:
                The analysis itself, with its results in the attributes that end in `_`.

        Raises:
            AnalysisError:
                The analysis was not prepared with a source.
        """
        if not getattr(self, '_prepared', False):
            error_msg = f'{type(self).__name__} must be prepared with a source before it runs. Call `prepare(source)`.'
            raise AnalysisError(error_msg)
        return self._run()
