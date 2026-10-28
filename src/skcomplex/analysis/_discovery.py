"""List the analyses that apply to a trajectory, a system, or an array, by the kinds of data it holds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import sys
from typing import Self

import networkx as nx
import numpy as np

from ..core.system import System
from ..engine.runtime import Trajectory
from ..specs import NetworkSnapshotSpec
from ..types import DataKind
from ._base import SERIES_DIMENSIONS, BaseAnalysis, Source


class DiscoveryAnalysis(BaseAnalysis):
    """Analyses that apply to a trajectory, a system, or an array, listed under the kind of data they read.

    Preparing reads the kinds of data the source holds. Running builds every analysis of the library that takes no
    required setting, once per kind of a family, asks each for its data kind, and keeps those whose kind the source
    holds. A one-dimensional array is one series, and an array of more dimensions is one field and a set of series, one
    per column.

    Attributes:
        kinds_:
            The data kinds of the source, in the order agents, field, network, path, series.

        readings_:
            How the analyses of every data kind read the source, by data kind.

        analyses_:
            The unprepared analyses that apply to the source, by data kind.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.analysis import DiscoveryAnalysis
        >>> discovery = DiscoveryAnalysis().check().prepare(np.zeros((16, 16))).run()
        >>> discovery.kinds_
        ['field', 'series']
        >>> discovery.readings_['series']
        'every column of the array as a series'
        >>> any(type(item).__name__ == 'EntropyAnalysis' for item in discovery.analyses_['series'])
        True
    """

    def _get_data_kind(self) -> DataKind:
        """Return the system kind, since discovery reads the source as a whole rather than one kind of its data."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> Self:
        """Read the data kinds of the source and how the analyses of every kind read it."""
        kinds = self._get_data_kinds(source)
        self.kinds_ = [str(kind) for kind in kinds]
        if isinstance(source, Trajectory | System | nx.Graph | NetworkSnapshotSpec):
            self.readings_ = {str(kind): kind.reading for kind in kinds}
            return self
        self.readings_ = {str(kind): self._get_array_reading(kind, np.ndim(source)) for kind in kinds}
        return self

    @staticmethod
    def _get_array_reading(kind: DataKind, dimensions: int) -> str:
        """Return how the analyses of a kind read a plain array of some dimensions."""
        if kind == DataKind.FIELD:
            return 'the array as one grid'
        if dimensions > SERIES_DIMENSIONS:
            return 'every column of the array as a series'
        return 'the array as one series'

    @staticmethod
    def _get_classes(base: type[BaseAnalysis]) -> list[type[BaseAnalysis]]:
        """Return every concrete general analysis below a base, by name, leaving out the realizations of the models."""
        found = []
        for child in base.__subclasses__():
            found.extend(DiscoveryAnalysis._get_classes(child))
            if child.__module__.startswith('skcomplex.analysis.') and not inspect.isabstract(child):
                found.append(child)
        return sorted(set(found), key=lambda item: item.__name__)

    @staticmethod
    def _build(analysis_class: type[BaseAnalysis]) -> list[BaseAnalysis]:
        """Return an analysis with its default settings, one per kind of its module for a family, or none."""
        parameters = inspect.signature(analysis_class).parameters.values()
        if any(parameter.default is inspect.Parameter.empty for parameter in parameters):
            return []
        names = [parameter.name for parameter in parameters]
        kinds = getattr(sys.modules[analysis_class.__module__], 'KINDS', None)
        if 'kind' in names and kinds is not None:
            return [analysis_class(kind=kind) for kind in kinds]
        return [analysis_class()]

    def _run(self, inputs: object = None) -> Self:
        """Build every analysis of the library and keep those that read a kind of data of the source."""
        self.analyses_ = {kind: [] for kind in self.kinds_}
        for analysis_class in self._get_classes(BaseAnalysis):
            for analysis in self._build(analysis_class):
                kind = analysis._get_data_kind()
                if kind in self.analyses_:
                    self.analyses_[kind].append(analysis)
        return self
