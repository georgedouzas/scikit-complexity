"""Name the kinds of data that analyses consume and register every analysis under its kind."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..base import Registry

type Function = Callable[..., Any]


class DataKind(StrEnum):
    """Kind of data an analysis consumes, whatever model or measurement produced it.

    Examples:
        >>> from skcomplex.core.analysis import DataKind
        >>> DataKind('field').reading
        'a grid of values, or frames of grids'
    """

    SERIES = 'series'
    PATH = 'path'
    FIELD = 'field'
    AGENTS = 'agents'
    NETWORK = 'network'
    ENSEMBLE = 'ensemble'

    @property
    def reading(self) -> str:
        """Plain arrays or graphs the analyses of the kind read."""
        return READINGS[self]

    @property
    def adapter(self) -> str:
        """Name of the function that extracts the kind from a trajectory or an experiment result."""
        return ADAPTERS[self]


READINGS = {
    DataKind.SERIES: 'a series of values, or the columns of an array as series',
    DataKind.PATH: 'the jump times and the states after every jump',
    DataKind.FIELD: 'a grid of values, or frames of grids',
    DataKind.AGENTS: 'positions, velocities, headings, or states of agents, one row per agent',
    DataKind.NETWORK: 'a NetworkX graph, or one graph per frame',
    DataKind.ENSEMBLE: 'one row per run and one column per common time',
}
ADAPTERS = {
    DataKind.SERIES: 'series_of',
    DataKind.PATH: 'path_of',
    DataKind.FIELD: 'fields_of',
    DataKind.AGENTS: 'snapshots_of',
    DataKind.NETWORK: 'networks_of',
    DataKind.ENSEMBLE: 'ensemble_of',
}


@dataclass(frozen=True)
class AnalysisEntry:
    """Analysis registered under the kind of data it consumes.

    Args:
        name:
            The name of the analysis function.

        function:
            The analysis function.

        kind:
            The kind of data it consumes.
    """

    name: str
    function: Function
    kind: DataKind


ANALYSES = Registry[AnalysisEntry]('analyses')


def analysis[F: Function](kind: DataKind) -> Callable[[F], F]:
    """Return a decorator that registers an analysis function under the kind of data it consumes.

    Args:
        kind:
            The kind of data the function consumes.

    Returns:
        decorator:
            The decorator, which returns the function unchanged.

    Examples:
        >>> from skcomplex.core.analysis import ANALYSES, DataKind
        >>> ANALYSES.resolve('structure_factor').kind
        <DataKind.FIELD: 'field'>
    """

    def register(function: F) -> F:
        ANALYSES.register(function.__name__, AnalysisEntry(function.__name__, function, kind))
        return function

    return register
