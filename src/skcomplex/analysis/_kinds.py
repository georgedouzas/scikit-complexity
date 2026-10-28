"""Name the kinds of data that analyses consume."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from enum import StrEnum


class DataKind(StrEnum):
    """Kind of data an analysis consumes, whatever model or measurement produced it.

    A kind equals its value, so an analysis may return the plain string, such as `'series'`.

    Examples:
        >>> from skcomplex.analysis._kinds import DataKind
        >>> DataKind('field').reading
        'a grid of values, or frames of grids'
        >>> DataKind.SERIES == 'series'
        True
    """

    SERIES = 'series'
    PATH = 'path'
    FIELD = 'field'
    AGENTS = 'agents'
    NETWORK = 'network'
    ENSEMBLE = 'ensemble'
    SYSTEM = 'system'

    @property
    def reading(self) -> str:
        """Plain arrays or graphs the analyses of the kind read."""
        return READINGS[self]


READINGS = {
    DataKind.SERIES: 'a series of values, or the columns of an array as series',
    DataKind.PATH: 'the jump times and the states after every jump',
    DataKind.FIELD: 'a grid of values, or frames of grids',
    DataKind.AGENTS: 'positions, velocities, headings, or states of agents, one row per agent',
    DataKind.NETWORK: 'a NetworkX graph, or one graph per frame',
    DataKind.ENSEMBLE: 'one row per run and one column per common time',
    DataKind.SYSTEM: 'a system, a problem compiled from one, or a trajectory of one, whose equations are analysed',
}
