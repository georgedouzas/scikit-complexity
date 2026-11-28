"""List the analyses that apply to a trajectory, a system, or an array."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import sys
from dataclasses import dataclass
from functools import partial

import networkx as nx
import numpy as np

from ..engine.problems import NetworkSnapshotSpec
from ..engine.problems._composite_system import _CompositeSystem
from ..engine.runtime import Trajectory
from ..enums import DataKind
from ._alignment import alignment
from ._autocatalytic import autocatalytic_sets
from ._automata import automaton
from ._base import SERIES_DIMENSIONS, _composite, _get_analysis_tags, _get_data_kinds, _get_source_tags, bind_analysis
from ._cell_complex import cell_complex
from ._clusters import clusters
from ._coarse_grain import coarse_grain
from ._collapse import data_collapse
from ._complexity import complexity
from ._computational_mechanics import epsilon_machine
from ._correlation import correlation
from ._correlation_network import correlation_network
from ._defects import defects
from ._displacement import displacement
from ._distribution import distribution
from ._domains import domains
from ._elementary_modes import elementary_modes
from ._embedding import embedding
from ._emergence import emergence
from ._entropy import entropy
from ._ergodicity import ergodicity
from ._events import events
from ._fit import fit_law
from ._information import information
from ._invariants import invariants
from ._irreversibility import irreversibility
from ._kramers_moyal import kramers_moyal
from ._large_deviations import large_deviations
from ._linearization import linearization
from ._local_information import local_information
from ._loops import loops
from ._lyapunov import lyapunov
from ._markov import markov
from ._mode_decomposition import mode_decomposition
from ._motifs import motifs
from ._neighbors import neighbors
from ._networks import network_measures
from ._partial_information import partial_information
from ._passage import passage
from ._recurrence import recurrence
from ._renormalization import renormalization
from ._reservoirs import reservoir
from ._rolling import rolling
from ._scaling import scaling
from ._sections import section
from ._shape import chain_shape
from ._signatures import signatures
from ._sparse_identification import sparse_identification
from ._spectrum import spectrum
from ._stability import stability
from ._surrogates import surrogate_test
from ._symbolic import symbolic
from .types import DataSource

ANALYSES = (
    alignment,
    autocatalytic_sets,
    automaton,
    cell_complex,
    chain_shape,
    clusters,
    coarse_grain,
    complexity,
    correlation,
    correlation_network,
    data_collapse,
    defects,
    displacement,
    distribution,
    domains,
    elementary_modes,
    embedding,
    emergence,
    entropy,
    epsilon_machine,
    ergodicity,
    events,
    fit_law,
    information,
    invariants,
    irreversibility,
    kramers_moyal,
    large_deviations,
    linearization,
    local_information,
    loops,
    lyapunov,
    markov,
    mode_decomposition,
    motifs,
    neighbors,
    network_measures,
    partial_information,
    passage,
    recurrence,
    renormalization,
    reservoir,
    rolling,
    scaling,
    section,
    signatures,
    sparse_identification,
    spectrum,
    stability,
    surrogate_test,
    symbolic,
)


@dataclass(frozen=True)
class ApplicableAnalysesResult:
    """A result of `applicable_analyses`, the analyses that fit a source, by kind.

    It holds the kinds of data the source holds, such as `['field', 'series']` for a 2D array, what each kind reads from
    it, and the analyses that apply under each kind, ready to call. It answers which measures fit a run of an epidemic,
    a grid of a forest, or a table of prices.

    Read more in the user guide, under Analyse.

    Args:
        kinds:
            The data kinds of the source, in the order agents, field, network, path, series.

        readings:
            How the analyses of every data kind read the source, by data kind.

        analyses:
            The analyses that apply to the source by data kind, each bound to its kind for a family, so calling one on
            the source gives its result.

    See Also:
        applicable_analyses: Returns this result.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import applicable_analyses
        >>> applicable_analyses(np.zeros((16, 16))).kinds
        ['field', 'series']
    """

    kinds: list[str]
    readings: dict[str, str]
    analyses: dict[str, list[partial[object]]]


def applicable_analyses(source: DataSource) -> ApplicableAnalysesResult:
    """Return the analyses that apply to a trajectory, a system, or an array.

    It reads the kinds of data the source holds, such as a field and a set of series for a 2D array, and keeps every
    analysis of the library that reads one of them and needs no setting, bound once per kind. A system or a trajectory
    of one also drops the analyses whose tags the system lacks, such as `lyapunov` of a random system. It answers which
    measures fit a run of an epidemic, a grid of a forest, or a table of prices.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The trajectory, the system, the graph, or the array.

    Returns:
        result:
            The data kinds of the source, how they are read, and the analyses of every kind.

    See Also:
        AnalysisTags: Names what an analysis needs of its data.
        bind_analysis: Fixes the settings of an analysis.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import applicable_analyses
        >>> applicable_analyses(np.zeros((16, 16))).readings['series']
        'every column of the array as a series'
    """
    source = _composite(source)
    kinds = [str(kind) for kind in _get_data_kinds(source)]
    if isinstance(source, Trajectory | _CompositeSystem | nx.Graph | NetworkSnapshotSpec):
        readings = {kind: DataKind(kind).reading for kind in kinds}
    else:
        readings = {kind: _get_array_reading(DataKind(kind), np.ndim(source)) for kind in kinds}
    tags = _get_source_tags(source)
    analyses: dict[str, list[partial[object]]] = {kind: [] for kind in kinds}
    for function in ANALYSES:
        parameters = list(inspect.signature(function).parameters.values())[1:]
        if any(parameter.default is inspect.Parameter.empty for parameter in parameters):
            continue
        family = getattr(sys.modules[function.__module__], 'KINDS', None)
        if family is not None and any(parameter.name == 'kind' for parameter in parameters):
            bound = [(bind_analysis(function, kind=kind), _get_analysis_tags(function, kind)) for kind in family]
        else:
            bound = [(bind_analysis(function), _get_analysis_tags(function))]
        for analysis, needed in bound:
            if needed.data in analyses and (tags is None or not needed.get_missing(tags)):
                analyses[needed.data].append(analysis)
    return ApplicableAnalysesResult(kinds, readings, analyses)


def _get_array_reading(kind: DataKind, dimensions: int) -> str:
    """Return how the analyses of a kind read a plain array of some dimensions."""
    if kind == DataKind.FIELD:
        return 'the array as one grid'
    if dimensions > SERIES_DIMENSIONS:
        return 'every column of the array as a series'
    return 'the array as one series'
