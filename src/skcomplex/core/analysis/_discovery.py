"""List the analyses that apply to a trajectory, a system, or an array, by the kinds of data it holds."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import networkx as nx
import numpy as np

from ..components.system import System
from ..engine.runtime import Trajectory
from ..specs import NetworkSnapshotSpec
from ..types import DataKind
from ._alignment import alignment
from ._autocatalytic import autocatalytic_sets
from ._automata import automaton
from ._base import SERIES_DIMENSIONS, _get_data_kinds, bind_analysis
from ._cell_complex import cell_complex
from ._clusters import clusters
from ._coarse_grain import coarse_grain
from ._collapse import data_collapse
from ._complexity import complexity
from ._computational_mechanics import epsilon_machine
from ._correlation import correlation
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
from ._operator import operator_measures
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
from .types import Source

# The data kind every analysis reads with its default settings, and the kinds of a family that read another.
DATA_KINDS: dict[Callable[..., object], tuple[DataKind, dict[str, DataKind]]] = {
    alignment: (DataKind.AGENTS, {}),
    autocatalytic_sets: (DataKind.NETWORK, {}),
    automaton: (DataKind.SYSTEM, {}),
    cell_complex: (DataKind.FIELD, {}),
    chain_shape: (DataKind.AGENTS, {}),
    clusters: (DataKind.FIELD, {}),
    coarse_grain: (DataKind.AGENTS, {}),
    complexity: (DataKind.SERIES, {}),
    correlation: (DataKind.FIELD, {'auto': DataKind.SERIES, 'two-time': DataKind.ENSEMBLE}),
    data_collapse: (DataKind.ENSEMBLE, {}),
    defects: (DataKind.FIELD, {}),
    displacement: (DataKind.AGENTS, {}),
    distribution: (DataKind.SERIES, {'records': DataKind.ENSEMBLE}),
    domains: (DataKind.FIELD, {}),
    elementary_modes: (DataKind.SERIES, {}),
    embedding: (DataKind.SERIES, {}),
    emergence: (DataKind.SERIES, {'effective-information': DataKind.PATH, 'causal': DataKind.PATH}),
    entropy: (DataKind.SERIES, {'ensemble': DataKind.ENSEMBLE}),
    epsilon_machine: (DataKind.SERIES, {}),
    ergodicity: (
        DataKind.SERIES,
        {'breaking': DataKind.ENSEMBLE, 'metric': DataKind.ENSEMBLE, 'distance': DataKind.ENSEMBLE},
    ),
    events: (DataKind.SERIES, {'counts': DataKind.PATH}),
    fit_law: (DataKind.SERIES, {}),
    information: (DataKind.SERIES, {}),
    invariants: (DataKind.SYSTEM, {}),
    irreversibility: (
        DataKind.PATH,
        {
            'langevin-entropy-production': DataKind.ENSEMBLE,
            'fluctuation-theorem': DataKind.ENSEMBLE,
            'uncertainty-relation': DataKind.ENSEMBLE,
            'response': DataKind.ENSEMBLE,
            'fluctuation-dissipation': DataKind.ENSEMBLE,
        },
    ),
    kramers_moyal: (DataKind.SERIES, {}),
    large_deviations: (
        DataKind.PATH,
        {'block': DataKind.SERIES, 'histogram': DataKind.ENSEMBLE, 'reweight': DataKind.ENSEMBLE},
    ),
    linearization: (DataKind.SYSTEM, {}),
    local_information: (DataKind.SERIES, {}),
    loops: (DataKind.SYSTEM, {}),
    lyapunov: (DataKind.SYSTEM, {'largest': DataKind.SERIES}),
    markov: (DataKind.PATH, {}),
    mode_decomposition: (DataKind.SERIES, {}),
    motifs: (DataKind.NETWORK, {}),
    neighbors: (DataKind.AGENTS, {'like-share': DataKind.FIELD}),
    network_measures: (DataKind.NETWORK, {}),
    operator_measures: (DataKind.SYSTEM, {}),
    partial_information: (DataKind.SERIES, {}),
    passage: (DataKind.PATH, {'return-times': DataKind.SERIES}),
    recurrence: (DataKind.SERIES, {}),
    renormalization: (DataKind.NETWORK, {}),
    reservoir: (DataKind.SERIES, {}),
    rolling: (DataKind.SERIES, {}),
    scaling: (
        DataKind.SERIES,
        {'box-counting': DataKind.FIELD, 'variance-mean': DataKind.FIELD, 'power': DataKind.ENSEMBLE},
    ),
    section: (DataKind.SYSTEM, {}),
    signatures: (DataKind.SYSTEM, {}),
    sparse_identification: (DataKind.SERIES, {}),
    spectrum: (DataKind.FIELD, {'power': DataKind.SERIES}),
    stability: (DataKind.SYSTEM, {}),
    surrogate_test: (DataKind.SERIES, {}),
    symbolic: (DataKind.SYSTEM, {}),
}


@dataclass(frozen=True)
class ApplicableAnalysesResult:
    """Analyses that apply to a source, listed under the kind of data they read.

    Args:
        kinds:
            The data kinds of the source, in the order agents, field, network, path, series.

        readings:
            How the analyses of every data kind read the source, by data kind.

        analyses:
            The analyses that apply to the source by data kind, each bound to its kind for a family, so calling one on
            the source gives its result.
    """

    kinds: list[str]
    readings: dict[str, str]
    analyses: dict[str, list[partial[object]]]


def applicable_analyses(source: Source) -> ApplicableAnalysesResult:
    """Return the analyses that apply to a trajectory, a system, or an array, listed under the kind of data they read.

    The kinds of data the source holds are read first. Every analysis of the library that takes no required setting is
    bound once per kind of a family, and those whose data kind the source holds are kept. A one-dimensional array is one
    series, and an array of more dimensions is one field and a set of series, one per column.

    Args:
        source:
            The trajectory, the system, the graph, or the array.

    Returns:
        result:
            The data kinds of the source, how they are read, and the analyses of every kind.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import applicable_analyses
        >>> found = applicable_analyses(np.zeros((16, 16)))
        >>> found.kinds
        ['field', 'series']
        >>> found.readings['series']
        'every column of the array as a series'
        >>> any(item.func.__name__ == 'entropy' for item in found.analyses['series'])
        True
    """
    kinds = [str(kind) for kind in _get_data_kinds(source)]
    if isinstance(source, Trajectory | System | nx.Graph | NetworkSnapshotSpec):
        readings = {kind: DataKind(kind).reading for kind in kinds}
    else:
        readings = {kind: _get_array_reading(DataKind(kind), np.ndim(source)) for kind in kinds}
    analyses: dict[str, list[partial[object]]] = {kind: [] for kind in kinds}
    for function, (default, by_kind) in sorted(DATA_KINDS.items(), key=lambda item: item[0].__name__):
        parameters = list(inspect.signature(function).parameters.values())[1:]
        if any(parameter.default is inspect.Parameter.empty for parameter in parameters):
            continue
        family = getattr(sys.modules[function.__module__], 'KINDS', None)
        if family is not None and any(parameter.name == 'kind' for parameter in parameters):
            bound = [(bind_analysis(function, kind=kind), by_kind.get(kind, default)) for kind in family]
        else:
            bound = [(bind_analysis(function), default)]
        for analysis, kind in bound:
            if kind in analyses:
                analyses[kind].append(analysis)
    return ApplicableAnalysesResult(kinds, readings, analyses)


def _get_array_reading(kind: DataKind, dimensions: int) -> str:
    """Return how the analyses of a kind read a plain array of some dimensions."""
    if kind == DataKind.FIELD:
        return 'the array as one grid'
    if dimensions > SERIES_DIMENSIONS:
        return 'every column of the array as a series'
    return 'the array as one series'
