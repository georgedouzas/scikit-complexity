"""Build networks from the correlations of many series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import stats

from ..enums import DataKind
from ..types import Array
from ._analysis_tags import AnalysisTags
from ._base import _check_choice, _check_settings, _read_real, _read_series
from .types import DataSource

METHODS = ('pearson', 'spearman', 'kendall')
CORRECTIONS = ('bonferroni', 'benjamini-hochberg')
BACKBONES = ('spanning-tree', 'disparity')
SETTINGS = {None: {'level', 'correction'}, 'spanning-tree': set(), 'disparity': {'level'}}
TAGS = AnalysisTags(DataKind.SERIES)


@dataclass(frozen=True)
class CorrelationNetworkResult:
    """A result of `correlation_network`, the network of correlations of many series.

    It holds the correlation and the p-value of every pair of series, the threshold of the p-values, the kept links,
    and the graph of the links, such as `[('0', '1')]` for two series that share a signal beside one that does not. It
    gives the market of stocks that move together, the brain regions that fire together, and the climate grid points
    whose temperatures follow each other.

    Read more in the user guide, under Analyse.

    Args:
        labels:
            The name of every series in column order.

        strengths:
            The correlation matrix of shape `(variables, variables)`, with ones on the diagonal.

        p_values:
            The two sided p-value of every pair, of the same shape, `nan` on the diagonal.

        threshold:
            The p-value below which a pair is a link, or `nan` for a backbone.

        links:
            The kept pairs as `(source, target)` tuples of labels, the source before the target in label order.

        graph:
            An undirected NetworkX graph whose nodes are the labels in order and whose edges are the links, each with
            `weight`, the absolute correlation, `strength`, the signed correlation, and `p_value`.

    See Also:
        correlation_network: Returns this result.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import correlation_network
        >>> x, y, z = np.random.default_rng(0).standard_normal((3, 200))
        >>> correlation_network(np.column_stack([x, x + y, z])).graph.number_of_edges()
        1
    """

    labels: list[str]
    strengths: Array
    p_values: Array
    threshold: float
    links: list[tuple[str, str]]
    graph: nx.Graph


def correlation_network(
    source: DataSource,
    *,
    method: str = 'pearson',
    variables: Sequence[str] | None = None,
    level: float = 0.05,
    correction: str | None = 'benjamini-hochberg',
    backbone: str | None = None,
) -> CorrelationNetworkResult:
    """Return the network of the significant correlations of many series.

    Every series is a node, and a pair of series is a link when their correlation is significant after a correction for
    the many tests of all pairs, or when it belongs to a backbone of the complete weighted network. It turns the
    returns of stocks into the market network of Mantegna, the signals of brain regions into a functional connectivity
    network, and the temperatures of grid points into a climate network, and its graph goes to `network_measures`. Rows
    with a missing value in any series are dropped, so every pair uses the same times. The correlation matrix holds
    `variables²` floats, which fits a few thousand series in memory.

    Read more in the user guide, under Analyse.

    Args:
        source:
            The trajectory, or an array with one row per time and one column per series.

        method:
            The correlation, `'pearson'`, `'spearman'` of the ranks, or `'kendall'` of the concordant pairs. Kendall
            loops over all pairs and suits tens of series.

        variables:
            The series of a trajectory, or the labels of the columns of an array, or `None` for every scalar series.

        level:
            The significance level of the links, or of the disparity filter.

        correction:
            The correction of the p-values over all pairs, `'bonferroni'`, `'benjamini-hochberg'`, or `None` for the
            level itself. Bonferroni bounds the chance of any false link, and Benjamini and Hochberg the share of false
            links among the kept ones.

        backbone:
            `None` for the corrected links, `'spanning-tree'` for the maximum spanning tree of the signed correlations,
            or `'disparity'` for the pairs that the disparity filter of Serrano, Boguñá, and Vespignani keeps at
            `level` for one of their ends. The disparity filter keeps few links of a dense network with even weights.

    Returns:
        result:
            The correlations, the p-values, the links, and their graph.

    Raises:
        AnalysisError:
            The method, correction, or backbone is unknown, or a setting the backbone does not read is changed.

    See Also:
        correlation: Correlates one series with itself.
        network_measures: Measures the graph of the result.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import correlation_network
        >>> x, y, z = np.random.default_rng(0).standard_normal((3, 200))
        >>> correlation_network(np.column_stack([x, x + y, z])).links
        [('0', '1')]
    """
    _check_choice('correlation_network', 'method', method, METHODS)
    _check_choice('correlation_network', 'correction', correction, (*CORRECTIONS, None))
    _check_choice('correlation_network', 'backbone', backbone, (*BACKBONES, None))
    _check_settings(
        correlation_network,
        'backbone',
        SETTINGS[backbone] | {'method', 'variables'},
        backbone=backbone,
        method=method,
        variables=variables,
        level=level,
        correction=correction,
    )
    series = _read_series(source, variables)
    values = _read_real(series.values, 'series')
    values = values[~np.isnan(values).any(axis=1)]
    strengths, p_values = _correlate(values, method)
    if backbone == 'spanning-tree':
        pairs, threshold = _spanning_tree(strengths), float('nan')
    elif backbone == 'disparity':
        pairs, threshold = _disparity(strengths, level), float('nan')
    else:
        pairs, threshold = _significant(p_values, level, correction)
    labels = series.labels
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    for first, second in pairs:
        graph.add_edge(
            labels[first],
            labels[second],
            weight=abs(float(strengths[first, second])),
            strength=float(strengths[first, second]),
            p_value=float(p_values[first, second]),
        )
    links = [(labels[first], labels[second]) for first, second in pairs]
    return CorrelationNetworkResult(labels, strengths, p_values, threshold, links, graph)


def _correlate(values: Array, method: str) -> tuple[Array, Array]:
    """Return the correlation matrix and the two sided p-values of the columns.

    A constant column has `nan` correlations and p-values, and the diagonal has ones and `nan`.
    """
    size, count = values.shape
    with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
        warnings.simplefilter('ignore', stats.ConstantInputWarning)
        if method == 'pearson':
            strengths = np.atleast_2d(np.corrcoef(values, rowvar=False))
            statistic = np.abs(strengths) * np.sqrt((size - 2) / (1 - strengths**2))
            p_values = 2 * stats.t.sf(statistic, size - 2)
        elif method == 'spearman' and count > 1:
            result = stats.spearmanr(values)
            strengths = np.broadcast_to(result.statistic, (count, count)).copy()
            p_values = np.broadcast_to(result.pvalue, (count, count)).copy()
        else:
            strengths, p_values = np.ones((count, count)), np.zeros((count, count))
            for first, second in itertools.combinations(range(count), 2):
                result = stats.kendalltau(values[:, first], values[:, second])
                strengths[first, second] = strengths[second, first] = result.statistic
                p_values[first, second] = p_values[second, first] = result.pvalue
    strengths = np.asarray(strengths, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    np.fill_diagonal(strengths, 1.0)
    np.fill_diagonal(p_values, np.nan)
    return strengths, p_values


def _significant(p_values: Array, level: float, correction: str | None) -> tuple[list[tuple[int, int]], float]:
    """Return the pairs that pass the level after a correction, and the threshold.

    The threshold is the p-value below which a pair passes, and a pair with a `nan` p-value counts as a test that
    never passes.
    """
    first, second = np.triu_indices(p_values.shape[0], 1)
    tested = np.nan_to_num(p_values[first, second], nan=1.0)
    count = tested.size
    if correction == 'bonferroni':
        threshold = level / count if count else level
        kept = tested < threshold
    elif correction == 'benjamini-hochberg' and count:
        kept = stats.false_discovery_control(tested, method='bh') < level
        threshold = int(kept.sum()) * level / count
    else:
        threshold = level
        kept = tested < threshold
    return [(int(row), int(column)) for row, column in zip(first[kept], second[kept], strict=True)], float(threshold)


def _spanning_tree(strengths: Array) -> list[tuple[int, int]]:
    """Return the pairs of the maximum spanning tree of the signed correlations.

    Pairs with a `nan` correlation are left out of the complete network.
    """
    complete = nx.Graph()
    complete.add_nodes_from(range(strengths.shape[0]))
    for first, second in itertools.combinations(range(strengths.shape[0]), 2):
        if np.isfinite(strengths[first, second]):
            complete.add_edge(first, second, strength=float(strengths[first, second]))
    tree = nx.maximum_spanning_tree(complete, weight='strength')
    return sorted((min(first, second), max(first, second)) for first, second in tree.edges())


def _disparity(strengths: Array, level: float) -> list[tuple[int, int]]:
    """Return the pairs that the disparity filter keeps at a level.

    A pair is kept when `(1 - w / s) ** (k - 1)` is below the level at one of its ends, with `w` its absolute
    correlation, `s` the sum of the absolute correlations of the end, and `k` its number of pairs.
    """
    weights = np.abs(strengths)
    np.fill_diagonal(weights, np.nan)
    finite = np.isfinite(weights)
    weights = np.where(finite, weights, 0.0)
    sums = weights.sum(axis=1, keepdims=True)
    degrees = finite.sum(axis=1, keepdims=True)
    shares = np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0)
    alphas = np.where(finite, (1 - shares) ** (degrees - 1), 1.0)
    kept = (alphas < level) | (alphas < level).T
    first, second = np.nonzero(np.triu(kept, 1))
    return [(int(row), int(column)) for row, column in zip(first, second, strict=True)]
