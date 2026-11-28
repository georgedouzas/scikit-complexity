"""Test the networks built from the correlations of many series."""

import itertools

import networkx as nx
import numpy as np
import pytest
from scipy import stats

from skcomplex.core.analysis import CorrelationNetworkResult, correlation_network, network_measures
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.mathematics.dynamical_systems import LorenzSystem

GROUP = 5
STEPS = 500


def _two_groups(seed: int = 1) -> np.ndarray:
    """Return 10 series in two groups of 5, each group sharing one Gaussian factor added to Gaussian noise."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((STEPS, 2))
    return np.repeat(factors, GROUP, axis=1) + rng.standard_normal((STEPS, 2 * GROUP))


def _crosses(link: tuple[str, str]) -> bool:
    """Return whether a link of the two groups joins the groups."""
    return int(link[0]) // GROUP != int(link[1]) // GROUP


def _mixed(seed: int = 0) -> np.ndarray:
    """Return 8 series with correlations of mixed strength."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((200, 8))
    return values + 0.3 * values[:, [0]] * np.arange(8) / 8


def test_two_groups_link_within_groups_only():
    """Test that two groups of series give the 20 pairs within a group and none across, for every method."""
    for method in ('pearson', 'spearman', 'kendall'):
        result = correlation_network(_two_groups(), method=method)
        assert isinstance(result, CorrelationNetworkResult)
        assert len(result.links) == 20
        assert not any(_crosses(link) for link in result.links)


def test_pearson_correlations_and_p_values_match_scipy():
    """Test that the Pearson correlation and p-value of every pair equal those of scipy.stats.pearsonr."""
    values = _mixed()
    result = correlation_network(values, method='pearson')
    for first, second in itertools.combinations(range(values.shape[1]), 2):
        expected = stats.pearsonr(values[:, first], values[:, second])
        assert result.strengths[first, second] == pytest.approx(expected.statistic, abs=1e-12)
        assert result.p_values[first, second] == pytest.approx(expected.pvalue, abs=1e-12)
    assert np.all(np.diag(result.strengths) == 1)
    assert np.all(np.isnan(np.diag(result.p_values)))


def test_labels_of_a_trajectory_follow_the_variables():
    """Test that the labels and the nodes of a trajectory network are the three named series in their order."""
    names = ['lorenz.z', 'lorenz.x', 'lorenz.y']
    result = correlation_network(LorenzSystem().prepare().simulate(until=5.0), variables=names)
    assert result.labels == names
    assert list(result.graph.nodes) == names


def test_graph_is_read_by_network_measures():
    """Test that the graph is undirected with the absolute correlation as weight and the signed one beside it."""
    values = _two_groups()
    values[:, 1] = -values[:, 1]
    result = correlation_network(values)
    assert not result.graph.is_directed()
    for first, second, data in result.graph.edges(data=True):
        row, column = int(first), int(second)
        assert data['weight'] == abs(result.strengths[row, column])
        assert data['strength'] == result.strengths[row, column]
        assert data['p_value'] == result.p_values[row, column]
    assert result.graph['0']['1']['strength'] < 0
    communities = network_measures(result.graph, kind='communities', random_state=0)
    assert communities.modularity[0] > 0.4


def test_unknown_method_lists_the_methods():
    """Test that an unknown method raises an error that lists the three methods."""
    with pytest.raises(AnalysisError, match="pearson, spearman, kendall"):
        correlation_network(_mixed(), method='distance')


def test_bonferroni_threshold():
    """Test that Bonferroni over 20 series sets the threshold 0.05 over 190 and keeps only pairs below it."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal((300, 20)) + 0.4 * rng.standard_normal((300, 1))
    result = correlation_network(values, correction='bonferroni', level=0.05)
    assert result.threshold == pytest.approx(0.05 / 190)
    assert result.links
    assert all(result.graph.edges[link]['p_value'] < result.threshold for link in result.links)


def test_benjamini_hochberg_links_match_scipy():
    """Test that the default links are the pairs whose adjusted p-value is below the level."""
    result = correlation_network(_mixed())
    first, second = np.triu_indices(len(result.labels), 1)
    adjusted = stats.false_discovery_control(result.p_values[first, second], method='bh')
    expected = [
        (str(row), str(column)) for row, column, value in zip(first, second, adjusted, strict=True) if value < 0.05
    ]
    assert result.links == expected
    assert result.threshold == pytest.approx(len(expected) * 0.05 / first.size)


def test_independent_series_rarely_link():
    """Test that independent white noise series hold a Bonferroni link in at most a tenth of the networks."""
    networks = [np.random.default_rng(seed).standard_normal((1000, 20)) for seed in range(200)]
    linked = [bool(correlation_network(values, correction='bonferroni').links) for values in networks]
    assert np.mean(linked) <= 0.1


def test_corrections_are_nested():
    """Test that Bonferroni links are Benjamini-Hochberg links, which are uncorrected links."""
    values = _mixed()
    none, hochberg, bonferroni = (
        set(correlation_network(values, correction=correction).links)
        for correction in (None, 'benjamini-hochberg', 'bonferroni')
    )
    assert bonferroni <= hochberg <= none
    assert none - bonferroni


def test_spanning_tree_of_two_groups():
    """Test that the spanning tree of two groups holds 9 links, of which one joins the groups."""
    result = correlation_network(_two_groups(), backbone='spanning-tree')
    assert len(result.links) == 9
    assert sum(_crosses(link) for link in result.links) == 1
    assert nx.is_tree(result.graph)
    assert np.isnan(result.threshold)


def test_disparity_filter_of_two_groups():
    """Test that the disparity backbone of two groups at level 0.2 holds the pairs within a group only."""
    result = correlation_network(_two_groups(), backbone='disparity', level=0.2)
    assert len(result.links) == 20
    assert not any(_crosses(link) for link in result.links)


def test_disparity_significance():
    """Test that a pair is a disparity link exactly when its significance is below the level at one end."""
    result = correlation_network(_mixed(), backbone='disparity', level=0.3)
    weights = np.abs(result.strengths) - np.eye(len(result.labels))
    sums = weights.sum(axis=1)
    alphas = (1 - weights / sums[:, None]) ** (len(result.labels) - 2)
    expected = [
        (str(first), str(second))
        for first, second in itertools.combinations(range(len(result.labels)), 2)
        if min(alphas[first, second], alphas[second, first]) < 0.3
    ]
    assert result.links == expected
    assert expected


def test_spanning_tree_does_not_read_the_level():
    """Test that the spanning tree with a level raises an error that names the level."""
    with pytest.raises(AnalysisError, match='does not read level'):
        correlation_network(_mixed(), backbone='spanning-tree', level=0.01)


def test_disparity_does_not_read_the_correction():
    """Test that the disparity filter with a correction raises an error that names the correction."""
    with pytest.raises(AnalysisError, match='does not read correction'):
        correlation_network(_mixed(), backbone='disparity', correction='bonferroni')


def test_missing_values_and_constant_series():
    """Test that rows with a missing value are dropped and that a constant series has no correlation and no link."""
    values = _two_groups()
    gapped = values.copy()
    gapped[3, 2] = np.nan
    dropped = correlation_network(np.delete(values, 3, axis=0))
    assert np.array_equal(correlation_network(gapped).strengths, dropped.strengths)
    values[:, 0] = 1.0
    result = correlation_network(values, correction=None)
    assert np.isnan(result.strengths[0, 1:]).all()
    assert result.graph.degree['0'] == 0
