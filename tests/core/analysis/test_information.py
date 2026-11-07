"""Test the information analysis: symbols, mutual information, transfer, storage, and total correlations."""

import itertools
import math
from dataclasses import FrozenInstanceError

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import solve_ivp

from skcomplex.core.analysis import InformationResult, information
from skcomplex.core.errors import AnalysisError

SAMPLES = 20_000


def _analysis(kind: str, source: object, **settings: object) -> InformationResult:
    """Return the run information analysis of a source for one kind."""
    return information(source, kind=kind, **settings)


def _information(kind: str, source: object, **settings: object) -> np.ndarray | float:
    """Return the information of a source for one kind."""
    return _analysis(kind, source, **settings).information


def _pair(kind: str, first: npt.ArrayLike, second: npt.ArrayLike, **settings: object) -> float:
    """Return the information from a first series to a second one."""
    return float(np.asarray(_information(kind, np.column_stack([first, second]), **settings))[0, 1])


def _bits(seed: int, columns: int) -> np.ndarray:
    """Return fair independent bits, one column per variable."""
    return np.random.default_rng(seed).integers(0, 2, (SAMPLES, columns))


def _logistic(r: float, size: int, start: float = 0.3) -> np.ndarray:
    """Return iterates of the logistic map after a transient of 500."""
    values = np.empty(size + 500)
    values[0] = start
    for t in range(1, values.size):
        values[t] = r * values[t - 1] * (1 - values[t - 1])
    return values[500:]


def _coupled_rossler(coupling: float, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the first coordinates of a Rössler driver and a detuned response coupled one way, sampled every 1."""

    def drift(_, state):
        x1, y1, z1, x2, y2, z2 = state
        return [
            -y1 - z1,
            x1 + 0.2 * y1,
            0.2 + z1 * (x1 - 5.7),
            -0.95 * y2 - z2 + coupling * (x1 - x2),
            0.95 * x2 + 0.2 * y2,
            0.2 + z2 * (x2 - 5.7),
        ]

    times = np.arange(100, 100 + samples)
    solution = solve_ivp(drift, (0, times[-1]), [1, 1, 0, -2, 0.5, 0], t_eval=times, rtol=1e-8)
    return solution.y[0], solution.y[3]


def test_periodic_sequence():
    """Test that the repeated sequence 0011 has its correlation information at length 3."""
    sequence = [0, 0, 1, 1] * 500
    information = _information('correlation-information', sequence, dimension=4)[0]
    assert information == pytest.approx([0.0, 1.0, 0.0], abs=1e-5)


def test_information_flows_from_driver():
    """Test that a series copying a random series with lag 1 receives 1 bit from it and gives none back."""
    driver = np.random.default_rng(1).integers(0, 2, 20_000)
    follower = np.roll(driver, 1)
    information = _information('transfer', np.column_stack([driver, follower]))
    assert information[0, 1] == pytest.approx(1.0, abs=0.01)
    assert information[1, 0] == pytest.approx(0.0, abs=0.01)
    assert np.isnan(np.diag(information)).all()


def test_entropies_and_divergences():
    """Test the KL divergence and the mutual information of independent symbols."""
    divergences = _information('kl-divergence', np.asarray([[1.0, 0.5], [0.0, 0.5]]))
    assert divergences[0, 1] == pytest.approx(1.0)
    assert divergences[1, 0] == float('inf')
    assert np.diag(divergences).tolist() == [0.0, 0.0]
    independent = np.random.default_rng(2).integers(0, 2, (2, 50_000))
    assert _pair('mutual', *independent) == pytest.approx(0.0, abs=0.001)


def test_mutual_information_of_a_series_with_itself_is_its_entropy():
    """Test that the diagonal of the mutual information holds the entropy of every series."""
    assert _information('mutual', [0, 1, 0, 1]).tolist() == [[1.0]]


def test_symbolisation_and_permutations():
    """Test quantile and bin symbols and the error of an unknown symbolisation."""
    assert _analysis('mutual', [1.0, 2.0, 3.0, 4.0], symbols='bins').symbols[:, 0].tolist() == [0, 0, 1, 1]
    noise = np.random.default_rng(3).normal(size=1000)
    symbols = _analysis('mutual', noise, symbols='quantiles', bins=4).symbols[:, 0]
    assert np.bincount(symbols).tolist() == [250] * 4
    with pytest.raises(AnalysisError, match=r"symbols='words'"):
        information([0, 1, 0, 1], symbols='words')


def test_symbolize_long_ordinal_patterns():
    """Test that ordinal patterns of length 12 of white noise give one symbol per distinct rank pattern."""
    noise = np.random.default_rng(9).standard_normal(20_000)
    symbols = _analysis('mutual', noise, symbols='ordinal', order=12).symbols[:, 0]
    patterns = [
        tuple(np.argsort(window, kind='stable')) for window in np.lib.stride_tricks.sliding_window_view(noise, 12)
    ]
    pairs = set(zip(symbols.tolist(), patterns, strict=True))
    assert len(pairs) == len(set(symbols.tolist())) == len(set(patterns))


def test_ordinal_patterns_rank_every_permutation():
    """Test that the patterns of length 3 are the lexicographic ranks 0 to 5."""
    ranks = []
    for permutation in itertools.permutations(range(3)):
        values = np.empty(3)
        values[list(permutation)] = np.arange(3)
        ranks.append(int(_analysis('mutual', values, symbols='ordinal').symbols[0, 0]))
    assert ranks == list(range(6))


def test_auto_mutual_information_of_noise():
    """Test that white noise in 4 quantiles holds 2 bits at delay 0 and nothing at delay 1."""
    noise = np.random.default_rng(0).standard_normal(10_000)
    information = _information('auto-mutual', noise, symbols='quantiles', bins=4, max_delay=2)[0]
    assert information[0] == pytest.approx(2.0, abs=0.01)
    assert information[1] < 0.01


def test_active_information_storage_of_a_periodic_sequence():
    """Test that the sequence 0011 stores 1 bit with history 2 and a fair coin stores almost nothing."""
    assert _information('active-storage', [0, 0, 1, 1] * 500, history=2)[0] == pytest.approx(1, abs=1e-5)
    coin = np.random.default_rng(6).integers(0, 2, 100_000)
    assert _information('active-storage', coin, history=2)[0] < 0.01


def test_transfer_entropy_linear_gaussian():
    """Test that the coupled autoregression gives the Granger value in nats and no reverse transfer."""
    rng = np.random.default_rng(7)
    size = 20_000
    x, noise = rng.standard_normal(size), rng.standard_normal(size)
    y = np.zeros(size)
    for t in range(1, size):
        y[t] = 0.5 * y[t - 1] + 0.4 * x[t - 1] + noise[t]
    information = _information('transfer', np.column_stack([x, y]), estimator='knn', base=math.e)
    assert information[0, 1] == pytest.approx(0.5 * math.log(1.16), abs=0.02)
    assert information[1, 0] < 0.01


def test_transfer_entropy_directed_coupling_of_two_domains():
    """Test that symbolized logistic maps and continuous Rössler oscillators coupled one way show the direction."""
    size = 20_000
    driver = _logistic(4.0, size)
    response = np.empty(size)
    response[0] = 0.6
    for t in range(1, size):
        response[t] = 0.8 * 4 * response[t - 1] * (1 - response[t - 1]) + 0.2 * 4 * driver[t - 1] * (1 - driver[t - 1])
    information = _information('transfer', np.column_stack([driver, response]), symbols='bins')
    assert information[0, 1] > 5 * information[1, 0]
    source, target = _coupled_rossler(0.2, 12_000)
    information = _information('transfer', np.column_stack([source, target]), estimator='knn', history=2, k=8)
    assert information[0, 1] > 5 * max(information[1, 0], 0.0)
    assert information[0, 1] > 0.1


def test_transfer_entropy_rejects_unknown_estimators():
    """Test that an unknown estimator raises an error naming the known ones."""
    with pytest.raises(AnalysisError, match='plugin, knn'):
        information([0, 1, 0, 1], kind='transfer', estimator='kernel')
    with pytest.raises(AnalysisError, match='plugin, knn'):
        information([0, 1, 0, 1], kind='transfer', estimator='gaussian')


def test_o_information_xor_is_synergistic():
    """Test that X, Y, and Z = X xor Y with fair independent bits have an O-information of -1 bit."""
    bits = _bits(0, 2)
    data = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
    assert _information('o-information', data) == pytest.approx(-1.0, abs=0.01)
    assert _information('o-information-gradients', data) == pytest.approx([-1.0, -1.0, -1.0], abs=0.01)


def test_o_information_copies_is_redundant():
    """Test that three copies of one fair bit have an O-information of +1 bit."""
    bit = _bits(1, 1)
    data = np.column_stack([bit, bit, bit])
    assert _information('o-information', data) == pytest.approx(1.0, abs=0.01)
    assert _information('total-correlation', data) == pytest.approx(2.0, abs=0.01)
    assert _information('dual-total-correlation', data) == pytest.approx(1.0, abs=0.01)


def test_total_correlation_independent_variables_is_zero():
    """Test that independent bits share no information under any measure."""
    data = _bits(2, 4)
    assert _information('total-correlation', data) == pytest.approx(0.0, abs=0.01)
    assert _information('dual-total-correlation', data) == pytest.approx(0.0, abs=0.01)


def test_total_correlation_gaussian_matches_closed_form():
    """Test that the Gaussian total correlation of two correlated Gaussians is `-log2(1 - r^2) / 2`."""
    rng = np.random.default_rng(3)
    correlation = 0.6
    data = rng.multivariate_normal([0.0, 0.0], [[1.0, correlation], [correlation, 1.0]], SAMPLES)
    sample = np.corrcoef(data, rowvar=False)[0, 1]
    expected = -np.log2(1 - sample**2) / 2
    assert _information('total-correlation', data, estimator='gaussian') == pytest.approx(expected, abs=1e-9)
    assert _information('dual-total-correlation', data, estimator='gaussian') == pytest.approx(expected, abs=1e-9)
    assert _information('o-information', data, estimator='gaussian') == pytest.approx(0.0, abs=1e-9)


def test_o_information_gaussian_common_driver_is_redundant():
    """Test that three noisy copies of one Gaussian are redundant under the Gaussian and neighbour estimators."""
    rng = np.random.default_rng(4)
    driver = rng.normal(size=4000)
    data = np.column_stack([driver + 0.3 * rng.normal(size=driver.size) for _ in range(3)])
    gaussian = _information('o-information', data, estimator='gaussian')
    assert gaussian > 0.5
    assert _information('o-information', data, estimator='knn') == pytest.approx(gaussian, abs=0.15)


def test_o_information_gaussian_sum_is_synergistic():
    """Test that two independent Gaussians and their noisy sum are synergistic under the Gaussian estimator."""
    rng = np.random.default_rng(5)
    first, second = rng.normal(size=(2, SAMPLES))
    data = np.column_stack([first, second, first + second + 0.1 * rng.normal(size=SAMPLES)])
    assert _information('o-information', data, estimator='gaussian') < -0.5


def test_total_correlation_unknown_estimator():
    """Test that an unknown estimator raises an error naming the known ones."""
    with pytest.raises(AnalysisError, match='plugin, gaussian, knn'):
        information([0, 1, 0, 1], kind='total-correlation', estimator='binned')


def test_unknown_kind_lists_the_kinds():
    """Test that an unknown kind names the kinds of the class."""
    with pytest.raises(AnalysisError, match='mutual, auto-mutual, transfer'):
        information([0, 1, 0, 1], kind='unknown')


def test_ignored_settings_are_rejected():
    """Test that a setting the kind or the symbolization does not read raises an error naming it."""
    with pytest.raises(AnalysisError, match='does not read lag'):
        information([0, 1, 0, 1], kind='mutual', lag=2)
    with pytest.raises(AnalysisError, match='does not read bins'):
        information([0, 1, 0, 1], kind='mutual', bins=4)
    with pytest.raises(AnalysisError, match='does not read symbols'):
        information([0, 1, 0, 1], kind='kl-divergence', symbols='bins')


def test_negative_history_is_rejected():
    """Test that a history below 1 raises an error naming it."""
    with pytest.raises(AnalysisError, match='history=0'):
        information([0, 1, 0, 1], kind='transfer', history=0)


def test_result_is_frozen():
    """Test that the result of an information measure is a frozen value."""
    result = _analysis('transfer', _bits(0, 2), history=2)
    with pytest.raises(FrozenInstanceError):
        result.information = 0.0
