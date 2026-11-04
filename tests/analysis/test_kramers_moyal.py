"""Test the Kramers-Moyal analysis of drift and diffusion."""

import numpy as np
import pytest

from skcomplex.analysis import KramersMoyalAnalysis
from skcomplex.errors import AnalysisError


def _logistic_series(size: int) -> np.ndarray:
    """Return iterates of the logistic map, which plain float arithmetic repeats exactly on every platform."""
    values = []
    x = 0.3
    for _ in range(size):
        x = 3.9 * x * (1 - x)
        values.append(x)
    return np.array(values)


@pytest.mark.parametrize(
    ('bandwidth', 'positions', 'drift', 'diffusion'),
    [
        (
            0.15,
            [
                0.010615989854119295,
                0.14171994299554763,
                0.27282389613697594,
                0.4039278492784043,
                0.5350318024198326,
                0.666135755561261,
                0.7972397087026895,
                0.9283436618441178,
                1.059447614985546,
            ],
            [
                30.706503786143678,
                33.15020146906587,
                46.5326551649683,
                51.74004329528832,
                43.76744730125744,
                17.50869024864151,
                -40.26769533677231,
                -66.42397245939407,
                -68.87351355149232,
            ],
            [
                4.898653237730481,
                5.866386149065086,
                11.248055058643995,
                13.458913184606649,
                10.083095530411532,
                4.16228245152115,
                12.97444165078101,
                23.788548706491063,
                24.959587717977872,
            ],
        ),
        (
            0.4,
            [
                -0.21160623236810294,
                -0.024946723671119053,
                0.16171278502586484,
                0.3483722937228487,
                0.5350318024198326,
                0.7216913111168164,
                0.9083508198138004,
                1.0950103285107844,
                1.2816698372077682,
            ],
            [
                33.66103349657743,
                36.55296660140587,
                42.58157618074476,
                43.88096119259263,
                29.406357088954504,
                -22.953814730157735,
                -49.37521649233975,
                -61.168860367538656,
                -63.62291896423686,
            ],
            [
                5.992527750538118,
                7.201026392129172,
                9.698122710443238,
                10.30122422710511,
                11.335151965179172,
                15.79946690593396,
                18.782939800906544,
                21.46115801776948,
                22.18203676343565,
            ],
        ),
        (
            None,
            [
                0.05743316375045939,
                0.1768328234178027,
                0.29623248308514605,
                0.41563214275248933,
                0.5350318024198326,
                0.654431462087176,
                0.7738311217545193,
                0.8932307814218626,
                1.012630441089206,
            ],
            [
                25.223899574100066,
                38.75459192522548,
                52.73387590996795,
                53.06448981582072,
                44.48520532483936,
                21.503005182738065,
                -14.424376400074603,
                -54.63310545315911,
                -84.81869432757651,
            ],
            [
                3.1887840663122633,
                7.64942649741778,
                13.9157152234052,
                14.084262830459243,
                9.985612181340969,
                2.6239361207397125,
                1.6324853738947,
                15.867123828393705,
                36.01981890174001,
            ],
        ),
    ],
)
def test_kramers_moyal_keeps_its_reference_values(bandwidth, positions, drift, diffusion):
    """Test that the positions and the coefficients keep values recorded from the previous library."""
    analysis = KramersMoyalAnalysis(dt=0.01, bins=9, bandwidth=bandwidth).check().prepare(_logistic_series(400)).run()
    assert analysis.positions_ == pytest.approx(positions, rel=1e-6)
    assert analysis.drift_ == pytest.approx(drift, rel=1e-6)
    assert analysis.diffusion_ == pytest.approx(diffusion, rel=1e-6)


def test_kramers_moyal_ornstein_uhlenbeck_coefficients():
    """Test that 10^6 steps of an Ornstein-Uhlenbeck process give drift -x within 0.1 and diffusion 1 within 0.05."""
    rng = np.random.default_rng(2)
    dt = 0.02
    noise = rng.standard_normal(1_000_000) * np.sqrt(2 * dt)
    values = np.empty(noise.size)
    values[0] = 0.0
    for t in range(1, values.size):
        values[t] = values[t - 1] - values[t - 1] * dt + noise[t]
    coefficients = KramersMoyalAnalysis(dt=dt, bins=41).check().prepare(values).run()
    inside = np.abs(coefficients.positions_) < 1.5
    assert np.max(np.abs(coefficients.drift_[inside] + coefficients.positions_[inside])) < 0.1
    assert np.max(np.abs(coefficients.diffusion_[inside] - 1)) < 0.05


def test_kramers_moyal_reads_one_series():
    """Test that two series and a negative step raise errors."""
    with pytest.raises(AnalysisError, match='reads one series'):
        KramersMoyalAnalysis().check().prepare(np.zeros((10, 2)))
    with pytest.raises(AnalysisError, match='dt'):
        KramersMoyalAnalysis(dt=-1.0).check()
