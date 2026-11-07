"""Test the `_elementary_flux_modes` module of `skcomplex.systems.biology.systems_biology`."""

import time
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import nnls

from skcomplex.core.analysis import elementary_modes
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.system import System
from skcomplex.core.errors import EnumerationLimitError
from skcomplex.systems.biology.common.metabolism import load_network
from skcomplex.systems.biology.systems_biology import elementary_flux_modes


def _analysed(system):
    """Return the elementary modes of a system, normalised to the uptake."""
    return elementary_modes(system, normalize='cell.v_b1')


def _column(analysis, name):
    """Return the column of a flux of the modes."""
    return analysis.modes[:, list(analysis.labels).index(f'cell.v_{name}')]


def test_modes_have_one_yield_and_decompose_the_optimum():
    """Test the 2 modes of yield 1/12, and that the optimum of growth 50 is their non-negative combination."""
    analysis = _analysed(elementary_flux_modes())
    assert analysis.modes.shape[0] == 2
    assert set(_column(analysis, 'vbio')) == {Fraction(1, 12)}
    assert set(_column(analysis, 'v6') / _column(analysis, 'vbio')) == {Fraction(20)}
    names = [item.rpartition('v_')[2] for item in analysis.labels]
    values = dict(
        zip(
            ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'b1', 'b2', 'vbio'),
            (600, 0, 0, 550, 50, 1000, 100, 600, 1000, 50),
            strict=True,
        ),
    )
    optimum = np.array([values[name] for name in names], dtype=float)
    _, residual = nnls(np.asarray(analysis.modes, dtype=float).T, optimum)
    assert residual < 1e-9


def test_knock_out_leaves_the_second_route():
    """Test that with `v1` closed one mode remains, through `v2` and `v3`, with the same yield."""
    analysis = _analysed(elementary_flux_modes(knockouts=('v1',)))
    assert analysis.modes.shape[0] == 1
    assert _column(analysis, 'v2')[0] == Fraction(1)
    assert _column(analysis, 'v3')[0] == Fraction(1)
    assert _column(analysis, 'vbio')[0] == Fraction(1, 12)


def test_genome_scale_network_stops_at_the_limit():
    """Test that the modes of the E. coli core system raise the limit error within one second of a limit of 5 s."""
    system = System(elements=[load_network('textbook', 'cell')], time=StaticTime())
    start = time.perf_counter()
    with pytest.raises(EnumerationLimitError, match='variables'):
        elementary_modes(system, timeout=5.0)
    assert time.perf_counter() - start < 6.0
