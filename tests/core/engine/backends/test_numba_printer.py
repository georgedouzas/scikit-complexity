"""Test the `_numba_printer` module of `skcomplex.core.engine.backends`."""

import time

import numpy as np
import pytest

from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.systems.chemistry.pattern_formation import SchnakenbergSystem


def test_member_kernels_of_a_small_field_run_serially(tmp_path):
    """Test that the kernels of a 40 by 40 field compile without parallel threads."""
    backend = NumbaBackend(cache_dir=tmp_path)
    SchnakenbergSystem((40, 40), backend=backend).prepare().simulate(until=0.01)
    source = '\n'.join(path.read_text() for path in tmp_path.rglob('skcomplex_kernels_*.py'))
    assert 'parallel=False' in source
    assert 'parallel=True' not in source


@pytest.mark.slow
def test_schnakenberg_field_forms_turing_pattern_within_thirty_seconds():
    """Test that a 40 by 40 Schnakenberg field forms its pattern by time 40 in 30 s."""
    start = time.perf_counter()
    trajectory = SchnakenbergSystem((40, 40), Dv=40.0).prepare().simulate(until=40.0)
    elapsed = time.perf_counter() - start
    u = np.asarray(trajectory.get_values('cells.u'))[-1].reshape(40, 40)
    power = np.abs(np.fft.fft2(u - u.mean())) ** 2
    power[0, 0] = 0.0
    rows, columns = np.unravel_index(np.argmax(power), power.shape)
    wavelength = 40 / np.hypot(min(rows, 40 - rows), min(columns, 40 - columns))
    assert elapsed < 30.0
    assert u.std() > 0.5
    assert 10.0 < wavelength < 20.0
