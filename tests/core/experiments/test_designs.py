"""Test the `_designs` module of `skcomplex.core.experiments`."""

import numpy as np
import pytest
from scipy import stats

from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments._designs import _get_rows, _get_sensitivity_rows


def test_grid_runs_every_combination():
    """Test that a grid gives every combination of the values, the last name varying fastest."""
    rows = _get_rows({'r': [1, 2], 'K': [10, 20, 30]}, 'grid', 0, None)
    assert len(rows) == 6
    assert rows[:2] == [{'r': 1, 'K': 10}, {'r': 1, 'K': 20}]


def test_grid_rejects_a_value_that_is_not_a_sequence():
    """Test that a grid of one number instead of a sequence raises an error naming it."""
    with pytest.raises(ValidationError, match='sequence of values per name, but r'):
        _get_rows({'r': 0.5}, 'grid', 0, None)


def test_rows_run_as_they_are():
    """Test that given rows are copied as they are, and that rows that are not mappings or a sampled design fail."""
    assert _get_rows([{'K': 10}, {'K': 20}], 'grid', 0, None) == [{'K': 10}, {'K': 20}]
    with pytest.raises(ValidationError, match='mapping from name to value'):
        _get_rows([1.0], 'grid', 0, None)
    with pytest.raises(ValidationError, match='takes no sobol design'):
        _get_rows([{'K': 10}], 'sobol', 4, None)


@pytest.mark.parametrize('design', ['random', 'latin_hypercube', 'sobol'])
def test_sampled_designs_stay_within_bounds_and_are_seeded(design):
    """Test that a sampled design gives `n` rows within the bounds, the same for the same seed."""
    bounds = {'r': (0.5, 1.5), 'K': (100, 1000)}
    rows = _get_rows(bounds, design, 16, 3)
    assert len(rows) == 16
    assert all(0.5 <= row['r'] <= 1.5 and 100 <= row['K'] <= 1000 for row in rows)
    assert rows == _get_rows(bounds, design, 16, 3)


def test_latin_hypercube_fills_every_stratum():
    """Test that a Latin hypercube puts one sample in every stratum."""
    rows = _get_rows({'r': (0.0, 1.0)}, 'latin_hypercube', 4, 0)
    assert sorted(int(row['r'] * 4) for row in rows) == [0, 1, 2, 3]


def test_sampled_design_reads_distributions():
    """Test that a sampled design maps its samples through the quantile function of a SciPy distribution."""
    rows = _get_rows({'r': stats.norm(10, 1)}, 'random', 200, 0)
    assert 9.5 < sum(row['r'] for row in rows) / len(rows) < 10.5


@pytest.mark.parametrize('bound', [(0.0, 1.0, 2.0), 'low', 3.0], ids=['triple', 'text', 'number'])
def test_sampled_design_rejects_invalid_bounds(bound):
    """Test that a bound that is neither a pair nor a distribution raises an error naming it."""
    with pytest.raises(ValidationError, match='The bounds of r must be'):
        _get_rows({'r': bound}, 'random', 4, None)


def test_unknown_design_and_negative_size_are_rejected():
    """Test that an unknown design and a negative number of rows raise named errors."""
    with pytest.raises(ValidationError, match="Unknown design 'halton'"):
        _get_rows({'r': (0.0, 1.0)}, 'halton', 4, None)
    with pytest.raises(ValidationError, match='nonnegative integer'):
        _get_rows({'r': (0.0, 1.0)}, 'random', -1, None)


def test_one_at_a_time_rows_move_one_name_from_low_to_high():
    """Test that one-at-a-time rows move every name between its bounds with the others at their midpoints."""
    rows = _get_sensitivity_rows({'r': (0.0, 1.0), 'K': (50, 150)}, 'oat', 0, None)
    assert rows == [{'r': 0.0, 'K': 100.0}, {'r': 1.0, 'K': 100.0}, {'r': 0.5, 'K': 50.0}, {'r': 0.5, 'K': 150.0}]


def test_morris_rows_move_one_name_per_step():
    """Test that every step of a Morris trajectory changes exactly one name and stays within the bounds."""
    rows = _get_sensitivity_rows({'a': (0.0, 1.0), 'b': (10.0, 20.0)}, 'morris', 5, 0)
    values = np.asarray([[row['a'], row['b']] for row in rows]).reshape(5, 3, 2)
    assert ((np.diff(values, axis=1) != 0).sum(axis=2) == 1).all()
    assert values[..., 0].min() >= 0.0
    assert values[..., 1].max() <= 20.0


def test_saltelli_rows_hold_base_points_and_their_mixes():
    """Test that Saltelli's sample of 8 base points over two names has 32 rows."""
    assert len(_get_sensitivity_rows({'r': (0.0, 1.0), 'K': (50, 150)}, 'sobol', 8, 0)) == 32
