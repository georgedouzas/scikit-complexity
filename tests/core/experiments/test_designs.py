"""Test the `_designs` module of `skcomplex.core.experiments`."""

import pytest

from skcomplex.core.experiments import (
    Grid,
    LatinHypercube,
    RandomDesign,
    Rows,
    SobolDesign,
)


@pytest.mark.parametrize(
    'design',
    [
        RandomDesign(16, r=(0.2, 1.0), random_state=0),
        LatinHypercube(16, r=(0.2, 1.0), random_state=0),
        SobolDesign(16, r=(0.2, 1.0), random_state=0),
    ],
    ids=['random', 'latin', 'sobol'],
)
def test_sampled_designs_stay_in_bounds(design):
    """Test that sampled designs give the requested number of rows within the bounds."""
    values = [row['r'] for row in design.rows()]
    assert len(values) == 16
    assert all(0.2 <= value <= 1.0 for value in values)


def test_designs_cross_and_concatenate():
    """Test that designs cross with `*` and concatenate with `+`."""
    crossed = Grid(r=[1, 2]) * Grid(K=[10, 20, 30])
    joined = Grid(r=[1, 2]) + Rows([{'r': 3}])
    assert len(crossed.rows()) == 6
    assert [row['r'] for row in joined.rows()] == [1, 2, 3]
