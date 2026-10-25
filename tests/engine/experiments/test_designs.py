"""Test the `_designs` module of `skcomplex.engine.experiments`."""

import pytest

from skcomplex.core.base import ValidationError
from skcomplex.engine.experiments import (
    GridDesign,
    LatinHypercubeDesign,
    RandomDesign,
    RowsDesign,
    SobolDesign,
)


@pytest.mark.parametrize(
    'design',
    [
        RandomDesign(16, r=(0.2, 1.0), random_state=0),
        LatinHypercubeDesign(16, r=(0.2, 1.0), random_state=0),
        SobolDesign(16, r=(0.2, 1.0), random_state=0),
    ],
    ids=['random', 'latin', 'sobol'],
)
def test_sampled_designs_stay_in_bounds(design):
    """Test that sampled designs give the requested number of rows within the bounds."""
    values = [row['r'] for row in design.prepare().run()]
    assert len(values) == 16
    assert all(0.2 <= value <= 1.0 for value in values)


def test_designs_cross_and_concatenate():
    """Test that designs cross with `product` and concatenate with `concat`."""
    crossed = GridDesign(r=[1, 2]).product(GridDesign(K=[10, 20, 30]))
    joined = GridDesign(r=[1, 2]).concat(RowsDesign([{'r': 3}]))
    assert len(crossed.prepare().run()) == 6
    assert [row['r'] for row in joined.prepare().run()] == [1, 2, 3]


def test_grid_and_rows_designs_combine_by_product():
    """Test that the product of a grid design and a rows design runs every combination of their rows."""
    rows = GridDesign(r=[1, 2]).product(RowsDesign([{'K': 10}, {'K': 20}])).prepare().run()
    assert rows == [{'r': 1, 'K': 10}, {'r': 1, 'K': 20}, {'r': 2, 'K': 10}, {'r': 2, 'K': 20}]


def test_design_prepare_sets_rows():
    """Test that preparing a design sets its rows and running returns a copy of them."""
    design = GridDesign(r=[1, 2]).prepare()
    assert design.rows_ == [{'r': 1}, {'r': 2}]
    design.run()[0]['r'] = 5
    assert design.rows_[0]['r'] == 1


@pytest.mark.parametrize(
    'design',
    [RandomDesign(-1, r=(0.0, 1.0)), RandomDesign(4, r=(0.0, 1.0, 2.0)), RowsDesign([1.0])],  # type: ignore[list-item,unused-ignore]
    ids=['negative-size', 'bad-bounds', 'bad-row'],
)
def test_design_check_rejects_invalid_arguments(design):
    """Test that an invalid design builds and fails when checked."""
    with pytest.raises(ValidationError):
        design.check()
