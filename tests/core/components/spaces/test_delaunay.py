"""Test the `_delaunay` module of `skcomplex.core.components.spaces`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import ContinuousSpace, DelaunayNeighborhood
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.errors import BackendError
from skcomplex.core.systems import System


def _count(s):
    """Return the sum of the tags of the Delaunay neighbours, powers of two that name them, as the value `out`."""
    return [UpdateLaw(s.target.out, neighbors(s.target.other.tag, over=DelaunayNeighborhood(), reduce='sum'))]


def _system(positions, side, boundary, backend='auto', tags=None):
    """Return members at positions in a square, tagged by ones or by given tags, that count their neighbours."""
    size = len(positions)
    values = {'out': np.zeros(size), 'tag': np.ones(size) if tags is None else tags, 'x': positions}
    cells = Element('cells', values=values, size=size, space=ContinuousSpace(extent=(side, side), boundary=boundary))
    counting = Interaction('counting', roles={'target': 'cells'}, laws=_count)
    return System(elements=[cells], interactions=[counting], backend=backend)


def _problem(positions, side, boundary, backend=None, tags=None):
    """Return the problem of members at positions in a square, tagged by ones or by given tags."""
    return _system(positions, side, boundary, backend or NumpyBackend(), tags).prepare()._problem


def _counts(problem, state=None):
    """Return the neighbour counts of every member after one update."""
    state = problem.initial_state if state is None else state
    return problem.states.read(problem.update(0.0, state, problem.initial_parameters), 'cells.out')


def test_delaunay_neighbours_of_a_square_and_its_centre():
    """Test that the centre of a square has four neighbours and every corner three."""
    positions = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [1.0, 1.0]])
    counts = _counts(_problem(positions, 2.0, 'open'))
    assert counts.tolist() == [3.0, 3.0, 3.0, 3.0, 4.0]


def test_delaunay_neighbours_cross_a_periodic_boundary():
    """Test that members near opposite edges of a periodic square are neighbours, unlike in an open square."""
    positions = np.array([[0.1, 2.5], [4.9, 2.5], [2.5, 2.5], [2.0, 1.0], [3.0, 4.0], [2.5, 0.2], [2.6, 4.7]])
    tags = 2.0 ** np.arange(len(positions))

    def neighbours_of_first(boundary):
        total = int(_counts(_problem(positions, 5.0, boundary, tags=tags))[0])
        return {index for index in range(len(positions)) if total >> index & 1}

    assert 1 in neighbours_of_first('periodic')
    assert 1 not in neighbours_of_first('open')


def test_delaunay_neighbours_follow_moving_members():
    """Test that the neighbours of a member at a new position are those of the new position."""
    positions = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [2.0, 2.0], [0.2, 2.0]])
    problem = _problem(positions, 4.0, 'open')
    moved = problem.initial_state.copy()
    _, indices = problem.states.locate('cells.x')
    values = moved[indices].reshape(-1, 2)
    values[5] = [3.8, 2.0]
    moved[indices] = values.ravel()
    fresh = _problem(values, 4.0, 'open')
    np.testing.assert_array_equal(_counts(problem, moved), _counts(fresh))


def test_delaunay_neighbourhood_does_not_run_on_numba():
    """Test that the Numba backend names the NumPy backend for Delaunay neighbourhoods."""
    positions = np.array([[0.5, 0.5], [1.5, 0.5], [1.0, 1.5]])
    with pytest.raises(BackendError, match='NumPy backend'):
        _problem(positions, 2.0, 'open', NumbaBackend())


def test_delaunay_neighbourhood_of_a_large_group_runs_on_numpy():
    """Test that a group of 1000 members, which defaults to Numba, runs Delaunay neighbourhoods on NumPy."""
    rng = np.random.default_rng(0)
    problem = _system(rng.uniform(0.0, 10.0, (1000, 2)), 10.0, 'open').prepare()._problem
    assert problem.backend == 'numpy'
    assert np.all(_counts(problem) >= 2)
