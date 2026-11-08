"""Test the `_delaunay` module of `skcomplex.core.components.spaces`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import ContinuousSpace, DelaunayNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.errors import BackendError
from skcomplex.core.specs import StateSpec


class CountingElement(Element):
    """Element whose state `out` sums the tags of its Delaunay neighbours, powers of two that name them."""

    def __init__(
        self,
        label: str,
        *,
        out: Annotated[object, StateSpec()] = 0.0,
        tag: Annotated[object, StateSpec()] = 1.0,
        x: Annotated[object, StateSpec(vector=True, role='position')] = 0.0,
    ) -> None:
        super().__init__(label)
        self.out = out
        self.tag = tag
        self.x = x

    def _get_laws(self, s):
        """Return the sum of the tags of the Delaunay neighbours."""
        return [UpdateLaw(s.out, neighbors(s.other.tag, over=DelaunayNeighborhood(), reduce='sum'))]


def _problem(positions, side, boundary, backend=None, tags=None):
    """Return the problem of members at positions in a square, tagged by ones or by given tags."""
    values = {'x': positions, 'tag': np.ones(len(positions)) if tags is None else tags}
    group = GroupElement('cells', element=CountingElement, size=len(positions), values=values)
    system = System(elements=[group], space=ContinuousSpace(extent=(side, side), boundary=boundary))
    return PassCompiler(backend=backend or NumpyBackend()).prepare(system).run().problem_


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
    values = {'x': rng.uniform(0.0, 10.0, (1000, 2)), 'tag': np.ones(1000)}
    group = GroupElement('cells', element=CountingElement, size=1000, values=values)
    system = System(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0), boundary='open'))
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.backend == 'numpy'
    assert np.all(_counts(problem) >= 2)
