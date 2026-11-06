"""Test the matrix group spec."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import MatrixGroupSpec

GROUPS = [MatrixGroupSpec('SU', 2), MatrixGroupSpec('SU', 3), MatrixGroupSpec('U', 1), MatrixGroupSpec('U', 2)]


def _is_in(group, matrices):
    """Return whether matrices are unitary, and of unit determinant for a special unitary group."""
    inverse = np.conj(np.swapaxes(matrices, -1, -2))
    unitary = np.allclose(matrices @ inverse, np.eye(group.degree), atol=1e-12)
    return unitary and (group.name == 'U' or np.allclose(np.linalg.det(matrices), 1.0, atol=1e-12))


@pytest.mark.parametrize('group', GROUPS, ids=str)
def test_components_and_matrices_invert_each_other(group):
    """Test that the components of random matrices give back the matrices, with `2 n^2` components each."""
    matrices = group.uniform_of(np.random.default_rng(0), 5)
    components = group.components_of(matrices)
    assert components.shape == (5, group.size)
    assert np.array_equal(group.matrices_of(components), matrices)
    assert np.array_equal(group.matrices_of(group.identity), np.eye(group.degree))


@pytest.mark.parametrize('group', GROUPS, ids=str)
def test_draws_stay_in_the_group(group):
    """Test that Haar draws and moves near the identity lie in the group, and moves come close to the identity."""
    random = np.random.default_rng(1)
    assert _is_in(group, group.uniform_of(random, 20))
    moves = group.near_identity_of(random, 0.01, 20)
    assert _is_in(group, moves)
    assert np.abs(moves - np.eye(group.degree)).max() < 0.2


def test_haar_draws_of_su2_have_the_moments_of_the_measure():
    """Test that the trace of 20000 SU(2) draws has mean 0 within 0.02 and mean square 1 within 0.03."""
    traces = np.trace(MatrixGroupSpec('SU', 2).uniform_of(np.random.default_rng(2), 20000), axis1=1, axis2=2).real
    assert abs(traces.mean()) < 0.02
    assert abs((traces**2).mean() - 1.0) < 0.03


def test_moves_near_the_identity_are_symmetric():
    """Test that the mean of 20000 moves is Hermitian within 0.01, as a matrix and its inverse are equally likely."""
    moves = MatrixGroupSpec('SU', 3).near_identity_of(np.random.default_rng(3), 0.5, 20000)
    mean = moves.mean(axis=0)
    assert np.abs(mean - np.conj(mean.T)).max() < 0.01


@pytest.mark.parametrize('group', GROUPS, ids=str)
def test_projection_returns_matrices_of_the_group_and_keeps_them(group):
    """Test that a perturbed product is projected into the group, and that a matrix of the group stays as it is."""
    random = np.random.default_rng(4)
    matrices = group.uniform_of(random, 10)
    noisy = matrices + 1e-6 * random.normal(size=matrices.shape)
    projected = group.projection_of(noisy)
    assert _is_in(group, projected)
    assert np.abs(projected - matrices).max() < 1e-5
    assert np.allclose(group.projection_of(matrices), matrices, atol=1e-12)


def test_holonomy_is_ordered_and_does_not_commute():
    """Test that the holonomy of a path is the ordered product, and reversed paths give a different product."""
    group = MatrixGroupSpec('SU', 2)
    links = group.uniform_of(np.random.default_rng(5), 3)
    forward = group.holonomy_of(links, [0, 1, 2], [1, -1, 1])
    assert np.allclose(forward, links[0] @ np.conj(links[1].T) @ links[2])
    backward = group.holonomy_of(links, [2, 1, 0], [-1, 1, -1])
    assert np.allclose(backward, np.conj(forward.T))
    assert not np.allclose(group.holonomy_of(links, [1, 0], [1, 1]), group.holonomy_of(links, [0, 1], [1, 1]))


def test_holonomy_pads_with_the_identity_and_keeps_the_shape_of_the_paths():
    """Test that a sign of 0 contributes the identity, and that paths of any leading shape give one matrix each."""
    group = MatrixGroupSpec('SU', 3)
    links = group.uniform_of(np.random.default_rng(6), 4)
    padded = group.holonomy_of(links, [[3, 2, 0], [1, 0, 0]], [[1, 1, 0], [-1, 0, 0]])
    assert padded.shape == (2, 3, 3)
    assert np.allclose(padded[0], links[3] @ links[2])
    assert np.allclose(padded[1], np.conj(links[1].T))


def test_holonomy_trace_is_gauge_invariant():
    """Test that the trace of a closed loop is unchanged by a random gauge transformation of its nodes."""
    group = MatrixGroupSpec('SU', 3)
    random = np.random.default_rng(7)
    links = group.uniform_of(random, 4)
    gauge = group.uniform_of(random, 4)
    sources, targets = np.arange(4), (np.arange(4) + 1) % 4
    transformed = gauge[sources] @ links @ np.conj(np.swapaxes(gauge[targets], -1, -2))
    loops = [np.trace(group.holonomy_of(item, [0, 1, 2, 3], [1, 1, 1, 1])) for item in (links, transformed)]
    assert loops[0] == pytest.approx(loops[1], abs=1e-12)


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'name': 'SO'}, 'Unknown matrix group'), ({'degree': 0}, 'positive integer'), ({'degree': 2.0}, 'positive')],
)
def test_check_rejects_unknown_groups_and_degrees(arguments, match):
    """Test that an unknown group or a degree that is not a positive integer raises an error."""
    with pytest.raises(ValidationError, match=match):
        MatrixGroupSpec(**arguments)._validated()


def test_components_of_the_wrong_size_and_paths_of_wrong_signs_raise_an_error():
    """Test that components that are not `2 n^2` per matrix, or signs other than 1, -1, and 0, raise an error."""
    group = MatrixGroupSpec('SU', 2)
    with pytest.raises(ValidationError, match='8 real components'):
        group.matrices_of(np.zeros((3, 18)))
    with pytest.raises(ValidationError, match='one sign'):
        group.holonomy_of(np.eye(2)[None], [0], [2])
