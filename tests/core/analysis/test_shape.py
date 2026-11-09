"""Test the `_shape` module of `skcomplex.core.analysis`."""

from typing import Annotated

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from skcomplex.core.analysis import chain_shape
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ArrayUpdateLaw
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import RoleSpec, StateSpec


def _freely_rotating_chains(chains, points, bond_angle, rng):
    """Return chains of unit bonds at a fixed bond angle and uniform dihedral angles."""
    positions = np.zeros((chains, points, 3))
    bonds = np.zeros((chains, 3))
    bonds[:, 0] = 1.0
    positions[:, 1] = bonds
    theta = np.pi - bond_angle
    normals = np.tile([0.0, 0.0, 1.0], (chains, 1))
    for index in range(2, points):
        dihedral = rng.uniform(-np.pi, np.pi, chains)
        side = np.cross(normals, bonds)
        turned = np.cos(dihedral)[:, None] * side + np.sin(dihedral)[:, None] * normals
        bonds = np.cos(theta) * bonds + np.sin(theta) * turned
        normals = np.cross(bonds, side)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        positions[:, index] = positions[:, index - 1] + bonds
    return positions


def test_shape_end_to_end_of_a_freely_rotating_chain():
    """Test that the mean squared end-to-end distance of a freely rotating chain matches its formula within 5%."""
    n, bond_angle = 200, np.radians(112.0)
    positions = _freely_rotating_chains(2000, n + 1, bond_angle, np.random.default_rng(0))
    bonds = np.linalg.norm(np.diff(positions, axis=1), axis=2)
    assert bonds == pytest.approx(1.0)
    values = chain_shape(positions, kind='end-to-end').values
    cosine = np.cos(np.pi - bond_angle)
    assert np.mean(values**2) == pytest.approx(n * (1 + cosine) / (1 - cosine), rel=0.05)


def test_shape_rmsd_vanishes_after_superposition():
    """Test that a conformation under a random rigid motion deviates from the original below 1e-6."""
    rng = np.random.default_rng(1)
    original = rng.normal(size=(30, 3))
    moved = Rotation.random(random_state=rng).apply(original) + rng.normal(size=3)
    values = chain_shape(moved[None], kind='rmsd', reference=original).values
    assert values[0] < 1e-6


def test_shape_gyration_of_two_points():
    """Test that two points at distance 2 have a radius of gyration of 1."""
    chain = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    assert chain_shape(chain).values[0] == pytest.approx(1.0)


def test_shape_rejects_a_deviation_without_a_reference():
    """Test that the kind `rmsd` needs a reference."""
    with pytest.raises(AnalysisError, match='reference'):
        chain_shape(np.zeros((1, 3, 2)), kind='rmsd')


def test_shape_rejects_an_unknown_kind():
    """Test that an unknown kind raises an error that names it."""
    with pytest.raises(AnalysisError, match='waist'):
        chain_shape(np.zeros((1, 3, 2)), kind='waist')


def test_shape_rejects_positions_of_another_rank():
    """Test that positions without frames raise an error that names their shape."""
    with pytest.raises(AnalysisError, match=r'\(2, 3\)'):
        chain_shape(np.zeros((2, 3)))


class CoordinateElement(Element):
    """Coordinate of a point of a chain."""

    def __init__(self, label, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


class StretchInteraction(Interaction):
    """Drive that scales a chain by the time of the step."""

    def __init__(self, label, *, target: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.target = target

    def _get_laws(self, namespace):
        """Return the drive."""

        def stretch(values, *, time):
            """Return points on a line, spaced by the time."""
            return np.outer(np.arange(values.shape[0]), [time, 0.0, 0.0])

        return [ArrayUpdateLaw(self._get_member(namespace, 'target').x, stretch)]


def test_shape_reads_a_lattice_state_of_a_trajectory():
    """Test that a lattice state whose last axis holds coordinates is read as frames by points by dimensions."""
    chain = GroupElement('chain', element=CoordinateElement, space=LatticeSpace((4, 3), boundary='fixed'), on='sites')
    system = System(elements=[chain], interactions=[StretchInteraction('s', target='chain')], time=DiscreteTime())
    trajectory = simulate(system, until=2, random_state=0)
    values = chain_shape(trajectory, kind='end-to-end', group='chain.x').values
    assert values.tolist() == [0.0, 3.0, 6.0]


def test_shape_reports_a_group_it_cannot_find():
    """Test that a trajectory without the group raises an error."""
    chain = GroupElement('chain', element=CoordinateElement, space=LatticeSpace((4, 3), boundary='fixed'), on='sites')
    trajectory = simulate(System(elements=[chain], time=DiscreteTime()), until=1, random_state=0)
    with pytest.raises(AnalysisError):
        chain_shape(trajectory, group='nothing')
