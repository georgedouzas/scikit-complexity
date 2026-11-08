"""Test the `_mobility` module of `skcomplex.core.components.interactions`."""

import math
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest
from sage.all__sagemath_symbolics import matrix, vector

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MobilityInteraction
from skcomplex.core.components.interactions._mobility import _mobility_matrix, _mobility_product, _source_flows
from skcomplex.core.components.spaces import ContinuousSpace, DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec


class SphereElement(Element):
    """Sphere with an order-1 position, an orientation, and a force and a torque of its own."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        p: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
        f: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        c: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.p = p
        self.f = f
        self.c = c


def _system(interaction, positions, *, orientations=None, forces=None, centers=None, time=None):
    """Return a system of spheres with given positions and per-sphere forces or trap centres."""
    positions = np.asarray(positions, dtype=float)
    size, dimension = positions.shape
    values = {
        'x': positions,
        'p': np.zeros_like(positions) if orientations is None else orientations,
        'f': np.zeros_like(positions) if forces is None else forces,
        'c': positions if centers is None else centers,
    }
    spheres = GroupElement('spheres', element=SphereElement, size=size, values=values)
    extra = {} if time is None else {'time': time}
    return System(elements=[spheres], interactions=[interaction], space=ContinuousSpace(n_dim=dimension), **extra)


def _velocities(interaction, positions, **kwargs):
    """Return the velocities of the spheres over a short run of the rate form."""
    system = _system(interaction, positions, **kwargs)
    trajectory = simulate(system, until=1e-6, dt=1e-6)
    return (trajectory.values('spheres.x')[-1] - trajectory.values('spheres.x')[0]) / 1e-6


def _external(s):
    """Return the force of a sphere, its parameter `f`."""
    return s.target.f


def test_stokes_drag_of_one_sphere():
    """Test that a lone sphere moves at `F / (6 pi eta a)`."""
    stokes = MobilityInteraction('stokes', target='spheres', force=_external, radius=0.5, viscosity=2.0)
    velocity = _velocities(stokes, [[0.0, 0.0, 0.0]], forces=[[3.0, 0.0, -1.0]])
    assert velocity[0] == pytest.approx(np.array([3.0, 0.0, -1.0]) / (6 * math.pi * 2.0 * 0.5), rel=1e-6)


@pytest.mark.parametrize(
    ('tensor', 'along', 'across'),
    [
        ('oseen', 1 / (4 * math.pi * 3.0), 1 / (8 * math.pi * 3.0)),
        (
            'rotne-prager-yamakawa',
            1 / (4 * math.pi * 3.0) - 1 / (6 * math.pi * 27.0),
            (1 + 2 / 27) / (8 * math.pi * 3.0),
        ),
        ('free-draining', 0.0, 0.0),
    ],
)
def test_pair_coupling_of_the_tensors(tensor, along, across):
    """Test that a force on one sphere moves the other as the Oseen and Rotne-Prager-Yamakawa tensors give."""
    stokes = MobilityInteraction('stokes', target='spheres', force=_external, tensor=tensor)
    velocity = _velocities(stokes, [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], forces=[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert velocity[1] == pytest.approx([along, across, 0.0], rel=1e-5, abs=1e-12)


def test_planar_positions_read_the_plane_of_space():
    """Test that positions in the plane move as the plane `z = 0` of the three-dimensional tensor."""
    stokes = MobilityInteraction('stokes', target='spheres', force=_external)
    planar = _velocities(stokes, [[0.0, 0.0], [3.0, 0.0]], forces=[[1.0, 1.0], [0.0, 0.0]])
    spatial = _velocities(stokes, [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], forces=[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert planar == pytest.approx(spatial[:, :2], rel=1e-9)


def test_matrix_is_symmetric_positive_and_matches_the_product():
    """Test that the grand mobility is symmetric, positive definite for RPY overlaps, and equal to the product."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 3, size=(6, 3))
    matrix_ = _mobility_matrix(x, (1.0, 1.0), 2)
    assert matrix_ == pytest.approx(matrix_.T)
    assert np.linalg.eigvalsh(matrix_).min() > 0
    load = rng.standard_normal(36)
    velocity, angular = _mobility_product(
        x,
        load[:18].reshape(6, 3).copy(),
        load[18:].reshape(6, 3).copy(),
        (1.0, 1.0),
        2,
    )
    assert np.concatenate([velocity.ravel(), angular.ravel()]) == pytest.approx(matrix_ @ load)


def test_rotlet_couplings():
    """Test that a torque moves another sphere as the rotlet and a force spins it by half the vorticity."""
    x = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    torque = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    velocity, angular = _mobility_product(x, np.zeros((2, 3)), torque, (1.0, 1.0), 1)
    assert velocity[1] == pytest.approx(np.cross(torque[0], x[1]) / (8 * math.pi * 64))
    assert angular[1] == pytest.approx(-torque[0] / (16 * math.pi * 64))
    force = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    velocity, angular = _mobility_product(x, force, np.zeros((2, 3)), (1.0, 1.0), 1)
    assert angular[1] == pytest.approx(np.cross(force[0], x[1]) / (8 * math.pi * 64))


def test_stresslet_flow_of_a_puller_draws_in_along_its_axis():
    """Test that the stresslet `sigma (e e - I / 3)` draws a sphere on its axis in at `sigma / (4 pi eta r^2)`."""
    x = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    stresslet = np.zeros((3, 3, 3))
    stresslet[0] = 2.0 * (np.outer([1, 0, 0], [1, 0, 0]) - np.eye(3) / 3)
    velocity, angular = _source_flows(x, stresslet, np.zeros((3, 3)), 1.0)
    assert velocity[1] == pytest.approx([-2.0 / (4 * math.pi * 9), 0.0, 0.0])
    assert velocity[2] == pytest.approx([0.0, 2.0 / (8 * math.pi * 9), 0.0])
    assert angular[1] == pytest.approx(np.zeros(3))
    dipole = np.zeros((3, 3))
    dipole[0] = [1.0, 0.0, 0.0]
    velocity, _ = _source_flows(x, np.zeros((3, 3, 3)), dipole, 1.0)
    assert velocity[1] == pytest.approx([2.0 / 27, 0.0, 0.0])
    assert velocity[2] == pytest.approx([-1.0 / 27, 0.0, 0.0])


def test_orientation_turns_with_the_angular_velocity_and_stays_unit():
    """Test that a constant torque turns the orientation about it at `T / (8 pi eta a^3)`."""
    spin = MobilityInteraction('spin', target='spheres', orientation='p', torque=lambda s: vector([0.0, 0.0, 1.0]))
    system = _system(spin, [[0.0, 0.0, 0.0]], orientations=[[1.0, 0.0, 0.0]])
    turn = simulate(system, until=16 * math.pi**2, dt=0.1).values('spheres.p')
    assert turn[-1, 0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-4)
    assert np.linalg.norm(turn[:, 0], axis=1) == pytest.approx(1.0, abs=1e-4)


def test_intrinsic_velocity_and_planar_torque():
    """Test that an intrinsic velocity adds to the mobility and a planar torque is a number about `z`."""
    swim = MobilityInteraction(
        'swim',
        target='spheres',
        orientation='p',
        velocity=lambda s: 2 * s.target.p,
        torque=lambda s: 8 * math.pi,
    )
    system = _system(swim, [[0.0, 0.0]], orientations=[[1.0, 0.0]])
    trajectory = simulate(system, until=math.pi / 2, dt=0.01)
    assert trajectory.values('spheres.p')[-1, 0] == pytest.approx([0.0, 1.0], abs=1e-4)
    assert trajectory.values('spheres.x')[-1, 0] == pytest.approx([2.0, 2.0], abs=1e-3)


def test_pair_potential_forces_enter_the_mobility():
    """Test that a pair potential of the distance pushes overlapping spheres apart within its cutoff only."""
    repulsion = MobilityInteraction(
        'repulsion',
        target='spheres',
        potential=lambda s, r: s.radius * (3 - r) ** 2,
        cutoff=3.0,
        tensor='free-draining',
    )
    velocity = _velocities(repulsion, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert velocity[1, 0] == pytest.approx(2.0 / (6 * math.pi), rel=1e-4)
    assert velocity[0, 0] == pytest.approx(-2.0 / (6 * math.pi), rel=1e-4)
    assert velocity[2] == pytest.approx(np.zeros(3))


def _trap(s):
    """Return the harmonic force of stiffness 10 towards the centre of a sphere."""
    return -10 * (s.target.x - s.target.c)


@pytest.mark.parametrize('root', ['cholesky', 'lanczos'])
def test_brownian_steps_have_the_mobility_covariance(root):
    """Test that free Brownian steps correlate as `M_12 / M_11` and trapped spheres reach the variance `kT / k`."""
    stokes = MobilityInteraction(
        'stokes',
        target='spheres',
        force=_trap,
        form='map',
        temperature=0.01,
        dt=0.05,
        root=root,
    )
    system = _system(stokes, [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], time=DiscreteTime())
    positions = simulate(system, until=4000, random_state=0).values('spheres.x')
    steps = np.diff(positions, axis=0)
    along = 6 * math.pi * (1 / (4 * math.pi * 2.5) - 1 / (6 * math.pi * 2.5**3))
    across = 6 * math.pi * (1 + 2 / (3 * 2.5**2)) / (8 * math.pi * 2.5)
    assert np.corrcoef(steps[:, 0, 0], steps[:, 1, 0])[0, 1] == pytest.approx(along, abs=0.05)
    assert np.corrcoef(steps[:, 0, 1], steps[:, 1, 1])[0, 1] == pytest.approx(across, abs=0.05)
    assert positions[:, 0].var(axis=0) == pytest.approx(np.full(3, 0.001), rel=0.3)


def test_brownian_rotation_turns_orientations():
    """Test that rotational Brownian steps decorrelate an orientation as `exp(-2 D_r t)` and keep it a unit vector."""
    stokes = MobilityInteraction('stokes', target='spheres', orientation='p', form='map', temperature=1.0, dt=0.01)
    size = 50
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 1000, size=(size, 3))
    orientations = np.tile([0.0, 0.0, 1.0], (size, 1))
    system = _system(stokes, positions, orientations=orientations, time=DiscreteTime())
    turned = simulate(system, until=100, random_state=0).values('spheres.p')
    rotation = 1 / (8 * math.pi)
    assert turned[-1, :, 2].mean() == pytest.approx(math.exp(-2 * rotation), abs=0.08)
    assert np.linalg.norm(turned[-1], axis=1) == pytest.approx(1.0)


def test_oseen_overlap_is_not_positive():
    """Test that noise from an Oseen mobility of overlapping spheres raises an error that suggests RPY."""
    stokes = MobilityInteraction('stokes', target='spheres', tensor='oseen', form='map', temperature=1.0)
    system = _system(stokes, [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], time=DiscreteTime())
    with pytest.raises(ValidationError, match='rotne-prager-yamakawa'):
        simulate(system, until=2, random_state=0)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'tensor': 'stokeslet'}, 'Unknown tensor'),
        ({'form': 'flow'}, 'Unknown form'),
        ({'root': 'eigen'}, 'Unknown root'),
        ({'force': 1.0}, 'Give a function'),
        ({'temperature': 1.0}, "form='map'"),
    ],
)
def test_variants_and_functions_are_checked(arguments, message):
    """Test that unknown variants, functions that are not callable, and noise in the rate form are reported."""
    with pytest.raises(ValidationError, match=message):
        MobilityInteraction('stokes', target='spheres', **arguments).prepare()


def test_stresslet_of_wrong_shape_is_reported():
    """Test that a stresslet whose shape does not match the positions is reported when the laws are built."""
    stokes = MobilityInteraction('stokes', target='spheres', stresslet=lambda s: matrix([[1.0, 0.0], [0.0, -1.0]]))
    with pytest.raises(ValidationError, match='3 by 3'):
        simulate(_system(stokes, [[0.0, 0.0, 0.0]]), until=0, dt=0.1)
