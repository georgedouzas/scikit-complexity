"""Test the `_alignment` module of `skcomplex.core.interactions`."""

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest
from sage.all__sagemath_symbolics import SR, I, cos, exp, pi, sin, vector

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import AlignmentInteraction
from skcomplex.core.laws import BaseLaw, UpdateLaw
from skcomplex.core.spaces import ContinuousSpace, DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec


class MoverElement(Element):
    """Member that moves at a constant speed along its heading."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(vector=True, position=True)] = 0.0,
        theta: Annotated[object, StateSpec(period=2 * pi)] = 0.0,
        v0: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.v0 = v0

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the move along the heading."""
        return [UpdateLaw(s.x, s.x + s.v0 * vector(SR, [cos(s.theta), sin(s.theta)]))]


class VelocityElement(Element):
    """Member with a position and a velocity."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(vector=True, position=True)] = 0.0,
        v: Annotated[object, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v


def _heading(member: SimpleNamespace) -> object:
    """Return the unit heading of a member in the complex plane."""
    return exp(I * member.theta)


def _polarization(s: SimpleNamespace) -> object:
    """Return the modulus of the mean heading, the polar order parameter."""
    return abs(s.mean_heading)


def _run(
    element: type[Element],
    values: dict[str, object],
    until: float,
    alignment: AlignmentInteraction,
    observables: Sequence[ObservableSpec] = (),
) -> Trajectory:
    """Return the trajectory of a group of members that align in a periodic square of side 10."""
    group = GroupElement('flock', element=element, size=len(np.asarray(values['x'])), values=values)
    system = System(
        elements=[group],
        interactions=[alignment],
        space=ContinuousSpace(extent=(10.0, 10.0), boundary='periodic'),
        time=DiscreteTime(),
    )
    simulation = Simulation(system, dt=1.0, random_state=0, observables=observables)
    return simulation.check().prepare().run(until=until).trajectory_


def _headings(trajectory: Trajectory) -> npt.NDArray[np.float64]:
    """Return the headings of the members at the last frame, wrapped to `(-pi, pi]`."""
    headings = np.asarray(trajectory.values('flock.theta')[-1], dtype=float)
    return np.angle(np.exp(1j * headings))


def test_align_orders_a_vicsek_flock_at_low_noise():
    """Test that self-propelled members aligning within a unit radius at low noise reach a polar order above 0.9."""
    rng = np.random.default_rng(0)
    values = {'x': rng.uniform(0.0, 10.0, (300, 2)), 'theta': rng.uniform(-math.pi, math.pi, 300), 'v0': 0.5}
    alignment = AlignmentInteraction('alignment', members='flock', noise=0.1)
    observables = [
        ObservableSpec('mean_heading', _heading, group='flock'),
        ObservableSpec('polarization', _polarization),
    ]
    trajectory = _run(MoverElement, values, 150.0, alignment, observables)
    polarization = np.asarray(trajectory.values('polarization'), dtype=float)
    assert polarization[0] < 0.2
    assert polarization[-1] > 0.9


def test_align_takes_the_mean_heading_within_the_radius():
    """Test that polar alignment without noise sets every heading to the direction of the sum within the radius."""
    values = {'x': [[5.0, 5.0], [5.5, 5.0], [8.0, 5.0]], 'theta': [0.0, math.pi / 2, 1.0]}
    alignment = AlignmentInteraction('alignment', members='flock')
    headings = _headings(_run(MoverElement, values, 1.0, alignment))
    np.testing.assert_allclose(headings, [math.pi / 4, math.pi / 4, 1.0])


def test_align_nematic_flips_opposite_headings():
    """Test that nematic alignment aligns nearly opposite headings along their common axis."""
    values = {'x': [[5.0, 5.0], [5.5, 5.0]], 'theta': [0.0, math.pi - 0.2]}
    alignment = AlignmentInteraction('alignment', members='flock', symmetry='nematic')
    headings = _headings(_run(MoverElement, values, 1.0, alignment))
    np.testing.assert_allclose(headings, [-0.1, math.pi - 0.1])


def test_align_repulsion_turns_close_members_apart():
    """Test that the repulsion zone turns two close members with equal headings away from each other."""
    values = {'x': [[5.0, 5.0], [5.2, 5.0]], 'theta': [math.pi / 2, math.pi / 2]}
    alignment = AlignmentInteraction('alignment', members='flock', repulsion=0.5)
    headings = _headings(_run(MoverElement, values, 1.0, alignment))
    np.testing.assert_allclose(headings, [math.atan2(2, -1), math.atan2(2, 1)])


def test_align_attraction_turns_distant_members_together():
    """Test that the attraction zone turns members beyond the radius towards each other."""
    values = {'x': [[4.0, 5.0], [6.0, 5.0]], 'theta': [math.pi / 2, math.pi / 2]}
    alignment = AlignmentInteraction('alignment', members='flock', attraction=3.0)
    headings = _headings(_run(MoverElement, values, 1.0, alignment))
    np.testing.assert_allclose(headings, [math.pi / 4, 3 * math.pi / 4])


def test_align_turns_velocities_and_keeps_their_speeds():
    """Test that velocity alignment turns every velocity to the mean direction and keeps its speed."""
    values = {'x': [[5.0, 5.0], [5.5, 5.0]], 'v': [[2.0, 0.0], [0.0, 1.0]]}
    alignment = AlignmentInteraction('alignment', members='flock', heading=None, velocity='v')
    trajectory = _run(VelocityElement, values, 1.0, alignment)
    velocities = np.asarray(trajectory.values('flock.v')[-1], dtype=float)
    np.testing.assert_allclose(velocities, [[math.sqrt(2), math.sqrt(2)], [math.sqrt(0.5), math.sqrt(0.5)]])


@pytest.mark.parametrize(
    'arguments',
    [
        {'symmetry': 'axial'},
        {'heading': None},
        {'velocity': 'v'},
        {'radius': 0.0},
        {'repulsion': -1.0},
        {'attraction': 0.5},
    ],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown symmetry, not one direction state, and invalid distances are rejected."""
    with pytest.raises(ValidationError):
        AlignmentInteraction('alignment', members='flock', **arguments).check()
