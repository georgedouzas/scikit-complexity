"""Test the `FiniteVolumeKernel` class."""

import math
import time
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import divergence
from skcomplex.core.components.spaces import AxisSpec, BaseSpace, LatticeSpace, PhaseSpace
from skcomplex.core.engine.solvers import SSPRKSolver, StepSpec
from skcomplex.core.errors import SolverError
from skcomplex.core.systems import System
from skcomplex.systems.physics.fluids import BurgersSystem, ShallowWaterSystem
from skcomplex.systems.social.cooperation import SimplexSpace

N_STATES = 5
DAM_SPEED_UP = 15.0


class AmountElement(Element):
    """Element that holds one nonnegative scalar `x`, an amount."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x


def _cyclic_laws(s):
    """Return one conservation law per state, the Burgers flux plus the next state, with no closed-form eigenvalues."""
    states = [getattr(s.target, f'u{index}') for index in range(N_STATES)]
    return [
        RateLaw(state, -divergence([state**2 / 2 + states[(index + 1) % N_STATES] / 2]))
        for index, state in enumerate(states)
    ]


def _volume(system):
    """Return the compiled problem and the finite-volume kernel of the only field of a system."""
    problem = system.prepare().system_._problem
    return problem, problem.volumes[0]


def test_max_speed_matches_numerical_eigenvalues():
    """Test that the closed-form speeds of the dam break agree with the numerical eigenvalues."""
    problem, volume = _volume(ShallowWaterSystem(200))
    rng = np.random.default_rng(0)
    depth, discharge = volume.starts
    x = problem.initial_state.copy()
    x[depth : depth + 200] = rng.uniform(0.5, 2.0, 200)
    x[discharge : discharge + 200] = rng.uniform(-1.0, 1.0, 200)
    closed = volume.get_max_speed(x, problem.initial_parameters)
    assert volume.closed_form is True
    numerical = volume.set_params(speeds=()).get_max_speed(x, problem.initial_parameters)
    assert volume.closed_form is False
    assert closed == pytest.approx(numerical, rel=1e-12)


def _timed_dam_break(closed):
    """Return the time and the final state of the dam break on 1600 cells, with closed-form or numerical speeds."""
    problem = ShallowWaterSystem(1600, solver=SSPRKSolver()).prepare().system_._problem
    if not closed:
        problem.volumes[0].set_params(speeds=())
    solver = SSPRKSolver().set_params(until=0.5).prepare(problem)
    step = StepSpec(0.0, solver.initial_state_.copy(), problem.initial_parameters.copy(), 0.5)
    start = time.perf_counter()
    state = solver.run(step, final=True).state_
    return time.perf_counter() - start, state


@pytest.mark.benchmark
def test_run_is_faster_with_closed_form_speeds():
    """Test that closed-form speeds run the dam break faster than numerical eigenvalues with the same solution."""
    _timed_dam_break(True)
    fast, closed = _timed_dam_break(True)
    slow, numerical = _timed_dam_break(False)
    assert slow / fast >= DAM_SPEED_UP
    np.testing.assert_allclose(closed, numerical, rtol=0, atol=1e-10)


def test_run_keeps_burgers_shock_conservative_and_monotone():
    """Test that the Burgers shock keeps its total and forms no new extrema with the closed-form speeds."""
    system = BurgersSystem(400).prepare()
    initial = np.asarray(system.system_.elements[0].u)
    _, volume = _volume(system)
    values = system.simulate(until=2.0, dt=2.0).states[-1]
    assert volume.closed_form is True
    assert values.sum() == pytest.approx(initial.sum(), rel=1e-12)
    assert initial.min() - 1e-12 <= values.min() <= values.max() <= initial.max() + 1e-12


def test_run_falls_back_to_numerical_eigenvalues_without_closed_form():
    """Test that a Jacobian without closed-form eigenvalues uses numerical ones and conserves the totals."""
    n = 40
    centers = (np.arange(n) + 0.5) * 2 * math.pi / n
    values = {f'u{index}': 1.0 + 0.2 * np.sin(centers + index) for index in range(N_STATES)}
    space = LatticeSpace(n, spacing=2 * math.pi / n, discretization='finite-volume')
    field = Element('line', values=values, space=space)
    flux = Interaction('flux', roles={'target': 'line'}, laws=_cyclic_laws)
    system = System(elements=[field], interactions=[flux], solver=SSPRKSolver())
    _, volume = _volume(system)
    trajectory = system.simulate(until=0.5, dt=0.5)
    assert volume.closed_form is False
    for index in range(N_STATES):
        total = values[f'u{index}'].sum()
        assert trajectory.get_values(f'line.u{index}')[-1].sum() == pytest.approx(total, rel=1e-12)


def test_max_speed_rejects_states_outside_the_hyperbolic_region():
    """Test that non-real closed-form speeds raise a solver error naming the field."""
    problem, volume = _volume(ShallowWaterSystem(20))
    x = problem.initial_state.copy()
    x[volume.starts[0] : volume.starts[0] + 20] = -1.0
    with pytest.raises(SolverError, match='channel'):
        volume.get_max_speed(x, problem.initial_parameters)


@pytest.mark.parametrize('space_type', [PhaseSpace, SimplexSpace])
def test_max_speed_divides_by_the_smallest_cell_volume(space_type):
    """Test that the wave speed of a field on a simplex is doubled by its half cells, and kept on a phase space."""
    position = LatticeSpace((1,), discretization='finite-volume')
    space = space_type(position, [AxisSpec('x', 8, 0.0, 1.0), AxisSpec('y', 8, 0.0, 1.0)])
    groups = AmountElement('groups', space=space)
    flow = Interaction(
        'flow',
        roles={'target': 'groups'},
        laws=lambda s: [RateLaw(s.target.x, -divergence([0, 0.5 * s.target.x, -0.25 * s.target.x]))],
    )
    problem, volume = _volume(System(elements=[groups], interactions=[flow]))
    speed = volume.get_max_speed(problem.initial_state, problem.initial_parameters)
    assert speed == pytest.approx(1.0 if space_type is SimplexSpace else 0.5)
