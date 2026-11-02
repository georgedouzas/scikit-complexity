"""Test the `FiniteVolumeKernel` class."""

import math
import time
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.laws import FluxLaw
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.compile import PassCompiler
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import SSPRKSolver
from skcomplex.errors import SolverError
from skcomplex.models.physics.fluids import burgers, shallow_water
from skcomplex.specs import StateSpec, VisualSpec

N_STATES = 5
DAM_SPEED_UP = 15.0


class CyclicElement(Element):
    """Cell of five states whose fluxes couple each state to the next, with no closed-form Jacobian eigenvalues."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u0: Annotated[object, StateSpec()] = 0.0,
        u1: Annotated[object, StateSpec()] = 0.0,
        u2: Annotated[object, StateSpec()] = 0.0,
        u3: Annotated[object, StateSpec()] = 0.0,
        u4: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3
        self.u4 = u4

    def _get_laws(self, s):
        """Return one flux law per state, the Burgers flux plus the next state."""
        states = [s.u0, s.u1, s.u2, s.u3, s.u4]
        return [FluxLaw(state, state**2 / 2 + states[(index + 1) % N_STATES] / 2) for index, state in enumerate(states)]


def _volume(system):
    """Return the compiled problem and the finite-volume kernel of its only field."""
    problem = PassCompiler().check().prepare(system).run().problem_
    return problem, problem.volumes[0]


def test_max_speed_matches_numerical_eigenvalues():
    """Test that the closed-form speeds of the dam break agree with the numerical eigenvalues."""
    problem, volume = _volume(shallow_water(200))
    rng = np.random.default_rng(0)
    depth, discharge = volume.starts
    x = problem.initial_state.copy()
    x[depth : depth + 200] = rng.uniform(0.5, 2.0, 200)
    x[discharge : discharge + 200] = rng.uniform(-1.0, 1.0, 200)
    closed = volume.max_speed(x, problem.initial_parameters)
    assert volume.closed_form is True
    numerical = volume.set_params(speeds=()).max_speed(x, problem.initial_parameters)
    assert volume.closed_form is False
    assert closed == pytest.approx(numerical, rel=1e-12)


def _timed_dam_break(closed):
    """Return the time and the final state of the dam break on 1600 cells, with closed-form or numerical speeds."""
    simulation = Simulation(shallow_water(1600), dt=0.5, solver=SSPRKSolver()).check().prepare()
    if not closed:
        simulation.problem_.volumes[0].set_params(speeds=())
    start = time.perf_counter()
    simulation.run(until=0.5)
    return time.perf_counter() - start, simulation.state_


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
    system = burgers(400)
    initial = np.asarray(system.elements[0].values['u'])
    _, volume = _volume(system)
    values = Simulation(system, dt=2.0, solver=SSPRKSolver()).check().prepare().run(until=2.0).state_
    assert volume.closed_form is True
    assert values.sum() == pytest.approx(initial.sum(), rel=1e-12)
    assert initial.min() - 1e-12 <= values.min() <= values.max() <= initial.max() + 1e-12


def test_run_falls_back_to_numerical_eigenvalues_without_closed_form():
    """Test that a Jacobian without closed-form eigenvalues uses numerical ones and conserves the totals."""
    n = 40
    centers = (np.arange(n) + 0.5) * 2 * math.pi / n
    values = {f'u{index}': 1.0 + 0.2 * np.sin(centers + index) for index in range(N_STATES)}
    field = FieldElement(
        'line',
        element=CyclicElement,
        space=LatticeSpace(n, spacing=2 * math.pi / n),
        values=values,
        discretization='finite-volume',
    )
    system = System(elements=[field])
    _, volume = _volume(system)
    trajectory = Simulation(system, dt=0.5, solver=SSPRKSolver()).check().prepare().run(until=0.5).trajectory_
    assert volume.closed_form is False
    for index in range(N_STATES):
        total = values[f'u{index}'].sum()
        assert trajectory.values(f'line.u{index}')[-1].sum() == pytest.approx(total, rel=1e-12)


def test_max_speed_rejects_states_outside_the_hyperbolic_region():
    """Test that non-real closed-form speeds raise a solver error naming the field."""
    problem, volume = _volume(shallow_water(20))
    x = problem.initial_state.copy()
    x[volume.starts[0] : volume.starts[0] + 20] = -1.0
    with pytest.raises(SolverError, match='channel'):
        volume.max_speed(x, problem.initial_parameters)
