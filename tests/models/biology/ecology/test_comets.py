"""Test the `_comets` module of `skcomplex.models.biology.ecology`."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from skcomplex.engine.compile import compile_system
from skcomplex.models import create, run
from skcomplex.specs import RunStateSpec

THRESHOLD = 1e-3
SPACING = 1.0


def _radius(biomass, shape):
    """Return the radius of the colony from the area of the sites above a level."""
    field = np.ravel(biomass).reshape(shape)
    return np.sqrt(np.count_nonzero(field > 0.01) / np.pi)


def test_colony_expands_at_a_nearly_constant_speed():
    """Test the front speed, the consumed glucose against the uptake, and the zero fluxes of sites without biomass."""
    shape = (25, 25)
    trajectory = run('comets').trajectory_
    times = trajectory.times
    biomass = trajectory.values('biomass.x')
    glucose = trajectory.values('glucose.c')
    exchange = trajectory.values('cells.v_EX_glc__D_e')
    radii = np.array([_radius(item, shape) for item in biomass])
    late = times > times[-1] / 3
    slope, _ = np.polyfit(times[late], radii[late], 1)
    speeds = np.diff(radii[late]) / np.diff(times[late])
    assert slope > 0
    assert abs(np.mean(speeds[len(speeds) // 2 :]) - slope) < 0.3 * slope
    consumed = np.sum(glucose[0]) - np.sum(glucose[-1])
    rate = -np.sum(np.reshape(exchange, (times.size, -1)) * np.reshape(biomass, (times.size, -1)), axis=1)
    assert consumed == pytest.approx(trapezoid(rate, times), rel=0.01)
    empty = np.reshape(biomass[-1], -1) < THRESHOLD
    assert np.all(np.reshape(exchange[-1], -1)[empty] == 0.0)


def test_program_solves_only_the_sites_above_the_threshold():
    """Test that one evaluation solves the one inoculated site."""
    problem = compile_system(create('comets', shape=(5, 5)))
    kernel = problem.programs[0]
    kernel.run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters))
    assert kernel.solves_ == 1
