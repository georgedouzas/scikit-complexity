"""Test the `_koopman` module of `skcomplex.models.engineering.control`."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures

from skcomplex.core.spaces import ContinuousTime, DiscreteTime
from skcomplex.engine.compile import compile_system
from skcomplex.engine.solvers import CoSimulationSolver
from skcomplex.errors import ValidationError
from skcomplex.models import describe, run
from skcomplex.models.engineering.control import koopman_model_predictive_control
from skcomplex.types import FormKind


def _tracking_error(trajectory):
    """Return the mean squared error of the output from the reference."""
    return float(np.mean((trajectory.values('plant/state.y') - trajectory.values('reference')) ** 2))


@pytest.mark.parametrize(('plant', 'until'), [('duffing', 100.0), ('van_der_pol', 100.0), ('motor', 300.0)])
def test_koopman_model_predictive_control_tracks_reference(plant, until):
    """Test that the controlled output tracks the reference with a smaller error than the uncontrolled plant."""
    controlled = run(
        'koopman_model_predictive_control',
        plant=plant,
        n_runs=20,
        duration=200,
        settings={'until': until},
    ).trajectory_
    free = run('koopman_model_predictive_control', plant=plant, controlled=False, settings={'until': until})
    assert _tracking_error(controlled) < _tracking_error(free.trajectory_)


def test_koopman_model_predictive_control_duffing_reads_the_input_through_its_gain():
    """Test that the input of the forced Duffing oscillator enters through `(3 + cos(x)) u`, as the review writes it."""
    plant_rate = koopman_model_predictive_control(controlled=False).interactions[2].rate
    x, y, control = 0.7, -0.2, 0.5
    s = SimpleNamespace(target=SimpleNamespace(x=x, y=y), sources=[SimpleNamespace(u=control)], t=0.0)
    expected = x - x**3 - 0.3 * y + 0.5 + (3 + np.cos(x)) * 0.5 * control
    assert float(plant_rate(s)) == pytest.approx(0.1 * expected)


def test_koopman_model_predictive_control_keeps_input_in_bounds():
    """Test that the controller sets the input inside its bounds and that it changes over the run."""
    found = run('koopman_model_predictive_control', n_runs=20, duration=200, settings={'until': 60.0})
    inputs = found.trajectory_.values('controller/input.u')
    assert np.all(np.abs(inputs) <= 1.0)
    assert np.ptp(inputs) > 0.5


def test_koopman_model_predictive_control_holds_input_at_zero_without_control():
    """Test that the uncontrolled system holds the input at 0 and needs no fit."""
    system = koopman_model_predictive_control('motor', controlled=False)
    trajectory = run('koopman_model_predictive_control', plant='motor', controlled=False, settings={'until': 5.0})
    assert system.label == 'motor'
    assert np.all(trajectory.trajectory_.values('controller/input.u') == 0.0)


def test_koopman_model_predictive_control_is_a_co_simulation():
    """Test that the controller runs under discrete time and the plant under continuous time, in two partitions."""
    system = koopman_model_predictive_control(controlled=False)
    controller, plant = system.systems
    assert isinstance(controller.time, DiscreteTime)
    assert isinstance(plant.time, ContinuousTime)
    problem = compile_system(system)
    assert problem.form == FormKind.COUPLED
    assert [(item.name, item.problem.form) for item in problem.partitions] == [
        ('controller', FormKind.DISCRETE),
        ('plant', FormKind.ODE),
    ]


def test_koopman_model_predictive_control_reads_reference_of_time():
    """Test that the reference observable follows the review's reference of the plant, in samples of the controller."""
    trajectory = run('koopman_model_predictive_control', plant='motor', controlled=False, settings={'until': 30.0})
    times = trajectory.trajectory_.times
    expected = 0.5 * np.cos(2 * np.pi * 0.01 * times / 3)
    assert np.allclose(trajectory.trajectory_.values('reference'), expected)


def test_koopman_model_predictive_control_takes_a_dictionary():
    """Test that a dictionary that holds the variables replaces the monomials of degree 3."""
    system = koopman_model_predictive_control(dictionary=PolynomialFeatures(2, include_bias=False), n_runs=4)
    assert compile_system(system).form == FormKind.COUPLED


def test_koopman_model_predictive_control_rejects_unknown_plant():
    """Test that an unknown plant raises an error naming it."""
    with pytest.raises(ValidationError, match='boiler'):
        koopman_model_predictive_control('boiler')


def test_koopman_model_predictive_control_default_solver_is_co_simulation():
    """Test that a default run schedules the plant and the controller with the co-simulation solver."""
    simulation = run('koopman_model_predictive_control', controlled=False, settings={'until': 2.0})
    assert isinstance(simulation.stepper_, CoSimulationSolver)
    assert describe('koopman_model_predictive_control').field == 'engineering/control'
