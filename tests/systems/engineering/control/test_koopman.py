"""Test the `_koopman` module of `skcomplex.systems.engineering.control`."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures

from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import ValidationError
from skcomplex.systems.engineering.control import KoopmanModelPredictiveControlSystem


def _tracking_error(trajectory):
    """Return the mean squared error of the output from the reference."""
    return float(np.mean((trajectory.get_values('plant/state.y') - trajectory.get_values('reference')) ** 2))


@pytest.mark.parametrize(('plant', 'until'), [('duffing', 100.0), ('van_der_pol', 100.0), ('motor', 300.0)])
def test_koopman_model_predictive_control_tracks_reference(plant, until):
    """Test that the controlled output tracks the reference with a smaller error than the uncontrolled plant."""
    controlled = simulate(KoopmanModelPredictiveControlSystem(plant=plant, n_runs=20, duration=200), until=until)
    free = simulate(KoopmanModelPredictiveControlSystem(plant=plant, controlled=False), until=until)
    assert _tracking_error(controlled) < _tracking_error(free)


def test_koopman_model_predictive_control_duffing_reads_the_input_through_its_gain():
    """Test that the input of the forced Duffing oscillator enters through `(3 + cos(x)) u`, as the review writes it."""
    plant_rate = KoopmanModelPredictiveControlSystem(controlled=False).prepare().system_.interactions[2].rate
    x, y, control = 0.7, -0.2, 0.5
    s = SimpleNamespace(target=SimpleNamespace(x=x, y=y), sources=[SimpleNamespace(u=control)], t=0.0)
    expected = x - x**3 - 0.3 * y + 0.5 + (3 + np.cos(x)) * 0.5 * control
    assert float(plant_rate(s)) == pytest.approx(0.1 * expected)


def test_koopman_model_predictive_control_keeps_input_in_bounds():
    """Test that the controller sets the input inside its bounds and that it changes over the run."""
    found = simulate(KoopmanModelPredictiveControlSystem(n_runs=20, duration=200), until=60.0)
    inputs = found.get_values('controller/input.u')
    assert np.all(np.abs(inputs) <= 1.0)
    assert np.ptp(inputs) > 0.5


def test_koopman_model_predictive_control_holds_input_at_zero_without_control():
    """Test that the uncontrolled system holds the input at 0 and needs no fit."""
    trajectory = simulate(KoopmanModelPredictiveControlSystem(plant='motor', controlled=False), until=5.0)
    assert np.all(trajectory.get_values('controller/input.u') == 0.0)


def test_koopman_model_predictive_control_is_a_co_simulation():
    """Test that the controller runs under discrete time and the plant under continuous time, in two partitions."""
    system = KoopmanModelPredictiveControlSystem(controlled=False).prepare().system_
    controller, plant = system.systems
    assert isinstance(controller.time, DiscreteTime)
    assert isinstance(plant.time, ContinuousTime)
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.form == FormKind.COUPLED
    assert [(item.name, item.problem.form) for item in problem.partitions] == [
        ('controller', FormKind.DISCRETE),
        ('plant', FormKind.ODE),
    ]


def test_koopman_model_predictive_control_reads_reference_of_time():
    """Test that the reference observable follows the review's reference of the plant, in samples of the controller."""
    trajectory = simulate(KoopmanModelPredictiveControlSystem(plant='motor', controlled=False), until=30.0)
    times = trajectory.times
    expected = 0.5 * np.cos(2 * np.pi * 0.01 * times / 3)
    assert np.allclose(trajectory.get_values('reference'), expected)


def test_koopman_model_predictive_control_takes_a_dictionary():
    """Test that a dictionary that holds the variables replaces the monomials of degree 3."""
    system = (
        KoopmanModelPredictiveControlSystem(dictionary=PolynomialFeatures(2, include_bias=False), n_runs=4)
        .prepare()
        .system_
    )
    assert PassCompiler().prepare(system).run().problem_.form == FormKind.COUPLED


def test_koopman_model_predictive_control_rejects_unknown_plant():
    """Test that an unknown plant raises an error naming it."""
    with pytest.raises(ValidationError, match='boiler'):
        KoopmanModelPredictiveControlSystem('boiler').prepare()


def test_koopman_model_predictive_control_default_solver_is_co_simulation():
    """Test that a default run schedules the plant and the controller with the co-simulation solver."""
    system = KoopmanModelPredictiveControlSystem(controlled=False).prepare().system_
    trajectory = simulate(system, until=2.0, dt=1.0)
    assert trajectory.problem.form == FormKind.COUPLED
    assert KoopmanModelPredictiveControlSystem.field == 'engineering/control'
