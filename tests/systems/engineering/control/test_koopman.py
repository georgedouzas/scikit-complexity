"""Test the `_koopman` module of `skcomplex.systems.engineering.control`."""

from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR
from sklearn.preprocessing import PolynomialFeatures

from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import ValidationError
from skcomplex.systems.engineering.control import KoopmanModelPredictiveControlSystem, PlantInteraction


def _simulate(system, until):
    """Return the trajectory of a prepared system until a time."""
    return system.prepare().simulate(until=until)


def _tracking_error(trajectory):
    """Return the mean squared error of the output from the reference."""
    return float(np.mean((trajectory.get_values('plant/state.y') - trajectory.get_values('reference')) ** 2))


@pytest.mark.parametrize(('plant', 'until'), [('duffing', 100.0), ('van_der_pol', 100.0), ('motor', 300.0)])
def test_koopman_model_predictive_control_tracks_reference(plant, until):
    """Test that the controlled output tracks the reference with a smaller error than the uncontrolled plant."""
    controlled = _simulate(KoopmanModelPredictiveControlSystem(plant=plant, n_runs=20, duration=200), until)
    free = _simulate(KoopmanModelPredictiveControlSystem(plant=plant, controlled=False), until)
    assert _tracking_error(controlled) < _tracking_error(free)


def test_koopman_model_predictive_control_duffing_reads_the_input_through_its_gain():
    """Test that the input of the forced Duffing oscillator enters through `(3 + cos(x)) u`, as the review writes it."""
    x, y, control = 0.7, -0.2, 0.5
    namespace = SimpleNamespace(
        plant=SimpleNamespace(x=SR(x), y=SR(y)),
        control=SimpleNamespace(u=SR(control)),
        t=SR.var('t'),
    )
    rates = PlantInteraction('rates', plant='plant', control='input', kind='duffing').prepare(namespace)
    expected = x - x**3 - 0.3 * y + 0.5 + (3 + np.cos(x)) * 0.5 * control
    assert float(rates.laws_[1].value.subs(t=0)) == pytest.approx(0.1 * expected)


def test_koopman_model_predictive_control_keeps_input_in_bounds():
    """Test that the controller sets the input inside its bounds and that it changes over the run."""
    found = _simulate(KoopmanModelPredictiveControlSystem(n_runs=20, duration=200), 60.0)
    inputs = found.get_values('controller/input.u')
    assert np.all(np.abs(inputs) <= 1.0)
    assert np.ptp(inputs) > 0.5


def test_koopman_model_predictive_control_holds_input_at_zero_without_control():
    """Test that the uncontrolled system holds the input at 0 and needs no fit."""
    trajectory = _simulate(KoopmanModelPredictiveControlSystem(plant='motor', controlled=False), 5.0)
    assert np.all(trajectory.get_values('controller/input.u') == 0.0)


def test_koopman_model_predictive_control_is_a_co_simulation():
    """Test that the controller steps once per sample and the plant runs in continuous time, in two partitions."""
    system = KoopmanModelPredictiveControlSystem(controlled=False).prepare()
    assert system.form_ == 'co-simulation'
    problem = PassCompiler().prepare(system.system_).run().problem_
    assert [(item.name, item.problem.form) for item in problem.partitions] == [
        ('controller', FormKind.DISCRETE),
        ('plant', FormKind.ODE),
    ]


def test_koopman_model_predictive_control_reads_reference_of_time():
    """Test that the reference observable follows the review's reference of the plant, in samples of the controller."""
    trajectory = _simulate(KoopmanModelPredictiveControlSystem(plant='motor', controlled=False), 30.0)
    times = trajectory.times
    expected = 0.5 * np.cos(2 * np.pi * 0.01 * times / 3)
    assert np.allclose(trajectory.get_values('reference'), expected)


def test_koopman_model_predictive_control_takes_a_dictionary():
    """Test that a dictionary that holds the variables replaces the monomials of degree 3."""
    system = KoopmanModelPredictiveControlSystem(dictionary=PolynomialFeatures(2, include_bias=False), n_runs=4)
    assert system.prepare().form_ == 'co-simulation'


def test_koopman_model_predictive_control_rejects_unknown_plant():
    """Test that an unknown plant raises an error naming it."""
    with pytest.raises(ValidationError, match='boiler'):
        KoopmanModelPredictiveControlSystem('boiler').prepare()


def test_koopman_model_predictive_control_default_solver_is_co_simulation():
    """Test that a default run schedules the plant and the controller with the co-simulation solver."""
    trajectory = KoopmanModelPredictiveControlSystem(controlled=False).prepare().simulate(until=2.0, dt=1.0)
    assert trajectory.problem.form == FormKind.COUPLED
    assert KoopmanModelPredictiveControlSystem.field == 'engineering/control'
