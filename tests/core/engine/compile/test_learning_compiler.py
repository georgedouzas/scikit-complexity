"""Test the `_learning_compiler` module of `skcomplex.core.engine.compile` through `PassCompiler`."""

import numpy as np
import pytest

from skcomplex.core.components import Observable
from skcomplex.core.components.elements import GroupElement
from skcomplex.core.components.interactions import TrainingInteraction
from skcomplex.core.components.laws import DatasetSpec
from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.backends import JaxBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems._learning_problem_spec import LearningProblemSpec
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems.computer_science.neural_networks import ElmanNetworkSystem
from skcomplex.systems.computer_science.neural_networks._elman import TRAINED

from ....conftest import CallableGrowthInteraction, HoldingElement


def _elman(**changes):
    """Return a small Elman network whose training interaction takes some changed arguments."""
    system = ElmanNetworkSystem(n_bits=60).prepare().system_
    system.interactions[0].set_params(**changes)
    return system


def _issues(system, **arguments):
    """Return the issues the pass compiler reports for a system."""
    return PassCompiler(**arguments).prepare(system).report_.issues


def _codes(system, **arguments):
    """Return the codes of the issues the pass compiler reports for a system."""
    return [issue.code for issue in _issues(system, **arguments)]


def _squared(member):
    """Return a squared error that reads a variable the output units do not have."""
    return (member.y - member.missing) ** 2


def test_compile_a_system_that_trains_into_a_learning_problem():
    """Test that a system that trains compiles into a learning problem of its trained parameters and hyperparameters."""
    problem = PassCompiler().prepare(_elman()).run().problem_
    assert isinstance(problem, LearningProblemSpec)
    assert problem.states.get_entry('net/readout.W').shape == (1, 4)
    assert problem.parameters.has('training.learning_rate')
    assert problem.inner is not None
    assert problem.inner.backend == 'jax'


def test_untrained_parameter_path():
    """Test that a trained path that is no parameter of the nested system names the path and the parameters."""
    issues = _issues(_elman(trained=['readout.V']))
    assert [issue.code for issue in issues] == ['learning-trained-path']
    assert issues[0].path == 'training'
    assert 'readout.V' in issues[0].message
    assert 'readout.W' in issues[0].message


def test_the_named_parameters_are_trained_with_no_marker():
    """Test that the weights and biases the training names are adaptive states, though nothing declares them so."""
    compiler = PassCompiler().prepare(_elman())
    assert compiler.report_.ok
    assert set(TRAINED) <= set(compiler.learning_.inner_.states.names)
    assert 'inputs.x' in compiler.learning_.inner_.parameters.names


def test_unknown_loss():
    """Test that a loss that reads a variable the nested system does not have names the loss and the variable."""
    loss = Observable('error', group='outputs', reduce='sum', expression=_squared)
    issues = _issues(_elman(loss=loss))
    assert [issue.code for issue in issues] == ['learning-unknown-observable']
    assert 'error' in issues[0].message
    assert 'missing' in issues[0].message


@pytest.mark.parametrize(
    'values',
    [
        {'inputs.x': np.zeros((16, 12, 1)), 'outputs.target': np.zeros((16, 11, 1))},
        {'inputs.x': np.zeros((16, 12, 2)), 'outputs.target': np.zeros((16, 12, 1))},
    ],
)
def test_shape_mismatch(values):
    """Test that a dataset array with other steps or another trailing shape names the path and both shapes."""
    issues = _issues(_elman(data=DatasetSpec(values)))
    assert {issue.code for issue in issues} == {'learning-data-shape'}
    assert any('(16, 12, 1)' in issue.message for issue in issues)


def test_not_differentiable():
    """Test that a nested system holding a law given as a Python callable cannot be differentiated."""
    net = CompositeSystem(
        'net',
        elements=[GroupElement('cells', element=HoldingElement, size=2)],
        interactions=[CallableGrowthInteraction('growth', target='cells')],
    )
    data = DatasetSpec({'growth.r': np.zeros((2, 3))})
    loss = Observable('size', group='cells', reduce='sum', variable='n')
    training = TrainingInteraction('training', system='net', trained=['growth.r'], loss=loss, data=data)
    issues = _issues(CompositeSystem('fit', systems=[net], interactions=[training], time=DiscreteTime()))
    assert [issue.code for issue in issues] == ['learning-not-differentiable']
    assert 'discrete map or ordinary differential equations' in issues[0].message


def test_unsupported_solver():
    """Test that a solver other than the learning solver gives one error that names `GradientDescentSolver`."""
    issues = [issue for issue in _issues(_elman(), solver=RK4Solver()) if issue.severity == 'error']
    assert [issue.code for issue in issues] == ['solver-unsupported']
    assert 'GradientDescentSolver' in issues[0].message


def test_training_needs_discrete_time():
    """Test that a system that trains in a declared continuous time is reported."""
    system = _elman()._get_composite().set_params(time=ContinuousTime())
    assert 'law-time' in _codes(system)


def test_training_steps_once_per_epoch_without_a_time():
    """Test that a system that trains and declares no time steps once per epoch, with no issue."""
    assert _codes(_elman()) == []


def test_training_runs_on_the_jax_backend():
    """Test that a system that trains accepts the JAX backend."""
    assert _codes(_elman(), backend=JaxBackend()) == []
