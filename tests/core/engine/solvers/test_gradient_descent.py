"""Test the `_gradient_descent` module of `skcomplex.core.engine.solvers`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction, RateInteraction, TrainingInteraction
from skcomplex.core.components.laws import BaseLaw, RateLaw
from skcomplex.core.components.laws.utils import contraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import GradientDescentSolver
from skcomplex.core.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import DatasetSpec, ObservableSpec, ParameterSpec, RoleSpec, StateSpec, StepSpec
from skcomplex.systems.computer_science.neural_networks import ElmanNetworkSystem

RATES = {'alpha': 1.1, 'beta': 0.4, 'gamma': 0.4, 'delta': 0.1}
TIMES = np.linspace(0.4, 20.0, 50)


class PairElement(Element):
    """Prey and predators with adaptive rates and observed abundances."""

    def __init__(
        self,
        label: str,
        visual: object = None,
        *,
        prey: Annotated[float, StateSpec()] = 10.0,
        predator: Annotated[float, StateSpec()] = 5.0,
        alpha: Annotated[float, ParameterSpec(adaptive=True)] = 1.0,
        beta: Annotated[float, ParameterSpec(adaptive=True)] = 1.0,
        gamma: Annotated[float, ParameterSpec(adaptive=True)] = 1.0,
        delta: Annotated[float, ParameterSpec(adaptive=True)] = 1.0,
        observed_prey: Annotated[float, ParameterSpec()] = 0.0,
        observed_predator: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.prey = prey
        self.predator = predator
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.observed_prey = observed_prey
        self.observed_predator = observed_predator


def _prey_rate(s):
    """Return the rate of the prey."""
    return s.target.alpha * s.target.prey - s.target.beta * s.target.prey * s.target.predator


def _predator_rate(s):
    """Return the rate of the predators."""
    return s.target.delta * s.target.prey * s.target.predator - s.target.gamma * s.target.predator


def _squared_error(s):
    """Return the squared error of the abundances to the observed ones."""
    return (s.pair.prey - s.pair.observed_prey) ** 2 + (s.pair.predator - s.pair.observed_predator) ** 2


def _output_error(member):
    """Return the squared error of the prediction of an output unit."""
    return (member.y - member.target) ** 2


def _fitting(learning_rate=0.05, optimizer='adam'):
    """Return a system that fits the rates of a Lotka-Volterra system, started 30% off, to its series."""
    solution = solve_ivp(
        lambda t, z: [
            RATES['alpha'] * z[0] - RATES['beta'] * z[0] * z[1],
            RATES['delta'] * z[0] * z[1] - RATES['gamma'] * z[1],
        ],
        (0.0, 20.0),
        [10.0, 5.0],
        t_eval=TIMES,
        rtol=1e-12,
        atol=1e-12,
    )
    system = System(
        'lv',
        elements=[PairElement('pair', **{name: 1.3 * value for name, value in RATES.items()})],
        interactions=[
            RateInteraction('prey_rate', target='pair', state='prey', rate=_prey_rate),
            RateInteraction('predator_rate', target='pair', state='predator', rate=_predator_rate),
        ],
    )
    data = DatasetSpec(
        {'pair.observed_prey': solution.y[0][np.newaxis], 'pair.observed_predator': solution.y[1][np.newaxis]},
        times=TIMES,
    )
    training = TrainingInteraction(
        'fit',
        system='lv',
        trained=[f'pair.{name}' for name in RATES],
        loss=ObservableSpec('error', _squared_error),
        data=data,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=1,
    )
    return System('fitting', systems=[system], interactions=[training], time=DiscreteTime())


def _trajectory(system, until, **arguments):
    """Return the trajectory of a system that trains, run with the seed 0."""
    return simulate(system, until=until, random_state=0, **arguments)


def _gradient(system, rate=1.0):
    """Return the gradient of the loss of the whole dataset, from the change of one full batch step of SGD."""
    training = system.interactions[0]
    training.set_params(optimizer='sgd', learning_rate=rate, batch_size=training.data.sequences)
    trajectory = _trajectory(system, 1)
    names = trajectory.problem.states.names
    return {name: (trajectory.values(name)[0] - trajectory.values(name)[1]) / rate for name in names}


def _loss(system, values):
    """Return the loss of the whole dataset at given trained values, the reporter of frame 0."""
    trajectory = simulate(system, until=1, random_state=0, initial_state=values)
    return trajectory.values('training/loss')[0]


def test_check_unknown_precision():
    """Test that an unknown precision raises an error."""
    with pytest.raises(ValidationError, match='float16'):
        GradientDescentSolver(precision='float16').prepare()


def test_automatic_solver():
    """Test that a system that trains runs on the learning solver and the JAX backend without being told."""
    problem = simulate(ElmanNetworkSystem(n_bits=60).prepare().system_, until=0).problem
    assert DEFAULT_SOLVERS[problem.form] is GradientDescentSolver
    assert problem.inner.backend == 'jax'


def test_epochs_as_frames():
    """Test that every frame is one epoch and frame 0 holds the initial weights of the inner system."""
    system = ElmanNetworkSystem(n_bits=60).prepare().system_
    weights = system.systems[0].interactions[2].W
    trajectory = _trajectory(system, 5)
    assert trajectory.times.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert trajectory.values('net/readout.W').shape == (6, 1, 4)
    np.testing.assert_array_equal(trajectory.values('net/readout.W')[0], weights)


def test_time_between_frames():
    """Test that a fraction of an epoch between frames raises an error naming the value."""
    with pytest.raises(ValidationError, match=r'0\.5.*whole number of epochs'):
        simulate(ElmanNetworkSystem(n_bits=60).prepare().system_, until=2, dt=0.5)


def test_loss_at_the_start():
    """Test that the loss at frame 0 is the mean of the loss over NumPy runs of the inner system on every sequence."""
    system = ElmanNetworkSystem(n_bits=30).prepare().system_
    data, net = system.interactions[0].data, system.systems[0]
    trajectory = _trajectory(system, 1)
    error = ObservableSpec('error', group='outputs', reduce='sum', expression=_output_error)
    losses = []
    for row in range(data.sequences):
        inputs, targets = data.values['inputs.x'][row], data.values['outputs.target'][row]
        schedule = {
            'inputs.x': lambda t, values=inputs: values[min(round(t), len(values) - 1)],
            'outputs.target': lambda t, values=targets: values[max(round(t) - 1, 0)],
        }
        run = simulate(net, until=data.steps, backend=NumpyBackend(), schedule=schedule, observables=[error])
        losses.append(run.values('error')[1:].mean())
    assert trajectory.values('training/loss')[0] == pytest.approx(np.mean(losses), abs=1e-10)


def test_masked_steps():
    """Test that the loss and its gradient do not depend on the outputs at steps of weight 0."""
    system, other = ElmanNetworkSystem(n_bits=60).prepare().system_, ElmanNetworkSystem(n_bits=60).prepare().system_
    data = system.interactions[0].data
    weights = np.ones((data.sequences, data.steps))
    weights[:, :3] = 0.0
    targets = data.values['outputs.target'].copy()
    targets[:, :3] = 1.0 - targets[:, :3]
    system.interactions[0].set_params(data=DatasetSpec(dict(data.values), weights=weights))
    other.interactions[0].set_params(data=DatasetSpec({**data.values, 'outputs.target': targets}, weights=weights))
    gradient, changed = _gradient(system), _gradient(other)
    for name, value in gradient.items():
        np.testing.assert_allclose(changed[name], value, atol=1e-12)


def test_same_seed():
    """Test that two runs with the same seed record equal trajectories."""
    first, second = _trajectory(ElmanNetworkSystem(n_bits=60).prepare().system_, 20), _trajectory(
        ElmanNetworkSystem(n_bits=60).prepare().system_,
        20,
    )
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.values('training/loss'), second.values('training/loss'))


def test_updates_per_epoch():
    """Test that 996 sequences in batches of 16 give 63 updates per epoch."""
    problem = simulate(ElmanNetworkSystem().prepare().system_, until=0).problem
    stepper = GradientDescentSolver(random_state=0, until=1).prepare(problem)
    stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 1.0))
    assert int(stepper.optimizer_state_.inner_state[0].count) == 63


def test_zero_learning_rate():
    """Test that a learning rate of 0 keeps every trained parameter at its initial value."""
    system = ElmanNetworkSystem(n_bits=60, learning_rate=0.0).prepare().system_
    states = _trajectory(system, 5).states
    np.testing.assert_array_equal(states[-1], states[0])


def test_learning_rate_schedule():
    """Test that a schedule that halves the learning rate every 50 epochs halves the recorded parameter."""
    schedule = {'training.learning_rate': lambda t: 0.05 * 0.5 ** (t // 50)}
    values = _trajectory(ElmanNetworkSystem(n_bits=30).prepare().system_, 150, schedule=schedule).values(
        'training.learning_rate',
    )
    assert values[[49, 50, 100, 150]].tolist() == pytest.approx([0.05, 0.025, 0.0125, 0.00625])


def test_gradient_against_finite_differences():
    """Test that the gradient of the loss of an Elman network agrees with central finite differences."""
    system = ElmanNetworkSystem(n_bits=30).prepare().system_
    gradient = _gradient(system)
    problem = PassCompiler().prepare(system).run().problem_
    step = 1e-6
    for name in problem.states.names:
        value = np.asarray(problem.states.read(problem.initial_state, name), dtype=float)
        index = np.unravel_index(0, value.shape) if value.shape else ()
        up, down = value.copy(), value.copy()
        up[index] += step
        down[index] -= step
        difference = (_loss(system, {name: up}) - _loss(system, {name: down})) / (2 * step)
        assert gradient[name][index] == pytest.approx(difference, rel=1e-6, abs=1e-9)


def test_full_and_truncated_gradients():
    """Test that truncation changes the gradient of the input weights but not of the readout weights."""
    full, truncated = _gradient(ElmanNetworkSystem(n_bits=30).prepare().system_), _gradient(
        ElmanNetworkSystem(n_bits=30, truncation=1).prepare().system_,
    )
    assert not np.allclose(full['net/input.D'], truncated['net/input.D'])
    np.testing.assert_allclose(full['net/readout.W'], truncated['net/readout.W'], atol=1e-12)


def test_gradient_through_an_ode():
    """Test that the gradient of the squared error of a Lotka-Volterra system agrees with finite differences."""
    gradient = _gradient(_fitting(), rate=1e-6)
    problem = PassCompiler().prepare(_fitting()).run().problem_
    step = 1e-6
    for name in problem.states.names:
        value = float(problem.states.read(problem.initial_state, name))
        up = simulate(_fitting(), until=1, random_state=0, initial_state={name: value + step})
        down = simulate(_fitting(), until=1, random_state=0, initial_state={name: value - step})
        loss_up, loss_down = (item.values('fit/loss')[0] for item in (up, down))
        assert gradient[name] == pytest.approx((loss_up - loss_down) / (2 * step), rel=1e-5)


def test_lotka_volterra_fit():
    """Test that Adam fits the four rates of a Lotka-Volterra system within 1% and lowers the loss 10^4 times."""
    trajectory = _trajectory(_fitting(), 500)
    for name, value in RATES.items():
        assert trajectory.values(f'lv/pair.{name}')[-1] == pytest.approx(value, rel=1e-2)
    loss = trajectory.values('fit/loss')
    assert loss[-1] < 1e-4 * loss[0]


class TargetNodeElement(Element):
    """Node with a state and a target that a dataset sets."""

    def __init__(
        self,
        label: str,
        visual: object = None,
        *,
        x: Annotated[float, StateSpec()] = 0.5,
        target: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.target = target


class AdaptiveCubicInteraction(Interaction):
    """Rate of every member of a group, the contraction of an adaptive tensor of order 3 with the states."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        T: Annotated[object, ParameterSpec(shape=('target', 'target', 'target'), adaptive=True)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.T = T

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the contraction of the tensor with the states."""
        return [RateLaw(s.target.x, contraction(s.T, s.target.x, s.target.x))]


def _node_error(member):
    """Return the squared error of the state of a node to its target."""
    return (member.x - member.target) ** 2


def test_tensor_trains():
    """Test that the gradient of an adaptive tensor of order 3 read through a contraction agrees with differences."""
    rng = np.random.default_rng(0)
    tensor = rng.normal(0.0, 0.3, (4, 4, 4))
    inner = System(
        'community',
        elements=[GroupElement('species', element=TargetNodeElement, size=4)],
        interactions=[AdaptiveCubicInteraction('cubic', target='species', T=tensor)],
    )
    data = DatasetSpec({'species.target': rng.uniform(0.4, 0.8, (1, 5, 4))}, times=np.linspace(0.2, 1.0, 5))
    loss = ObservableSpec('error', group='species', reduce='sum', expression=_node_error)
    training = TrainingInteraction('training', system='community', trained=['cubic.T'], loss=loss, data=data)
    system = System('fitting', systems=[inner], interactions=[training], time=DiscreteTime())
    gradient = _gradient(system, rate=1e-6)['community/cubic.T']
    step = 1e-6
    for index in [(0, 0, 0), (1, 2, 3), (3, 1, 0)]:
        up, down = tensor.copy(), tensor.copy()
        up[index] += step
        down[index] -= step
        difference = _loss(system, {'community/cubic.T': up}) - _loss(system, {'community/cubic.T': down})
        assert gradient[index] == pytest.approx(difference / (2 * step), rel=1e-6)


def test_trained_system_agrees_with_training():
    """Test that the inner system with the trained values of the last frame records the loss the law evaluates."""
    system = ElmanNetworkSystem(n_bits=30).prepare().system_
    data, net = system.interactions[0].data, system.systems[0]
    trajectory = _trajectory(system, 3)
    trained = {name: trajectory.values(name)[-1] for name in trajectory.problem.states.names}
    single = ElmanNetworkSystem(n_bits=30).prepare().system_
    first = DatasetSpec({path: values[:1] for path, values in data.values.items()})
    single.interactions[0].set_params(data=first, metrics=())
    expected = _loss(single, trained)
    for interaction, name in zip(net.interactions, ('D', 'A', 'W'), strict=True):
        interaction.set_params(**{name: trained[f'net/{interaction.label}.{name}']})
    net.elements[1].set_params(values={'b': trained['net/hidden.b']})
    net.elements[2].set_params(values={'c': trained['net/outputs.c']})
    inputs, targets = data.values['inputs.x'][0], data.values['outputs.target'][0]
    schedule = {
        'inputs.x': lambda t: inputs[min(round(t), len(inputs) - 1)],
        'outputs.target': lambda t: targets[max(round(t) - 1, 0)],
    }
    error = ObservableSpec('error', group='outputs', reduce='sum', expression=_output_error)
    run = simulate(net, until=data.steps, backend=NumpyBackend(), schedule=schedule, observables=[error])
    assert run.values('error')[1:].mean() == pytest.approx(expected, abs=1e-10)
