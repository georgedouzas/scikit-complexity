"""Test the lowering of reductions and observable reads, their order, and their agreement across backends."""

import math
import time

import numpy as np
import pytest
import scipy.sparse
from sage.all__sagemath_symbolics import cos, heaviside, sin, tanh

from skcomplex.core.base import Domain, RoleError, ValidationError, state
from skcomplex.core.compile import compile_system
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Jump, Rate, Update, count, maximum, mean, minimum, total, variance
from skcomplex.core.runtime import Simulation
from skcomplex.core.system import Observable, System

RNG = np.random.default_rng(0)


def _cosine(member):
    """Return the cosine of a member's phase."""
    return cos(member.theta)


def _sine(member):
    """Return the sine of a member's phase."""
    return sin(member.theta)


Unit = Element.define('Unit', states={'x': 0.0})
Phase = Element.define(
    'Phase',
    states={'theta': state(0.0, domain=Domain.ANGLE)},
    parameters={'omega': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega)],
)
Inhibitor = Element.define('Inhibitor', states={'z': 0.0}, parameters={'phi': 3.0})
Drive = Interaction.define(
    'Drive',
    roles=['inhibitor', 'oscillators'],
    parameters={'theta_z': 0.5},
    laws=lambda s: [
        Rate(s.inhibitor.z, s.inhibitor.phi * (heaviside(maximum(s.oscillators.x) - s.theta_z) - s.inhibitor.z)),
    ],
)
MeanPulse = Interaction.define(
    'MeanPulse',
    roles=['neurons'],
    parameters={'kappa': 1.5},
    laws=lambda s: [
        Rate(
            s.neurons.theta,
            (1 + cos(s.neurons.theta)) * s.kappa * mean(2 / 3 * (1 - cos(s.neurons.theta)) ** 2),
        ),
    ],
)
MeanFieldCoupling = Interaction.define(
    'MeanFieldCoupling',
    roles=['target', 'X', 'Y'],
    parameters={'K': 1.0},
    laws=lambda s: [
        Rate(s.target.theta, s.K * (s.Y.value * cos(s.target.theta) - s.X.value * sin(s.target.theta))),
    ],
)
Reader = Interaction.define('Reader', roles=['target', 'source'], laws=lambda s: [Rate(s.target.x, s.source.value)])


def _group_of(laws, size=10, values=None):
    """Return a system of one group of units whose laws read reductions."""
    unit = Element.define('Reducing', states={'x': 0.0}, laws=laws)
    values = RNG.normal(size=size) if values is None else np.asarray(values, dtype=float)
    return System(components=[unit.group('units', size, x=values)]), values


def _rates(system, backend='numpy'):
    """Return the drift of a system at its initial state."""
    problem = compile_system(system, backend=backend)
    return problem.drift(0.0, problem.initial_state, problem.initial_parameters)


def _kuramoto(size, *, observables=False):
    """Return a mean-field Kuramoto group, coupled through reductions or through observables."""
    theta, omega = RNG.uniform(0, 2 * math.pi, size), RNG.normal(size=size)
    oscillators = Phase.group('oscillators', size, theta=theta, omega=omega)
    if observables:
        return (
            System(
                components=[oscillators],
                interactions=[MeanFieldCoupling('coupling', target='oscillators', X='X', Y='Y', K=2.0)],
            ),
            theta,
            omega,
        )
    coupling = Interaction.define(
        'Coupling',
        roles=['target'],
        parameters={'K': 1.0},
        laws=lambda s: [
            Rate(
                s.target.theta,
                s.K
                * (mean(sin(s.target.theta)) * cos(s.target.theta) - mean(cos(s.target.theta)) * sin(s.target.theta)),
            ),
        ],
    )
    return (
        System(components=[oscillators], interactions=[coupling('coupling', target='oscillators', K=2.0)]),
        theta,
        omega,
    )


def _all_to_all(theta, omega, K=2.0):
    """Return the Kuramoto drift summed over every pair."""
    return omega + K / theta.size * np.sin(theta[np.newaxis, :] - theta[:, np.newaxis]).sum(axis=1)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_nonlinear_function_of_total(backend):
    """Test that every member reads the hyperbolic tangent of the group total."""
    system, x = _group_of(lambda s: [Rate(s.x, -s.x + tanh(total(s.x)))], size=50)
    np.testing.assert_allclose(_rates(system, backend), -x + np.tanh(x.sum()), atol=1e-12)


def test_instantaneous_mean_in_interaction():
    """Test that theta neurons read the mean pulse of all neurons at the current state, with no other state."""
    theta = RNG.uniform(0, 2 * math.pi, 40)
    Neuron = Element.define('Neuron', states={'theta': state(0.0, domain=Domain.ANGLE)})
    system = System(
        components=[Neuron.group('neurons', 40, theta=theta)],
        interactions=[MeanPulse('pulses', neurons='neurons')],
    )
    problem = compile_system(system)
    expected = (1 + np.cos(theta)) * 1.5 * np.mean(2 / 3 * (1 - np.cos(theta)) ** 2)
    np.testing.assert_allclose(
        problem.drift(0.0, problem.initial_state, problem.initial_parameters),
        expected,
        atol=1e-12,
    )
    assert problem.states.names == ['neurons.theta']


@pytest.mark.parametrize(('largest', 'drive'), [(0.4, 0.0), (0.6, 1.0)])
def test_maximum_drives_single_element(largest, drive):
    """Test that the inhibitor is driven exactly when some oscillator exceeds the threshold."""
    values = np.linspace(-1.0, largest, 8)
    system = System(
        components=[Unit.group('oscillators', 8, x=values), Inhibitor('inhibitor')],
        interactions=[Drive('drive', inhibitor='inhibitor', oscillators='oscillators')],
    )
    problem = compile_system(system)
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert rates[problem.states.entry('inhibitor.z').start] == pytest.approx(3.0 * drive)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_count_and_mean_with_condition(backend):
    """Test that a count and a mean with a condition read the selected members."""
    values = [0.1, 0.2, 0.9, 0.3, 0.7, 0.0, 0.4, 0.6, 0.2, 0.1]
    system, x = _group_of(
        lambda s: [Rate(s.x, count(where=s.x > 0.5) + 10 * mean(s.x, where=s.x > 0.5))],
        values=values,
    )
    np.testing.assert_allclose(_rates(system, backend), 3 + 10 * x[x > 0.5].mean(), atol=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_no_selected_members(backend):
    """Test that the reductions of no selected members are 0, the count included."""
    selected = lambda s: s.x > 10  # noqa: E731
    system, _ = _group_of(
        lambda s: [
            Rate(
                s.x,
                total(s.x, where=selected(s))
                + mean(s.x, where=selected(s))
                + maximum(s.x, where=selected(s))
                + minimum(s.x, where=selected(s))
                + variance(s.x, where=selected(s))
                + count(where=selected(s)),
            ),
        ],
    )
    np.testing.assert_array_equal(_rates(system, backend), 0.0)


@pytest.mark.parametrize('sparse', [False, True], ids=['dense', 'sparse'])
@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_payoff_matrix(sparse, backend):
    """Test that a weighted total gives every member its row of the payoff matrix times the shares."""
    payoff = RNG.normal(size=(3, 3))
    weights = scipy.sparse.csr_matrix(payoff) if sparse else payoff
    system, x = _group_of(lambda s: [Rate(s.x, total(s.x, weights=weights))], size=3)
    np.testing.assert_allclose(_rates(system, backend), payoff @ x, atol=1e-12)


def test_weights_of_wrong_shape():
    """Test that weights whose shape does not match the readers and the members are rejected."""
    system, _ = _group_of(lambda s: [Rate(s.x, total(s.x, weights=np.ones((3, 4))))], size=3)
    with pytest.raises(ValidationError, match=r'total\(x_units, weights=.*units.*\(3, 4\).*\(3, 3\)'):
        compile_system(system)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_mean_fitness(backend):
    """Test that nested reductions give the replicator equation, whose rates sum to 0 on the simplex."""
    payoff = RNG.normal(size=(4, 4))
    shares = RNG.uniform(size=4)
    shares /= shares.sum()

    def replicator(s):
        fitness = total(s.x, weights=payoff)
        return [Rate(s.x, s.x * (fitness - total(s.x * fitness)))]

    system, x = _group_of(replicator, values=shares, size=4)
    rates = _rates(system, backend)
    np.testing.assert_allclose(rates, x * (payoff @ x - x @ payoff @ x), atol=1e-12)
    assert abs(rates.sum()) < 1e-12


def test_two_groups_in_one_reduction():
    """Test that a reduction reading two groups names both and asks for one reduction per group."""
    Predation = Interaction.define(
        'Predation',
        roles=['prey', 'predators', 'target'],
        laws=lambda s: [Rate(s.target.x, total(s.prey.x * s.predators.x))],
    )
    system = System(
        components=[Unit.group('prey', 3), Unit.group('predators', 3), Unit('target')],
        interactions=[Predation('predation', prey='prey', predators='predators', target='target')],
    )
    with pytest.raises(RoleError, match=r'total\(.*predators and prey.*one reduction per group'):
        compile_system(system)


def test_interaction_reads_second_group_through_reductions():
    """Test that an interaction acts on one group and reads another group through a mean."""
    Coupling = Interaction.define(
        'Coupling',
        roles=['target', 'source'],
        laws=lambda s: [Rate(s.target.x, mean(s.source.x) - s.target.x)],
    )
    first, second = RNG.normal(size=4), RNG.normal(size=6)
    system = System(
        components=[Unit.group('first', 4, x=first), Unit.group('second', 6, x=second)],
        interactions=[Coupling('coupling', target='first', source='second')],
    )
    for backend in ('numpy', 'numba'):
        np.testing.assert_allclose(_rates(system, backend), np.r_[second.mean() - first, np.zeros(6)], atol=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_kuramoto_through_observables(backend):
    """Test that laws reading observables of their own partition compile to one ODE at the current state."""
    system, theta, omega = _kuramoto(40, observables=True)
    observables = [Observable('X', _cosine, group='oscillators'), Observable('Y', _sine, group='oscillators')]
    problem = compile_system(system, backend=backend, observables=observables)
    assert problem.form.name == 'ODE'
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    np.testing.assert_allclose(rates, _all_to_all(theta, omega), atol=1e-12)


def test_callable_observable_in_same_partition():
    """Test that a law reading a callable observable of its own partition is rejected."""
    system = System(
        components=[Unit.group('herd', 3), Unit('soil')],
        interactions=[Reader('reader', target='soil', source='spread')],
    )
    with pytest.raises(ValidationError, match=r'spread.*symbolically.*another'):
        compile_system(system, observables=[Observable('spread', group='herd', variable='x', reduce=np.ptp)])


def test_observable_read_in_its_own_partition():
    """Test that the value a law reads equals the recorded observable at every frame."""
    Saver = Element.define(
        'Saver',
        states={'W': 1.0},
        parameters={'r': 0.1},
        laws=lambda s: [Rate(s.W, s.r * s.W)],
    )
    system = System(
        components=[Saver.group('savers', 5, W=RNG.uniform(1, 2, 5), r=RNG.uniform(0, 0.2, 5)), Unit('account')],
        interactions=[Reader('reader', target='account', source='mean_wealth')],
    )
    observables = [Observable('mean_wealth', group='savers', variable='W')]
    simulation = Simulation(system, dt=0.5, observables=observables).run(until=5)
    problem, trajectory = simulation.problem_, simulation.trajectory_
    assert not problem.partitions
    index = problem.states.entry('account.x').start
    for row, values in enumerate(trajectory.states):
        rate = problem.drift(0.0, values, problem.initial_parameters)[index]
        assert rate == pytest.approx(trajectory['mean_wealth'][row], abs=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_synchronous_map(backend):
    """Test that every member of coupled maps updates from the mean of the values at step n."""
    epsilon = 0.3
    system, x = _group_of(
        lambda s: [Update(s.x, (1 - epsilon) * 4 * s.x * (1 - s.x) + epsilon * mean(4 * s.x * (1 - s.x)))],
        values=RNG.uniform(size=10),
    )
    problem = compile_system(system, backend=backend)
    updated = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    logistic = 4 * x * (1 - x)
    np.testing.assert_allclose(updated, (1 - epsilon) * logistic + epsilon * logistic.mean(), atol=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_propensity_after_event(backend):
    """Test that the propensities of mean-field SIS equal those computed from the state after every event."""
    beta, gamma, size = 2.0, 1.0, 30

    def sis(s):
        return [
            Jump(beta * (1 - s.x) * count(where=s.x > 0.5) / size, {s.x: 1}),
            Jump(gamma * s.x, {s.x: -1}),
        ]

    system, _ = _group_of(sis, size=size, values=(RNG.uniform(size=size) < 0.3).astype(float))
    problem = compile_system(system, backend=backend)
    state, p = problem.initial_state.copy(), problem.initial_parameters
    for channel in RNG.integers(0, 2 * size, 20):
        infected = state > 0.5
        expected = np.r_[beta * (1 - state) * infected.sum() / size, gamma * state]
        rates = problem.propensities(0.0, state, p)
        np.testing.assert_allclose(rates, expected, atol=1e-12)
        if rates[channel] > 0:
            state = state + problem.stoichiometry[channel]
    trajectory = Simulation(system, backend=backend).run(until=2).trajectory_
    assert set(np.unique(trajectory['units.x'])) <= {0.0, 1.0}


def test_backends_agree():
    """Test that NumPy and Numba give the same mean-field Kuramoto drift on 2000 members."""
    system, _, _ = _kuramoto(2000)
    numpy_rates, numba_rates = _rates(system, 'numpy'), _rates(system, 'numba')
    np.testing.assert_allclose(numba_rates, numpy_rates, rtol=1e-12, atol=1e-12)


def test_linear_cost():
    """Test that one Numba drift evaluation grows linearly with the size of a mean-field group."""
    durations = []
    for size in (10_000, 100_000):
        problem = compile_system(_kuramoto(size)[0], backend='numba')
        x, p = problem.initial_state, problem.initial_parameters
        problem.drift(0.0, x, p)
        samples = []
        for _ in range(15):
            start = time.perf_counter()
            problem.drift(0.0, x, p)
            samples.append(time.perf_counter() - start)
        durations.append(min(samples))
    assert durations[1] <= 15 * durations[0]


def test_element_set_reduction():
    """Test that a role bound to an element set applies to each element and reduces over the set."""
    Total = Interaction.define('Total', roles=['units'], laws=lambda s: [Rate(s.units.x, total(s.units.x))])
    system = System(
        components=[Unit(f'unit_{index}', x=float(index + 1)) for index in range(3)],
        interactions=[Total('total', units=['unit_0', 'unit_1', 'unit_2'])],
    )
    problem = compile_system(system)
    assert problem.states.names == ['unit_0.x', 'unit_1.x', 'unit_2.x']
    for backend in ('numpy', 'numba'):
        np.testing.assert_allclose(_rates(system, backend), [6.0, 6.0, 6.0])


def _shared(share, values):
    """Return a system of one group whose interaction gives every member a share of a reduction."""
    Share = Interaction.define('Share', roles=['cells'], laws=lambda s: [Rate(s.cells.x, share(s))])
    return System(
        components=[Unit.group('units', len(values), x=values)],
        interactions=[Share('share', cells='units')],
    )


def test_compile_system_divides_by_count_in_interaction():
    """Test that an interaction law may divide a member state by the size of its group."""
    system = _shared(lambda s: s.cells.x / count(over=s.cells), np.arange(1.0, 5.0))
    np.testing.assert_allclose(_rates(system), np.arange(1.0, 5.0) / 4, atol=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_compile_system_divides_by_weighted_total_in_interaction(backend):
    """Test that an interaction law may divide a member state by a weighted total."""
    weights, values = RNG.uniform(0.5, 1.5, size=(4, 4)), np.arange(1.0, 5.0)
    system = _shared(lambda s: s.cells.x / total(s.cells.x, weights=weights), values)
    np.testing.assert_allclose(_rates(system, backend), values / (weights @ values), atol=1e-12)


def test_element_set_of_mixed_types():
    """Test that a role bound to elements of two types names the role and both types."""
    Other = Element.define('Other', states={'x': 0.0})
    Total = Interaction.define('Total', roles=['units'], laws=lambda s: [Rate(s.units.x, total(s.units.x))])
    system = System(
        components=[Unit('a'), Other('b')],
        interactions=[Total('total', units=['a', 'b'])],
    )
    with pytest.raises(RoleError, match=r"'units'.*Unit and Other"):
        compile_system(system)
