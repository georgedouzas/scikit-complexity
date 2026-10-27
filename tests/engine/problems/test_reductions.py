"""Test the lowering of reductions and observable reads, their order, and their agreement across backends."""

import math
import time

import numpy as np
import pytest
import scipy.sparse
from sage.all__sagemath_symbolics import cos, heaviside, sin, tanh
from skcomplex.core.components import Element
from skcomplex.core.declarations import parameter, role

from skcomplex.core.base import BaseInteraction, NumericDomain, RoleError, ValidationError, state
from skcomplex.core.elements import GroupElement
from skcomplex.core.laws import JumpLaw, RateLaw, UpdateLaw, count, maximum, mean, minimum, total, variance
from skcomplex.core.specs import ObservableSpec
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation

RNG = np.random.default_rng(0)


def _cosine(member):
    """Return the cosine of a member's phase."""
    return cos(member.theta)


def _sine(member):
    """Return the sine of a member's phase."""
    return sin(member.theta)


class UnitElement(Element):
    """Unit element."""

    x = state(0.0)


class PhaseElement(Element):
    """Phase element."""

    theta = state(0.0, domain=NumericDomain.ANGLE)
    omega = parameter(1.0)

    def laws(self, s):
        """Return the laws of the phase element."""
        return [RateLaw(s.theta, s.omega)]


class InhibitorElement(Element):
    """Inhibitor element."""

    z = state(0.0)
    phi = parameter(3.0)


class DriveInteraction(BaseInteraction):
    """Drive interaction."""

    inhibitor = role()
    oscillators = role()
    theta_z = parameter(0.5)

    def laws(self, s):
        """Return the laws of the drive interaction."""
        return [
            RateLaw(s.inhibitor.z, s.inhibitor.phi * (heaviside(maximum(s.oscillators.x) - s.theta_z) - s.inhibitor.z)),
        ]


class MeanPulseInteraction(BaseInteraction):
    """Mean pulse interaction."""

    neurons = role()
    kappa = parameter(1.5)

    def laws(self, s):
        """Return the laws of the mean pulse interaction."""
        return [
            RateLaw(
                s.neurons.theta,
                (1 + cos(s.neurons.theta)) * s.kappa * mean(2 / 3 * (1 - cos(s.neurons.theta)) ** 2),
            ),
        ]


class MeanFieldCouplingInteraction(BaseInteraction):
    """Mean field coupling interaction."""

    target = role()
    X = role()
    Y = role()
    K = parameter(1.0)

    def laws(self, s):
        """Return the laws of the mean field coupling interaction."""
        return [
            RateLaw(s.target.theta, s.K * (s.Y.value * cos(s.target.theta) - s.X.value * sin(s.target.theta))),
        ]


class ReaderInteraction(BaseInteraction):
    """Reader interaction."""

    target = role()
    source = role()

    def laws(self, s):
        """Return the laws of the reader interaction."""
        return [RateLaw(s.target.x, s.source.value)]


def _group_of(laws, size=10, values=None):
    """Return a system of one group of units whose laws read reductions."""

    class ReducingElement(Element):
        """Reducing element."""

        x = state(0.0)

        def laws(self, s):
            """Return the laws of the reducing element."""
            return laws(s)

    unit = ReducingElement
    values = RNG.normal(size=size) if values is None else np.asarray(values, dtype=float)
    return System(elements=[GroupElement('units', element=unit, size=size, values={'x': values})]), values


def _rates(system, backend='numpy'):
    """Return the drift of a system at its initial state."""
    problem = compile_system(system, backend=backend)
    return problem.drift(0.0, problem.initial_state, problem.initial_parameters)


def _kuramoto(size, *, observables=False):
    """Return a mean-field Kuramoto group, coupled through reductions or through observables."""
    theta, omega = RNG.uniform(0, 2 * math.pi, size), RNG.normal(size=size)
    oscillators = GroupElement('oscillators', element=PhaseElement, size=size, values={'theta': theta, 'omega': omega})
    if observables:
        return (
            System(
                elements=[oscillators],
                interactions=[MeanFieldCouplingInteraction('coupling', target='oscillators', X='X', Y='Y', K=2.0)],
            ),
            theta,
            omega,
        )

    class CouplingInteraction(BaseInteraction):
        """Coupling interaction."""

        target = role()
        K = parameter(1.0)

        def laws(self, s):
            """Return the laws of the coupling interaction."""
            return [
                RateLaw(
                    s.target.theta,
                    s.K
                    * (
                        mean(sin(s.target.theta)) * cos(s.target.theta)
                        - mean(cos(s.target.theta)) * sin(s.target.theta)
                    ),
                ),
            ]

    coupling = CouplingInteraction
    return (
        System(elements=[oscillators], interactions=[coupling('coupling', target='oscillators', K=2.0)]),
        theta,
        omega,
    )


def _all_to_all(theta, omega, K=2.0):
    """Return the Kuramoto drift summed over every pair."""
    return omega + K / theta.size * np.sin(theta[np.newaxis, :] - theta[:, np.newaxis]).sum(axis=1)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_nonlinear_function_of_total(backend):
    """Test that every member reads the hyperbolic tangent of the group total."""
    system, x = _group_of(lambda s: [RateLaw(s.x, -s.x + tanh(total(s.x)))], size=50)
    np.testing.assert_allclose(_rates(system, backend), -x + np.tanh(x.sum()), atol=1e-12)


def test_instantaneous_mean_in_interaction():
    """Test that theta neurons read the mean pulse of all neurons at the current state, with no other state."""
    theta = RNG.uniform(0, 2 * math.pi, 40)

    class NeuronElement(Element):
        """Neuron element."""

        theta = state(0.0, domain=NumericDomain.ANGLE)

    system = System(
        elements=[GroupElement('neurons', element=NeuronElement, size=40, values={'theta': theta})],
        interactions=[MeanPulseInteraction('pulses', neurons='neurons')],
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
        elements=[
            GroupElement('oscillators', element=UnitElement, size=8, values={'x': values}),
            InhibitorElement('inhibitor'),
        ],
        interactions=[DriveInteraction('drive', inhibitor='inhibitor', oscillators='oscillators')],
    )
    problem = compile_system(system)
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert rates[problem.states.entry('inhibitor.z').start] == pytest.approx(3.0 * drive)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_count_and_mean_with_condition(backend):
    """Test that a count and a mean with a condition read the selected members."""
    values = [0.1, 0.2, 0.9, 0.3, 0.7, 0.0, 0.4, 0.6, 0.2, 0.1]
    system, x = _group_of(
        lambda s: [RateLaw(s.x, count(where=s.x > 0.5) + 10 * mean(s.x, where=s.x > 0.5))],
        values=values,
    )
    np.testing.assert_allclose(_rates(system, backend), 3 + 10 * x[x > 0.5].mean(), atol=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_no_selected_members(backend):
    """Test that the reductions of no selected members are 0, the count included."""
    selected = lambda s: s.x > 10  # noqa: E731
    system, _ = _group_of(
        lambda s: [
            RateLaw(
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
    system, x = _group_of(lambda s: [RateLaw(s.x, total(s.x, weights=weights))], size=3)
    np.testing.assert_allclose(_rates(system, backend), payoff @ x, atol=1e-12)


def test_weights_of_wrong_shape():
    """Test that weights whose shape does not match the readers and the members are rejected."""
    system, _ = _group_of(lambda s: [RateLaw(s.x, total(s.x, weights=np.ones((3, 4))))], size=3)
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
        return [RateLaw(s.x, s.x * (fitness - total(s.x * fitness)))]

    system, x = _group_of(replicator, values=shares, size=4)
    rates = _rates(system, backend)
    np.testing.assert_allclose(rates, x * (payoff @ x - x @ payoff @ x), atol=1e-12)
    assert abs(rates.sum()) < 1e-12


def test_two_groups_in_one_reduction():
    """Test that a reduction reading two groups names both and asks for one reduction per group."""

    class PredationInteraction(BaseInteraction):
        """Predation interaction."""

        prey = role()
        predators = role()
        target = role()

        def laws(self, s):
            """Return the laws of the predation interaction."""
            return [RateLaw(s.target.x, total(s.prey.x * s.predators.x))]

    system = System(
        elements=[
            GroupElement('prey', element=UnitElement, size=3),
            GroupElement('predators', element=UnitElement, size=3),
            UnitElement('target'),
        ],
        interactions=[PredationInteraction('predation', prey='prey', predators='predators', target='target')],
    )
    with pytest.raises(RoleError, match=r'total\(.*predators and prey.*one reduction per group'):
        compile_system(system)


def test_interaction_reads_second_group_through_reductions():
    """Test that an interaction acts on one group and reads another group through a mean."""

    class CouplingInteraction(BaseInteraction):
        """Coupling interaction."""

        target = role()
        source = role()

        def laws(self, s):
            """Return the laws of the coupling interaction."""
            return [RateLaw(s.target.x, mean(s.source.x) - s.target.x)]

    first, second = RNG.normal(size=4), RNG.normal(size=6)
    system = System(
        elements=[
            GroupElement('first', element=UnitElement, size=4, values={'x': first}),
            GroupElement('second', element=UnitElement, size=6, values={'x': second}),
        ],
        interactions=[CouplingInteraction('coupling', target='first', source='second')],
    )
    for backend in ('numpy', 'numba'):
        np.testing.assert_allclose(_rates(system, backend), np.r_[second.mean() - first, np.zeros(6)], atol=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_kuramoto_through_observables(backend):
    """Test that laws reading observables of their own partition compile to one ODE at the current state."""
    system, theta, omega = _kuramoto(40, observables=True)
    observables = [
        ObservableSpec('X', _cosine, group='oscillators'),
        ObservableSpec('Y', _sine, group='oscillators'),
    ]
    problem = compile_system(system, backend=backend, observables=observables)
    assert problem.form.name == 'ODE'
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    np.testing.assert_allclose(rates, _all_to_all(theta, omega), atol=1e-12)


def test_callable_observable_in_same_partition():
    """Test that a law reading a callable observable of its own partition is rejected."""
    system = System(
        elements=[GroupElement('herd', element=UnitElement, size=3), UnitElement('soil')],
        interactions=[ReaderInteraction('reader', target='soil', source='spread')],
    )
    with pytest.raises(ValidationError, match=r'spread.*symbolically.*another'):
        compile_system(system, observables=[ObservableSpec('spread', group='herd', variable='x', reduce=np.ptp)])


def test_observable_read_in_its_own_partition():
    """Test that the value a law reads equals the recorded observable at every frame."""

    class SaverElement(Element):
        """Saver element."""

        W = state(1.0)
        r = parameter(0.1)

        def laws(self, s):
            """Return the laws of the saver element."""
            return [RateLaw(s.W, s.r * s.W)]

    system = System(
        elements=[
            GroupElement(
                'savers',
                element=SaverElement,
                size=5,
                values={'W': RNG.uniform(1, 2, 5), 'r': RNG.uniform(0, 0.2, 5)},
            ),
            UnitElement('account'),
        ],
        interactions=[ReaderInteraction('reader', target='account', source='mean_wealth')],
    )
    observables = [ObservableSpec('mean_wealth', group='savers', variable='W')]
    simulation = Simulation(system, dt=0.5, observables=observables).run(until=5)
    problem, trajectory = simulation.problem_, simulation.trajectory_
    assert not problem.partitions
    index = problem.states.entry('account.x').start
    for row, values in enumerate(trajectory.states):
        rate = problem.drift(0.0, values, problem.initial_parameters)[index]
        assert rate == pytest.approx(trajectory.values('mean_wealth')[row], abs=1e-12)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_synchronous_map(backend):
    """Test that every member of coupled maps updates from the mean of the values at step n."""
    epsilon = 0.3
    system, x = _group_of(
        lambda s: [UpdateLaw(s.x, (1 - epsilon) * 4 * s.x * (1 - s.x) + epsilon * mean(4 * s.x * (1 - s.x)))],
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
            JumpLaw(beta * (1 - s.x) * count(where=s.x > 0.5) / size, {s.x: 1}),
            JumpLaw(gamma * s.x, {s.x: -1}),
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
    assert set(np.unique(trajectory.values('units.x'))) <= {0.0, 1.0}


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

    class TotalInteraction(BaseInteraction):
        """Total interaction."""

        units = role()

        def laws(self, s):
            """Return the laws of the total interaction."""
            return [RateLaw(s.units.x, total(s.units.x))]

    system = System(
        elements=[UnitElement(f'unit_{index}', x=float(index + 1)) for index in range(3)],
        interactions=[TotalInteraction('total', units=['unit_0', 'unit_1', 'unit_2'])],
    )
    problem = compile_system(system)
    assert problem.states.names == ['unit_0.x', 'unit_1.x', 'unit_2.x']
    for backend in ('numpy', 'numba'):
        np.testing.assert_allclose(_rates(system, backend), [6.0, 6.0, 6.0])


def _shared(share, values):
    """Return a system of one group whose interaction gives every member a share of a reduction."""

    class ShareInteraction(BaseInteraction):
        """Share interaction."""

        cells = role()

        def laws(self, s):
            """Return the laws of the share interaction."""
            return [RateLaw(s.cells.x, share(s))]

    return System(
        elements=[GroupElement('units', element=UnitElement, size=len(values), values={'x': values})],
        interactions=[ShareInteraction('share', cells='units')],
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

    class OtherElement(Element):
        """Other element."""

        x = state(0.0)

    class TotalInteraction(BaseInteraction):
        """Total interaction."""

        units = role()

        def laws(self, s):
            """Return the laws of the total interaction."""
            return [RateLaw(s.units.x, total(s.units.x))]

    system = System(
        elements=[UnitElement('a'), OtherElement('b')],
        interactions=[TotalInteraction('total', units=['a', 'b'])],
    )
    with pytest.raises(RoleError, match=r"'units'.*UnitElement and OtherElement"):
        compile_system(system)
