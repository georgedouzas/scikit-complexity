"""Test the census pass on couplings over the neighbours of fields."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from skfem import MeshTri

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, NoiseLaw, ObjectiveLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import laplacian, logistic, neighbors
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace, NetworkSpace, RadiusNeighborhood
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.errors import IncompatibleLawsError, ValidationError
from skcomplex.core.systems import System

NEAREST = RadiusNeighborhood(1, metric='manhattan')


def _rate(field, function=None, *, reduce='sum', backend=NumpyBackend, D=0.0, size=None, space=None):
    """Return the rate of the state `u` of a field at its initial values.

    The rate is `D` times the Laplacian of `u`, plus the reduction of a function of both ends over the neighbours.
    """

    def laws(s):
        u = s.target.u
        rate = s.target.D * laplacian(u) if D else 0 * u
        if function is not None:
            rate = rate + neighbors(
                function(SimpleNamespace(member=s.target, other=s.target.other)),
                over=NEAREST if isinstance(space, LatticeSpace) else None,
                reduce=reduce,
            )
        return [RateLaw(u, rate)]

    element = Element('m', values={'u': field, 'D': D}, size=size, space=space)
    rule = Interaction('c', roles={'target': 'm'}, laws=laws)
    problem = System(elements=[element], interactions=[rule], backend=backend()).prepare()._problem
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    return np.ravel(problem.states.read(drift, 'm.u'))


def _logistic(s):
    """Return a smooth step of the other member's state."""
    return logistic(s.other.u - 0.4, 0, 0.1)


def _neighbor_sum(padded):
    """Return the sum over the four nearest sites of a smooth step of a padded 2D field."""
    step = 1 / (1 + np.exp(-(padded - 0.4) / 0.1))
    return step[:-2, 1:-1] + step[2:, 1:-1] + step[1:-1, :-2] + step[1:-1, 2:]


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_declared_laws_sum_a_function_of_both_ends_over_the_lattice_neighbours(backend):
    """Test that a coupling on a periodic lattice field equals the direct sum over the four nearest sites."""
    field = np.random.default_rng(0).random((6, 5))
    rate = _rate(field, _logistic, backend=backend, space=LatticeSpace((6, 5)))
    expected = _neighbor_sum(np.pad(field, 1, mode='wrap'))
    assert np.abs(rate - np.ravel(expected)).max() < 1e-12


@pytest.mark.skip(reason='Engine gap: a neighbour reduction on a lattice reads no fixed value beyond fixed edges.')
def test_declared_laws_read_the_fixed_value_beyond_fixed_edges():
    """Test that a coupling on a field with a fixed boundary reads the fixed value beyond the edges."""
    field = np.random.default_rng(1).random((6, 5))
    space = LatticeSpace((6, 5), boundary={'u': ('fixed', 0.5)})
    expected = _neighbor_sum(np.pad(field, 1, constant_values=0.5))
    assert np.abs(_rate(field, _logistic, space=space) - np.ravel(expected)).max() < 1e-12


@pytest.mark.skip(reason='Engine gap: a neighbour reduction on a lattice reads no fixed value beyond fixed edges.')
def test_declared_laws_agree_with_the_edge_kernel_away_from_fixed_edges():
    """Test that a coupling on a field with fixed edges matches the edge kernel of a group at the interior sites."""
    field = np.random.default_rng(2).random((7, 6))
    lowered = _rate(field, _logistic, space=LatticeSpace((7, 6), boundary={'u': ('fixed', 0.0)}))
    edges = _rate(np.ravel(field), _logistic, size=42, space=LatticeSpace((7, 6)))
    difference = np.reshape(lowered - edges, (7, 6))
    assert np.abs(difference[1:-1, 1:-1]).max() < 1e-12
    assert np.abs(difference).max() > 1e-3


def test_declared_laws_lower_a_linear_exchange_on_a_network():
    """Test that a coupling on a network field sums `g (x_j - x_i) + xi x_j` as `g L x + xi A x`."""
    graph = nx.connected_watts_strogatz_graph(20, 4, 0.3, seed=0)
    field = np.random.default_rng(3).random(20)
    rate = _rate(field, lambda s: 0.3 * (s.other.u - s.member.u) + 0.05 * s.other.u, space=NetworkSpace(graph))
    adjacency = nx.to_numpy_array(graph)
    laplacian_matrix = adjacency - np.diag(adjacency.sum(axis=1))
    expected = 0.3 * laplacian_matrix @ field + 0.05 * adjacency @ field
    assert np.abs(rate - expected).max() < 1e-12


@pytest.mark.skip(
    reason='Engine gap: a diffusive neighbour reduction on a spectral lattice is not lowered to its Fourier Laplacian.',
)
def test_declared_laws_lower_a_diffusive_coupling_to_the_spectral_laplacian():
    """Test that a coupling `D (x_j - x_i)` on a spectral field is `D` times the field's Fourier Laplacian."""
    positions = np.arange(32) * 2 * np.pi / 32
    field = np.sin(3 * positions)
    space = LatticeSpace(32, spacing=2 * np.pi / 32, discretization='spectral')
    rate = _rate(field, lambda s: 0.7 * (s.other.u - s.member.u), space=space)
    expected = -0.7 * (2 * np.pi / 32) ** 2 * 9 * field
    assert np.abs(rate - expected).max() < 1e-12


@pytest.mark.skip(
    reason='Engine gap: a diffusive neighbour reduction on a mesh is not lowered to the finite element Laplacian.',
)
def test_declared_laws_lower_a_diffusive_coupling_to_the_finite_element_laplacian():
    """Test that a coupling `D (x_j - x_i)` on a mesh field is `D` times the finite element Laplacian."""
    mesh = MeshTri.init_sqsymmetric().refined(2)
    field = np.sin(np.pi * mesh.p[0]) * mesh.p[1]
    space = CellComplexSpace(points=mesh.p, cells=mesh.t)
    coupled = _rate(field, lambda s: 0.7 * (s.other.u - s.member.u), space=space)
    assert np.abs(coupled - _rate(field, D=0.7, space=space)).max() < 1e-12


def test_declared_laws_divide_the_mean_by_the_lattice_neighbours():
    """Test that the mean of a coupling on a periodic lattice field divides the sum by the four nearest sites."""
    field = np.random.default_rng(4).random((5, 5))
    expected = _neighbor_sum(np.pad(field, 1, mode='wrap')) / 4
    assert np.abs(_rate(field, _logistic, reduce='mean', space=LatticeSpace((5, 5))) - np.ravel(expected)).max() < 1e-12


def _get_plant(laws, **values):
    """Return the prepared system of a plant whose program a rule sets."""
    plant = Element('plant', values={'v': 0.0, 'w': 0.0, **values})
    return System(elements=[plant], interactions=[Interaction('lp', roles={'target': 'plant'}, laws=laws)]).prepare()


def _get_dispatch(s):
    """Return the dispatch of two plants of a capacity at the least cost, the greatest output."""
    v, w = s.target.v, s.target.w
    bounds = [ConstraintLaw(v >= 0), ConstraintLaw(w >= 0), ConstraintLaw(v <= 3)]
    return [ObjectiveLaw(-(2 * v + w), [v, w]), ConstraintLaw(v + w <= s.target.cap), *bounds]


def test_linear_objective_with_its_constraints_is_solved_as_a_linear_program():
    """Test that a linear objective with linear inequalities sets its targets to the optimum of a linear program."""
    system = _get_plant(_get_dispatch, cap=5.0)
    trajectory = system.simulate(until=0.0)
    assert [float(trajectory.get_values(name)[0]) for name in ('plant.v', 'plant.w')] == [3.0, 2.0]
    assert float(trajectory.get_values('plant.objective_0')[0]) == -8.0
    assert system.parameters_ == ('plant.cap',)


def test_linear_program_records_the_duals_of_its_constraints():
    """Test that `duals=True` records the change of the optimum with the right side of each constraint."""

    def laws(s):
        v, w = s.target.v, s.target.w
        bounds = [ConstraintLaw(v >= 0), ConstraintLaw(w >= 0), ConstraintLaw(v <= 3)]
        return [ObjectiveLaw(-(2 * v + w), [v, w], duals=True), ConstraintLaw(v + w <= s.target.cap), *bounds]

    trajectory = _get_plant(laws, cap=5.0).simulate(until=0.0)
    duals = [float(trajectory.get_values(f'plant.dual_{index}')[0]) for index in range(4)]
    assert duals == pytest.approx([-1.0, 0.0, 0.0, -1.0])


def test_linear_program_is_solved_at_every_time_of_a_run():
    """Test that a program whose capacity grows in time sets its targets at every time."""
    plant = Element('plant', values={'v': 0.0, 'w': 0.0, 'cap': 1.0})
    grow = Interaction('grow', roles={'target': 'plant'}, laws=lambda s: [RateLaw(s.target.cap.diff(s.t), 1.0)])
    program = Interaction('lp', roles={'target': 'plant'}, laws=_get_dispatch)
    trajectory = System(elements=[plant], interactions=[program, grow]).prepare().simulate(until=2.0, dt=1.0)
    assert trajectory.get_values('plant.v').tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert trajectory.get_values('plant.w').tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_sequence_of_objectives_is_minimised_in_order():
    """Test that the second objective is minimised while the first one keeps its optimum."""

    def laws(s):
        v, w = s.target.v, s.target.w
        bounds = [ConstraintLaw(v >= 0), ConstraintLaw(w >= 0), ConstraintLaw(w <= 3), ConstraintLaw(v + w <= 4)]
        return [ObjectiveLaw([-(v + w), abs(v)], [v, w]), *bounds]

    trajectory = _get_plant(laws).simulate(until=0.0)
    assert [float(trajectory.get_values(name)[0]) for name in ('plant.v', 'plant.w')] == pytest.approx([1.0, 3.0])
    assert float(trajectory.get_values('plant.objective_1')[0]) == pytest.approx(1.0)


def test_objective_is_sought_only_where_it_is_active():
    """Test that an inactive program sets its targets to zero, and an infeasible one to zero with the policy."""

    def laws(s):
        v = s.target.v
        active = s.target.switch > 0
        return [ObjectiveLaw(-v, [v], active=active, infeasible='zero'), ConstraintLaw(v <= 2), ConstraintLaw(v >= 1)]

    assert float(_get_plant(laws, switch=1.0).simulate(until=0.0).get_values('plant.v')[0]) == 2.0
    assert float(_get_plant(laws, switch=0.0).simulate(until=0.0).get_values('plant.v')[0]) == 0.0

    def infeasible(s):
        v = s.target.v
        return [ObjectiveLaw(-v, [v], infeasible='zero'), ConstraintLaw(v <= 1), ConstraintLaw(v >= 2)]

    assert float(_get_plant(infeasible).simulate(until=0.0).get_values('plant.v')[0]) == 0.0


def test_inequality_of_another_interaction_joins_the_program_of_its_target():
    """Test that a bound set by another interaction on a target of a program constrains the program."""
    plant = Element('plant', values={'v': 0.0, 'w': 0.0, 'cap': 5.0})
    program = Interaction('lp', roles={'target': 'plant'}, laws=_get_dispatch)
    limit = Interaction(
        'limit',
        roles={'target': 'plant'},
        laws=lambda s: [ConstraintLaw(s.target.v <= 1, target=s.target.v)],
    )
    trajectory = System(elements=[plant], interactions=[program, limit]).prepare().simulate(until=0.0)
    assert [float(trajectory.get_values(name)[0]) for name in ('plant.v', 'plant.w')] == [1.0, 4.0]


def test_inequality_outside_a_program_names_its_target():
    """Test that an inequality of an interaction without an objective and without a target is refused."""
    limit = Interaction('limit', roles={'target': 'plant'}, laws=lambda s: [ConstraintLaw(s.target.v <= 1)])
    with pytest.raises(ValidationError, match='target'):
        System(elements=[Element('plant', values={'v': 0.0})], interactions=[limit]).prepare()


def test_nonlinear_objective_stays_a_minimum():
    """Test that an objective that is not linear in its targets is minimised, not solved as a linear program."""
    system = _get_plant(lambda s: [ObjectiveLaw((s.target.v - 2) ** 2, s.target.v)])
    assert float(system.simulate(until=0.0).get_values('plant.v')[0]) == pytest.approx(2.0, abs=1e-6)


def test_nonlinear_sequence_of_objectives_is_refused():
    """Test that a sequence of objectives that is not linear in its targets raises an error."""
    with pytest.raises(ValidationError, match='sequence of values'):
        _get_plant(lambda s: [ObjectiveLaw([s.target.v**2, s.target.w], [s.target.v, s.target.w])])


def _reflected(s):
    """Return a drift and noise of a walker with a wall at zero that reflects it."""
    x = s.target.x
    return [RateLaw(x.diff(s.t), -0.5), NoiseLaw(x, 1.0), UpdateLaw(x, abs(x))]


def test_update_law_beside_rate_laws_corrects_the_state_after_each_step():
    """Test that `UpdateLaw(x, abs(x))` beside rate laws reflects a walker at a wall after each step."""
    walker = Element('walker', values={'x': 0.1})
    wall = Interaction('wall', roles={'target': 'walker'}, laws=_reflected)
    system = System(elements=[walker], interactions=[wall]).prepare()
    positions = np.ravel(system.simulate(until=20.0, random_state=1).get_values('walker.x'))
    assert system.form_ == 'stochastic differential equations'
    assert positions.min() >= 0.0


@pytest.mark.parametrize(
    'update',
    [
        lambda x: UpdateLaw(x, 1.0, increment=True),
        lambda x: UpdateLaw(x, x + 1.0, order=0),
    ],
)
def test_update_law_that_steps_a_state_rates_change_is_rejected(update):
    """Test that a step of a state that a rate also changes, which is no correction, is a conflict of law kinds."""
    clash = Interaction(
        'clash',
        roles={'target': 'node'},
        laws=lambda s: [RateLaw(s.target.x, 1.0), update(s.target.x)],
    )
    system = System(elements=[Element('node', values={'x': 0.0})], interactions=[clash])
    with pytest.raises(IncompatibleLawsError, match=r'RateLaw: node; UpdateLaw: node.*\[laws-mixed\]'):
        system.prepare()


def test_update_law_of_a_discrete_child_stays_a_map():
    """Test that an update law of a nested system that steps stays a map beside a continuous parent."""
    decay = Interaction(
        'decay',
        roles={'target': 'tank'},
        laws=lambda s: [RateLaw(s.target.v.diff(s.t), -0.1 * s.target.v)],
    )
    double = Interaction('double', roles={'target': 'count'}, laws=lambda s: [UpdateLaw(s.target.n, 2 * s.target.n)])
    child = System('counter', elements=[Element('count', values={'n': 1.0})], interactions=[double])
    system = System(elements=[Element('tank', values={'v': 1.0})], interactions=[decay], systems=[child]).prepare()
    counts = np.ravel(system.simulate(until=3.0).get_values('counter/count.n'))
    assert counts[-1] == 8.0
