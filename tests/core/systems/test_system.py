"""Test the `_system` module of `skcomplex.core.systems`."""

import re
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import (
    ConstraintLaw,
    EventLaw,
    JumpLaw,
    NoiseLaw,
    ObjectiveLaw,
    RateLaw,
    TotalisticSpec,
    UpdateLaw,
)
from skcomplex.core.components.laws.utils import count, mean, neighbors, series, total
from skcomplex.core.components.spaces import BaseSpace, ContinuousSpace, LatticeSpace, NetworkSpace, RadiusNeighborhood
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.solvers import (
    EulerMaruyamaSolver,
    GillespieSolver,
    GradientFlowSolver,
    IterationSolver,
    MetropolisSolver,
    MinimizeSolver,
    RK4Solver,
    ScipyODESolver,
)
from skcomplex.core.errors import LabelError, SolverError, TagError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.general.reactions import ReactionInteraction


def meadow_system(solver: RK4Solver | str = 'auto') -> System:
    """Return a system of one nested Lotka-Volterra meadow."""
    return System('ecology', systems=[LotkaVolterraSystem(label='meadow')], solver=solver)


def test_get_params_lists_nested_names() -> None:
    """The parameters of a nested component appear under its label."""
    params = meadow_system().get_params()
    assert params['meadow__r'] == pytest.approx(1.1)
    assert params['solver'] == 'auto'


def test_set_params_tunes_component() -> None:
    """A nested name sets the parameter of the component with that label."""
    system = meadow_system().set_params(meadow__r=1.5)
    assert system.systems[0].r == pytest.approx(1.5)
    assert system.get_params()['meadow__r'] == pytest.approx(1.5)


def test_set_params_unknown_component_raises() -> None:
    """A nested name that matches no component raises an error that lists the components."""
    with pytest.raises(ValidationError, match=r"no component labelled 'forest'.*'meadow'"):
        meadow_system().set_params(forest__r=1.0)


def test_set_params_unknown_parameter_raises() -> None:
    """A nested name whose parameter the component lacks raises an error that names both."""
    with pytest.raises(ValidationError, match=r"'meadow'.*no parameter 'mass'"):
        meadow_system().set_params(meadow__mass=2.0)


def test_set_params_unknown_own_parameter_raises() -> None:
    """A plain name that is not a parameter of the system raises an error."""
    with pytest.raises(ValidationError, match="no parameter 'time'"):
        meadow_system().set_params(time=1.0)


def test_clone_keeps_components() -> None:
    """Cloning a system keeps its label, components, and settings."""
    copy = clone(meadow_system(solver=RK4Solver()))
    assert copy.label == 'ecology'
    assert isinstance(copy.solver, RK4Solver)
    assert copy.get_params()['meadow__prey'] == pytest.approx(40.0)


def test_prepare_compiles_once_for_many_runs() -> None:
    """Two runs of one prepared system keep what prepare learned."""
    system = meadow_system().prepare()
    equations = system.equations_
    system.simulate(until=1.0)
    system.simulate(until=2.0)
    assert system.equations_ is equations


def test_prepare_sets_learned_attributes() -> None:
    """A prepared system names its states and parameters and shows its equations and form."""
    system = meadow_system().prepare()
    assert system.states_ == ('meadow/prey.x', 'meadow/predators.x')
    assert 'meadow/growth.r' in system.parameters_
    predators = 'meadow/predation.b*meadow/predators.x*meadow/prey.x - meadow/death.d*meadow/predators.x'
    assert system.equations_['meadow/predators.x'] == predators
    assert system.form_ == 'ordinary differential equations'
    assert type(system.backend_).__name__ == 'NumpyBackend'


def test_prepare_names_nested_states_by_path() -> None:
    """The states of a nested system are named by their path from the system that holds it."""
    parent = System('world', systems=[meadow_system()]).prepare()
    assert parent.states_ == ('ecology/meadow/prey.x', 'ecology/meadow/predators.x')


def test_prepare_uses_named_solver() -> None:
    """A solver given to the system runs the problem."""
    system = meadow_system(solver=RK4Solver()).prepare()
    assert isinstance(system.solver_, RK4Solver)
    assert system.simulate(until=1.0, dt=0.01).times[-1] == pytest.approx(1.0)


def test_prepare_compiles_for_the_named_backend() -> None:
    """A backend given to the system is the one its problem is compiled for, not only the one that runs it."""
    system = System('ecology', systems=[LotkaVolterraSystem(label='meadow')], backend=NumbaBackend()).prepare()
    assert isinstance(system.backend_, NumbaBackend)
    assert system._problem.backend == 'numba'


def test_prepare_rejects_bad_solver() -> None:
    """A solver that is neither 'auto' nor a solver raises an error."""
    with pytest.raises(ValidationError, match="solver of the system 'ecology' must be 'auto'"):
        meadow_system(solver='fast').prepare()


def test_nested_system_runs_in_parent() -> None:
    """A system nested in another one runs inside it."""
    parent = System('world', systems=[meadow_system()]).prepare()
    assert 'ecology/meadow/prey.x' in parent.simulate(until=1.0).names


class _Consensus(Interaction):
    """Interaction that moves every member toward its neighbours, or toward the mean of its group."""

    def __init__(self, label: str, *, target: Role = 'nodes', over: str = 'neighbours') -> None:
        super().__init__(label)
        self.target = target
        self.over = over

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the pull toward the neighbours or toward the mean."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            pull = neighbors(context.target.other.x - x) if self.over == 'neighbours' else mean(x) - x
            self.laws_: list[BaseComponent] = [RateLaw(x.diff(context.t), pull)]


def test_prepare_writes_reductions_readably():
    """Test that sums over neighbours and means over a group appear in the equations by the states they read."""
    nodes = Element('nodes', values={'x': [1.0, 0.0, 0.0]}, space=NetworkSpace(nx.path_graph(3)))
    crowd = Element('crowd', values={'x': [1.0, 2.0, 6.0]}, size=3)
    system = System(
        elements=[nodes, crowd],
        interactions=[_Consensus('consensus'), _Consensus('pull', target='crowd', over='group')],
    ).prepare()
    assert system.equations_['nodes.x[i]'] == "neighbors(-nodes.x[i] + nodes.x[j], reduce='sum')"
    assert system.equations_['crowd.x[i]'] == 'mean(crowd.x) - crowd.x[i]'
    assert not any(re.search(r'(?<![\w.])_\w', equation) for equation in system.equations_.values())


def _keep(center, *around):
    """Return every cell unchanged."""
    return center


class _Rules(Interaction):
    """Interaction with a named local rule and a lookup local rule of the same cells."""

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set a named function rule, then a totalistic rule, of the cells."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            alive = context.target.alive
            around = RadiusNeighborhood(1, center=True)
            self.laws_: list[BaseComponent] = [
                UpdateLaw(alive, _keep, neighborhood=around),
                UpdateLaw(alive, TotalisticSpec({1: 1}), neighborhood=around, label='parity'),
            ]


def test_prepare_writes_local_rules_by_their_names():
    """Test that local rules appear in the equations by name and the neighbours they read, one after the other."""
    cells = Element('cells', values={'alive': [0.0, 1.0, 0.0]}, space=LatticeSpace(3))
    system = System(elements=[cells], interactions=[_Rules('rules')]).prepare()
    reads = "neighbors(cells.alive, reduce='each')"
    assert system.equations_ == {'cells.alive[i]': f'keep({reads}), then parity({reads})'}


class _Process(Interaction):
    """Interaction whose laws make one kind of problem."""

    def __init__(self, label: str, *, target: Role = 'p', kind: str = 'ode') -> None:
        super().__init__(label)
        self.target = target
        self.kind = kind

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the laws of the kind."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            if self.kind == 'dae':
                y = context.target.y
                self.laws_: list[BaseComponent] = [RateLaw(x.diff(t), -x + y), ConstraintLaw(y - 2 * x, 0, target=y)]
                return
            laws: dict[str, list[BaseComponent]] = {
                'ode': [RateLaw(x.diff(t), -x)],
                'sde': [RateLaw(x.diff(t), -x), NoiseLaw(x, 0.1)],
                'jump': [JumpLaw(x, {x: -1})],
                'map': [UpdateLaw(x, x / 2)],
                'hybrid': [RateLaw(x.diff(t), -1), EventLaw(x, {x: 1})],
                'objective': [ObjectiveLaw((x - 3) ** 2, x)],
            }
            self.laws_ = laws[self.kind]


def _process(kind: str, solver: object = 'auto') -> System:
    """Return a system of one element whose laws make one kind of problem."""
    values = {'x': 1.0, 'y': 0.0} if kind == 'dae' else {'x': 10.0 if kind == 'jump' else 1.0}
    return System(elements=[Element('p', values=values)], interactions=[_Process('law', kind=kind)], solver=solver)


@pytest.mark.parametrize(
    ('kind', 'solver', 'form', 'default'),
    [
        ('ode', 'auto', 'ordinary differential equations', ScipyODESolver),
        ('sde', 'auto', 'stochastic differential equations', EulerMaruyamaSolver),
        ('jump', 'auto', 'jump process', GillespieSolver),
        ('dae', 'auto', 'differential-algebraic equations', ScipyODESolver),
        ('map', 'auto', 'discrete map', IterationSolver),
        ('hybrid', 'auto', 'ordinary differential equations with events', ScipyODESolver),
        ('objective', 'auto', 'static solution', MinimizeSolver),
        ('objective', MetropolisSolver(), 'static solution', MetropolisSolver),
        ('objective', GradientFlowSolver(), 'static solution', GradientFlowSolver),
    ],
)
def test_prepare_infers_the_problem_and_its_solver(kind, solver, form, default):
    """Test that the laws decide the problem, its default solver, and the solvers a user may name instead."""
    system = _process(kind, solver).prepare()
    assert system.form_ == form
    assert isinstance(system.solver_, default)
    step = 1.0 if kind == 'map' else 0.5
    end = system.simulate(until=1.0, dt=step, random_state=0).times[-1]
    assert end == pytest.approx(0.0 if default is MinimizeSolver else 1.0)


def test_prepare_names_the_solvers_of_a_map_given_an_ode_solver():
    """Test that a map prepared with an ODE solver raises an error naming the map and the solvers that run it."""
    with pytest.raises(
        SolverError,
        match=r'does not support the laws UpdateLaw.*a discrete map, which .*IterationSolver',
    ):
        _process('map', RK4Solver()).prepare()


class _Conversion(Interaction):
    """Interaction that converts units of `a` into `b` one at a time at a rate."""

    def __init__(self, label: str, *, a: Role = 'A', b: Role = 'B', k: float = 1.0) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the transfer of one unit at the rate `k a`."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [JumpLaw(context.k * context.a.n, {context.a.n: -1, context.b.n: 1})]


def test_one_reaction_runs_as_jumps_or_as_its_mean_with_two_solvers():
    """Test that one conversion runs as a jump process by default and as its mean `100 exp(-t)` with an ODE solver."""
    elements = [Element('A', values={'n': 100.0}), Element('B', values={'n': 0.0})]
    jumps = System(elements=elements, interactions=[_Conversion('convert')]).prepare()
    mean = clone(jumps).set_params(solver=ScipyODESolver()).prepare()
    assert (jumps.form_, mean.form_) == ('jump process', 'ordinary differential equations')
    assert float(jumps.simulate(until=1.0, random_state=0).get_values('A.n')[-1]).is_integer()
    assert mean.simulate(until=1.0).get_values('A.n')[-1] == pytest.approx(100 * np.exp(-1), rel=1e-4)


class _Hold(Interaction):
    """Controller that sets its input once per step to the negative half of the plant."""

    def __init__(self, label: str, *, control: Role = 'input', plant: Role = 'plant') -> None:
        super().__init__(label)
        self.control = control
        self.plant = plant

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the next input."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_: list[BaseComponent] = [UpdateLaw(context.control.u, -context.plant.x / 2)]


class _Plant(Interaction):
    """Plant that relaxes towards its input, `x' = -x + u`."""

    def __init__(self, label: str, *, plant: Role = 'plant', control: Role = 'input') -> None:
        super().__init__(label)
        self.plant = plant
        self.control = control

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the plant."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.plant.x, context.t
            self.laws_: list[BaseComponent] = [RateLaw(x.diff(t), -x + context.control.u)]


def test_nested_systems_keep_the_time_their_laws_imply() -> None:
    """A discrete controller steps once per unit of time next to a continuous plant that holds its input."""
    system = System(
        systems=[
            System('control', elements=[Element('input', values={'u': 0.0})]),
            System('house', elements=[Element('plant', values={'x': 1.0})]),
        ],
        interactions=[
            _Hold('policy', control='control/input', plant='house/plant'),
            _Plant('rate', plant='house/plant', control='control/input'),
        ],
    ).prepare()
    trajectory = system.simulate(until=2.0, dt=1.0)
    first = 1.5 * np.exp(-1) - 0.5
    assert system.form_ == 'co-simulation'
    assert trajectory.get_values('control/input.u') == pytest.approx([0.0, -0.5, -first / 2], rel=1e-4)
    assert trajectory.get_values('house/plant.x')[1] == pytest.approx(first, rel=1e-4)


def test_system_without_laws_is_static() -> None:
    """A system whose values no law changes reports a static problem."""
    system = System(elements=[Element('ball', values={'x': 1.0, 'm': 2.0})]).prepare()
    assert (system.form_, system.states_, system.parameters_) == ('static solution', (), ('ball.x', 'ball.m'))


class _Balance(Interaction):
    """Interaction that sets the price where the supply `p` meets the demand `d - p`."""

    def __init__(self, label: str, *, target: Role = 'market', d: float = 4.0) -> None:
        super().__init__(label)
        self.target = target
        self.d = d

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the balance of supply and demand as the law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            price = context.target.p
            self.laws_ = [ConstraintLaw(price == context.d - price, target=price)]


def test_system_of_constraints_alone_is_static() -> None:
    """A system whose laws are all constraints is solved once, as a static problem."""
    system = System(elements=[Element('market', values={'p': 0.0})], interactions=[_Balance('balance')]).prepare()
    assert system.form_ == 'static solution'
    assert system.simulate(until=1.0).get_values('market.p')[-1] == pytest.approx(2.0)


class _Attraction(Interaction):
    """Interaction whose members move towards the mean of all the others of the group."""

    def __init__(self, label: str, *, target: Role = 'crowd') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the pull of every member towards the others."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_: list[BaseComponent] = [RateLaw(x.diff(context.t), total(context.target.other.x - x) / count())]


class _Majority(Interaction):
    """Interaction that pulls every member of a hyperedge towards the mean of its members."""

    def __init__(self, label: str, *, among: Role = 'nodes', along: bool | str | None = 'hyperedges') -> None:
        super().__init__(label, among=among, along=along)

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the pull towards the mean of the members of the hyperedge."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [RateLaw(s.member.x.diff(s.t), 2.0 * (mean(s.members.x, over=s.members) - s.member.x))]


def test_prepare_names_the_other_members_a_law_reads():
    """Test that equations name the other member of a group as `[j]` and those of a hyperedge as `[j0]`, `[j1]`."""
    crowd = Element('crowd', values={'x': [1.0, 2.0, 6.0]}, size=3)
    space = NetworkSpace(nx.empty_graph(4), hyperedges=[(0, 1, 2)])
    nodes = Element('nodes', values={'x': np.array([3.0, 0.0, 0.0, 5.0])}, space=space)
    system = System(elements=[crowd, nodes], interactions=[_Attraction('attraction'), _Majority('majority')]).prepare()
    assert system.equations_['crowd.x[i]'] == '-(sum(1)*crowd.x[i] - sum(crowd.x[j]))/count(1)'
    assert 'nodes.x[i]' in system.equations_['nodes.x[i]']
    assert 'nodes.x[j0]' in system.equations_['nodes.x[i]']
    assert not any('__' in equation for equation in system.equations_.values())


def test_prepare_reduces_over_the_members_of_a_hyperedge():
    """Test that `mean(..., over=s.members)` on a hyperedge pulls its three members to their mean, and others not."""
    space = NetworkSpace(nx.empty_graph(4), hyperedges=[(0, 1, 2)])
    nodes = Element('nodes', values={'x': np.array([3.0, 0.0, 0.0, 5.0])}, space=space)
    system = System(elements=[nodes], interactions=[_Majority('majority')]).prepare()
    last = system.simulate(until=0.5, dt=0.01).get_values('nodes.x')[-1]
    assert last == pytest.approx([1.0 + 2.0 * np.exp(-1.0), 1.0 - np.exp(-1.0), 1.0 - np.exp(-1.0), 5.0], abs=1e-5)


class _Lookup(Interaction):
    """Interaction whose rate is a lookup table of the state, as in system dynamics."""

    def __init__(self, label: str, *, target: Role = 'stock') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set a rate that a table of breakpoints between whole numbers gives."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_ = [RateLaw(x.diff(context.t), -series(x, (0.0, 0.25, 0.5, 1.0), (0.0, 1.0, 1.5, 2.0)))]


def test_prepare_keeps_the_breakpoints_of_a_lookup_table():
    """Test that a table read through an interaction keeps its breakpoints, so the state falls at the rate 2 above 1."""
    system = System(elements=[Element('stock', values={'x': 2.0})], interactions=[_Lookup('lookup')]).prepare()
    assert '< 0.25' in system.equations_['stock.x']
    assert system.simulate(until=0.5, dt=0.5).get_values('stock.x')[-1] == pytest.approx(1.0)


def _growth(label: str, target: str) -> Interaction:
    """Return an interaction of exponential growth of the value `n` of its target."""
    return Interaction(label, roles={'target': target}, laws=lambda s: [RateLaw(s.target.n.diff(s.t), s.target.n)])


def test_prepare_checks_every_component():
    """Test that preparing a system checks its components, which reject their invalid labels."""
    with pytest.raises(LabelError, match='Label 1'):
        System(elements=[Element(1, values={'n': 1.0})]).prepare()


def test_prepare_reports_two_children_of_one_label():
    """Test that two direct children with one label raise an error that names the label."""
    elements = [Element('a', values={'n': 1.0}), Element('a', values={'n': 1.0})]
    with pytest.raises(LabelError, match="'a' is used by more than one child"):
        System('farm', elements=elements).prepare()


def test_elements_on_edges_take_the_network_of_their_system():
    """Test that an element on the edges of the network of its system has one member per edge."""
    weights = Element('weights', values={'n': 1.0}, on='edges')
    system = System(
        elements=[weights], interactions=[_growth('growth', 'weights')], space=NetworkSpace(nx.cycle_graph(30)),
    )
    assert system.prepare().simulate(until=0).get_values('weights.n').shape == (1, 30)


def test_interactions_couple_the_elements_of_nested_systems_by_path():
    """Test that an interaction couples the elements of a nested system by their path."""
    meadow = System('meadow', elements=[Element('rabbits', values={'n': 2.0})])
    system = System(systems=[meadow], interactions=[_growth('growth', 'meadow/rabbits')]).prepare()
    assert system.simulate(until=1, dt=0.5).get_values('meadow/rabbits.n')[-1] == pytest.approx(2.0 * np.e, rel=1e-4)


def test_periodic_space_wraps_positions_and_keeps_velocities():
    """Test that a point crossing the side of a periodic space re-enters on the opposite side with its velocity."""
    space = ContinuousSpace(extent=(1.0, 1.0), boundary='periodic')
    point = Element('p', values={'x': 0.9, 'y': 0.5, "x'": 1.0, "y'": 0.0})
    trajectory = System(elements=[point], space=space).prepare().simulate(until=0.2, dt=0.2)
    assert trajectory.get_values('p.x')[-1] == pytest.approx(0.1)
    gas = Element('gas', values={'x': [0.95], 'y': [0.5], "x'": [1.0], "y'": [0.0]}, size=1)
    trajectory = System(elements=[gas], space=space).prepare().simulate(until=0.2, dt=0.1)
    assert trajectory.get_values('gas.x')[-1] == pytest.approx([0.15])
    assert trajectory.get_values("gas.x'")[-1] == pytest.approx([1.0])


def test_tags_follow_the_laws_of_the_components():
    """Test that a system of a rate and a noise interaction is stochastic, with both law kinds."""
    noisy = ReactionInteraction('growth', target='stock', products={'n': 1}, mode='langevin')
    tags = System(elements=[Element('stock', values={'n': 1.0})], interactions=[noisy]).get_tags()
    assert (tags.laws, tags.randomness, tags.time, tags.space, tags.domains) == (
        {'rate', 'noise'},
        'stochastic',
        {'continuous'},
        {'none'},
        {'real'},
    )


def test_tags_reject_elements_on_spaces_that_nothing_joins():
    """Test that elements on a lattice and on a network, in a system without a space, disagree on the space."""
    cells = Element('cells', values={'n': 1.0}, space=LatticeSpace((3,)))
    nodes = Element('nodes', values={'n': 1.0}, space=NetworkSpace(nx.cycle_graph(3)))
    with pytest.raises(TagError, match=r"The components 'cells' and 'nodes' disagree on the tag 'space'"):
        System(elements=[cells, nodes]).get_tags()
    assert System(elements=[cells, nodes], space=ContinuousSpace(n_dim=2)).get_tags().space == {
        'continuous',
        'lattice',
        'network',
    }
