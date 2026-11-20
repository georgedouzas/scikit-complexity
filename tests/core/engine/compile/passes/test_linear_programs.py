"""Test the `_linear_programs` pass of `skcomplex.core.engine.compile.passes`."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, ObjectiveLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import either
from skcomplex.core.components.spaces import BaseSpace, LatticeSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems import RunStateSpec
from skcomplex.core.engine.solvers import CoSimulationSolver, RK4Solver
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import InfeasibleError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role

TOY_NAMES = ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'b1', 'b2', 'vbio')
TOY_REVERSIBLE = ('v4', 'v6', 'b2')


def _constraints(*relations):
    """Return the constraint laws of relations."""
    return [ConstraintLaw(relation) for relation in relations]


class ToyNetworkInteraction(Interaction):
    """Toy network of Table 8.2 of Raman (2021), with the growth objective and optionally the total flux after it."""

    def __init__(self, label, *, target: Role = 'net', total: bool = False, duals: bool = False):
        super().__init__(label)
        self.target = target
        self.total = total
        self.duals = duals

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            balances = [
                s.b1 - s.v1 - s.v2 == 0,
                s.v1 + s.v3 - s.v4 - s.v5 == 0,
                s.v2 - s.v3 == 0,
                2 * s.v4 - s.v6 - s.v7 == 0,
                s.v6 - s.b2 == 0,
                s.v5 - s.vbio == 0,
                s.v7 - 2 * s.vbio == 0,
                s.v1 + s.v3 - s.v7 - 10 * s.vbio == 0,
            ]
            variables = [getattr(s, name) for name in TOY_NAMES]
            bounds = [item <= 1000 for item in variables]
            bounds += [getattr(s, name) >= (-1000 if name in TOY_REVERSIBLE else 0) for name in TOY_NAMES]
            objectives = [-s.vbio]
            if self.total:
                objectives.append(sum(abs(item) for item in variables))
            self.laws_ = [*_constraints(*balances, *bounds), ObjectiveLaw(objectives, variables, duals=self.duals)]


def _toy_network(total=False, duals=False):
    """Return the system of the toy network of Raman."""
    return System(
        elements=[Element('net', values=dict.fromkeys(TOY_NAMES, 0.0))],
        interactions=[ToyNetworkInteraction('network', total=total, duals=duals)],
    )


class FlowsElement(Element):
    """Two flows with the capacity of their sum, and switches that gate the first."""

    def __init__(self, label, *, v=0.0, w=0.0, capacity=10.0, g1=1.0, g2=1.0):
        super().__init__(label)
        self.v, self.w = v, w
        self.capacity, self.g1, self.g2 = capacity, g1, g2


class FlowInteraction(Interaction):
    """Program of the two flows, the greatest first flow under the capacity of their sum."""

    def __init__(self, label, *, target: Role = 'flows', infeasible: str = 'raise', minimum: float = 0.0):
        super().__init__(label)
        self.target = target
        self.infeasible = infeasible
        self.minimum = minimum

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            constraints = _constraints(
                s.v + s.w <= s.capacity,
                s.w >= self.minimum,
                s.v >= 0,
                s.v <= 1000 * either(s.g1 > 0.5, s.g2 > 0.5),
            )
            self.laws_ = [*constraints, ObjectiveLaw(-s.v, [s.v, s.w], infeasible=self.infeasible)]


def _flows(flows=None, *interactions, infeasible='raise', minimum=0.0):
    """Return the system of two flows and their program."""
    return System(
        elements=[flows or FlowsElement('flows')],
        interactions=[FlowInteraction('flow', infeasible=infeasible, minimum=minimum), *interactions],
    )


class UpperBoundInteraction(Interaction):
    """Upper bound of one variable, which merges into the program of the variable."""

    def __init__(self, label, *, target: Role = 'flows', bound: float = 8.0):
        super().__init__(label)
        self.target = target
        self.bound = bound

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            v = context.target.v
            self.laws_ = [ConstraintLaw(v <= context.bound, target=v)]


class CellInteraction(Interaction):
    """Growth and uptake of a cell set by a program, the yield fixed, and the growth of its biomass."""

    def __init__(self, label, *, target: Role = 'cell', active: bool = False, infeasible: str = 'zero'):
        super().__init__(label)
        self.target = target
        self.active = active
        self.infeasible = infeasible

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            self.laws_ = [
                *_constraints(s.growth == 0.1 * s.uptake, s.uptake >= 0),
                ObjectiveLaw(
                    -s.growth,
                    [s.uptake, s.growth],
                    infeasible=self.infeasible,
                    active=s.biomass > 0.5 if self.active else None,
                ),
                RateLaw(s.biomass, s.growth * s.biomass),
            ]


def _cell(label='cell', biomass=1.0, space=None):
    """Return a cell, or a cell at every site of a space."""
    return Element(label, values={'uptake': 0.0, 'growth': 0.0, 'biomass': biomass}, space=space)


def _medium(label='medium', glucose=10.0, space=None):
    """Return a medium that holds one substrate, or one at every site of a space."""
    return Element(label, values={'glucose': glucose}, space=space)


class UptakeInteraction(Interaction):
    """Michaelis-Menten bound of the uptake of a cell and the consumption of the substrate."""

    def __init__(self, label, *, cell: Role = 'cell', medium: Role = 'medium', vmax: float = 10.0, km: float = 0.015):
        super().__init__(label)
        self.cell = cell
        self.medium = medium
        self.vmax = vmax
        self.km = km

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            uptake = s.cell.uptake
            self.laws_ = [
                ConstraintLaw(uptake <= s.vmax * s.medium.glucose / (s.km + s.medium.glucose), target=uptake),
                RateLaw(s.medium.glucose, -uptake * s.cell.biomass),
            ]


def _culture_cell(biomass=1.0, glucose=10.0):
    """Return a well-mixed culture of one cell in a medium."""
    return System(
        elements=[_cell(biomass=biomass), _medium(glucose=glucose)],
        interactions=[CellInteraction('growth'), UptakeInteraction('u')],
    )


def _toy_values(system):
    """Return the values of the program of a system at its first frame."""
    trajectory = system.prepare().simulate(until=1.0)
    return {name: float(trajectory.get_values(f'net.{name}')[0]) for name in TOY_NAMES}, trajectory


def _one_off(label, target, laws):
    """Return a rule written without a class on one target."""
    return Interaction(label, roles={'target': target}, laws=laws)


def test_lowered_toy_network_reaches_the_optimum_of_raman():
    """Test that the toy network of Raman reaches growth 50 with every balance zero."""
    fluxes, _ = _toy_values(_toy_network())
    assert fluxes['vbio'] == pytest.approx(50, abs=1e-9)
    assert fluxes['v1'] + fluxes['v3'] - fluxes['v4'] - fluxes['v5'] == pytest.approx(0, abs=1e-9)
    assert fluxes['v1'] + fluxes['v3'] - fluxes['v7'] - 10 * fluxes['vbio'] == pytest.approx(0, abs=1e-9)


def test_lowered_flow_network_equals_the_maximum_flow_of_networkx():
    """Test that a transport network written as a program reaches the maximum flow of the same graph."""
    graph = nx.DiGraph()
    graph.add_edge('s', 'a', capacity=4.0)
    graph.add_edge('a', 'b', capacity=3.0)
    graph.add_edge('s', 'b', capacity=1.0)

    def routes(s):
        n = s.target
        constraints = _constraints(n.sa <= 4, n.ab <= 3, n.sb <= 1, n.sa >= 0, n.ab >= 0, n.sb >= 0, n.sa - n.ab == 0)
        return [*constraints, ObjectiveLaw(-(n.ab + n.sb), [n.sa, n.ab, n.sb])]

    system = System(
        elements=[Element('n', values={'sa': 0.0, 'ab': 0.0, 'sb': 0.0})],
        interactions=[_one_off('routes', 'n', routes)],
    )
    trajectory = system.prepare().simulate(until=1.0)
    expected, _ = nx.maximum_flow(graph, 's', 'b')
    assert -float(trajectory.get_values('n.objective_0')[0]) == pytest.approx(expected)


def test_lowered_lexicographic_objectives_keep_the_first_optimum():
    """Test that the total flux falls under the second objective, and the first optimum stays."""
    single, _ = _toy_values(_toy_network())
    lexicographic, trajectory = _toy_values(_toy_network(total=True))
    assert lexicographic['vbio'] == pytest.approx(single['vbio'], abs=1e-9)
    assert sum(map(abs, lexicographic.values())) <= sum(map(abs, single.values())) + 1e-6
    assert float(trajectory.get_values('net.objective_1')[0]) == pytest.approx(sum(map(abs, lexicographic.values())))


def test_issues_report_a_nonlinear_relation_with_both_variables():
    """Test that a constraint with the product of two variables gives the issue that names them."""

    def product(s):
        p = s.target
        return [ConstraintLaw(p.v * p.w <= 1), ObjectiveLaw(-p.v, [p.v, p.w])]

    system = System(
        elements=[Element('p', values={'v': 0.0, 'w': 0.0})],
        interactions=[_one_off('product', 'p', product)],
    )
    report = PassCompiler().prepare(system).report_
    issues = [issue for issue in report.issues if issue.code == 'linear-program-form']
    assert len(issues) == 1
    assert 'v_p' in issues[0].message
    assert 'w_p' in issues[0].message
    assert 'v_p*w_p <= 1' in issues[0].message


def test_issues_report_a_coefficient_that_reads_a_state():
    """Test that a coefficient of a variable that reads a state gives the form issue."""

    def reading(s):
        p = s.target
        return [ConstraintLaw(p.x * p.v <= 1), ObjectiveLaw(-p.v, [p.v]), RateLaw(p.x, 0 * p.x)]

    system = System(
        elements=[Element('p', values={'v': 0.0, 'x': 1.0})],
        interactions=[_one_off('reading', 'p', reading)],
    )
    with pytest.raises(ValidationError, match='linear-program-form'):
        system.prepare()


def test_issues_report_two_laws_that_hold_objectives():
    """Test that two merged laws with objectives give the issue that names both paths."""
    second = _one_off(
        'second',
        'flows',
        lambda s: [ConstraintLaw(s.target.v <= 1), ObjectiveLaw(-s.target.v, [s.target.v])],
    )
    report = PassCompiler().prepare(_flows(None, second)).report_
    issues = [issue for issue in report.issues if issue.code == 'linear-program-objectives']
    assert len(issues) == 1
    assert 'flow' in issues[0].message
    assert 'second' in issues[0].message


def test_issues_name_a_variable_that_no_constraint_bounds():
    """Test that a target of a program that no constraint reads is an issue of `prepare`."""
    flows = Element('flows', values={'direct': 0.0, 'detour': 0.0})

    def laws(s):
        direct, detour = s.target.direct, s.target.detour
        return [ObjectiveLaw(-(direct + detour), [direct, detour]), ConstraintLaw(direct <= 3, target=direct)]

    routing = Interaction('routing', roles={'target': 'flows'}, laws=laws)
    with pytest.raises(ValidationError, match=r'flows\.detour, which no constraint bounds.*linear-program-variable'):
        System(elements=[flows], interactions=[routing]).prepare()


def test_merged_laws_use_the_smaller_of_two_upper_bounds():
    """Test that two interactions that bound one variable by 8 and 6 leave the bound 6."""
    bounds = (UpperBoundInteraction('a', bound=8.0), UpperBoundInteraction('b', bound=6.0))
    trajectory = _flows(FlowsElement('flows', capacity=100.0), *bounds).prepare().simulate(until=1.0)
    assert float(trajectory.get_values('flows.v')[0]) == pytest.approx(6)


def test_merged_laws_bound_uptake_by_michaelis_menten():
    """Test that an uptake of 10 S / (0.015 + S) at S = 0.015 is bounded by 5 at that solve."""
    problem = _culture_cell(glucose=0.015).prepare()._problem
    state = problem.programs[0].run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)).values_
    assert problem.states.read(state, 'cell.uptake') == pytest.approx(5)


def test_switches_keep_a_reaction_with_one_isozyme():
    """Test that a reaction bounded through two genes keeps its bound with one off, and is zero with both off."""
    one_off = _flows(FlowsElement('flows', capacity=20.0, g1=0.0))
    both_off = _flows(FlowsElement('flows', capacity=20.0, g1=0.0, g2=0.0))
    kept = one_off.prepare().simulate(until=1.0).get_values('flows.v')[0]
    gone = both_off.prepare().simulate(until=1.0).get_values('flows.v')[0]
    assert float(kept) == pytest.approx(20)
    assert float(gone) == pytest.approx(0)


def test_switches_set_between_two_runs_change_the_objective():
    """Test that switching off an essential reaction between two runs makes the second objective zero."""
    system = _flows()
    first = -float(system.prepare().simulate(until=1.0).get_values('flows.objective_0')[0])
    system.set_params(flows__g1=0.0, flows__g2=0.0)
    second = -float(system.prepare().simulate(until=1.0).get_values('flows.objective_0')[0])
    assert first == pytest.approx(10)
    assert second == pytest.approx(0, abs=1e-9)


def test_switches_follow_a_repressing_node_through_an_update_law():
    """Test that a node that is on holds a reaction at zero, and that the reaction may carry flux after it turns off."""

    def program(s):
        cell = s.target
        return [*_constraints(cell.v >= 0, cell.v <= 5 * cell.switch), ObjectiveLaw(-cell.v, [cell.v])]

    def repression(s):
        return [UpdateLaw(s.node.expressed, 0 * s.node.expressed), UpdateLaw(s.target.switch, 1 - s.node.expressed)]

    system = System(
        elements=[Element('cell', values={'v': 0.0, 'switch': 0.0}), Element('node', values={'expressed': 1.0})],
        interactions=[
            _one_off('program', 'cell', program),
            Interaction('r', roles={'node': 'node', 'target': 'cell'}, laws=repression),
        ],
    )
    values = system.prepare().simulate(until=3).get_values('cell.v')
    assert float(values[0]) == pytest.approx(0)
    assert float(values[-1]) == pytest.approx(5)


def test_infeasible_program_raises_with_the_program_time_and_subset():
    """Test that a required flux above its supply raises an error that names both variables."""

    def maintenance(infeasible):
        def laws(s):
            cell = s.target
            constraints = _constraints(
                cell.maintenance >= 8,
                cell.uptake >= 0,
                cell.uptake <= 0,
                cell.maintenance - cell.uptake <= 0,
            )
            return [*constraints, ObjectiveLaw(cell.uptake, [cell.maintenance, cell.uptake], infeasible=infeasible)]

        return System(
            elements=[Element('cell', values={'maintenance': 0.0, 'uptake': 0.0})],
            interactions=[_one_off('maintenance', 'cell', laws)],
        )

    with pytest.raises(InfeasibleError, match=r'cell.*time 0\.0.*cell\.maintenance.*cell\.uptake'):
        maintenance('raise').prepare().simulate(until=1.0)
    trajectory = maintenance('zero').prepare().simulate(until=1.0)
    assert float(trajectory.get_values('cell.maintenance')[0]) == 0.0
    assert float(trajectory.get_values('cell.objective_0')[0]) == 0.0


def test_infeasible_program_with_the_hold_policy_keeps_the_last_values():
    """Test that the hold policy keeps the last values through an infeasible solve."""
    problem = _flows(infeasible='hold').prepare()._problem
    kernel = problem.programs[0]
    position = problem.parameters.names.index('flows.capacity')
    p = problem.initial_parameters.copy()
    first = kernel.run(RunStateSpec(0.0, problem.initial_state, p)).values_.copy()
    p[position] = -1.0
    second = kernel.run(RunStateSpec(1.0, problem.initial_state, p)).values_
    np.testing.assert_allclose(second, first)


def test_recorded_duals_give_the_change_of_the_objective():
    """Test that the dual of the binding constraint is the change of the minimised `-vbio` with its right side."""
    system = _toy_network(duals=True).prepare()
    trajectory = system.simulate(until=1.0)
    relations = list(system._problem.programs[0].program.relations)
    shared = [relations.index(f'{name}_net <= 1000') for name in ('v6', 'b2')]
    assert sum(float(trajectory.get_values(f'net.dual_{item}')[0]) for item in shared) == pytest.approx(-0.05)


def _clock():
    """Return a counter whose update makes its system step in discrete time."""
    count = _one_off('tick', 'clock', lambda s: [UpdateLaw(s.target.n, s.target.n + 1)])
    return Element('clock', values={'n': 0.0}), count


def test_form_of_programs_keeps_the_form_of_the_laws_beside_them():
    """Test the forms of a program beside rate laws and beside updates in discrete time."""
    assert _culture_cell().prepare()._problem.form == FormKind.ODE
    clock, tick = _clock()
    beside = System(elements=[FlowsElement('flows'), clock], interactions=[FlowInteraction('flow'), tick])
    assert beside.prepare()._problem.form == FormKind.DISCRETE


def test_form_of_a_program_alone_is_static():
    """Test that a program alone is a static solution."""
    assert _flows().prepare()._problem.form == FormKind.STATIC


def test_discrete_program_solves_once_per_step_before_the_updates():
    """Test that a program under discrete time solves once per step."""
    clock, tick = _clock()
    system = System(elements=[FlowsElement('flows'), clock], interactions=[FlowInteraction('flow'), tick])
    problem = system.prepare()._problem
    kernel = problem.programs[0]
    x, p = problem.initial_state, problem.initial_parameters
    for step in range(3):
        x = problem.update(float(step), x, p)
    assert kernel.solves_ == 3
    assert problem.states.read(x, 'flows.v') == pytest.approx(10)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_drift_solves_the_program_once_per_evaluation(backend):
    """Test that the drift solves one program once, however many rates read it, on both backends."""

    def rates(s):
        e = s.target
        return [
            ConstraintLaw(e.v <= e.a),
            ObjectiveLaw(-e.v, [e.v]),
            RateLaw(e.a, -e.v),
            RateLaw(e.b, e.v),
            RateLaw(e.c, 2 * e.v),
        ]

    system = System(
        elements=[Element('e', values={'v': 0.0, 'a': 1.0, 'b': 1.0, 'c': 1.0})],
        interactions=[_one_off('rates', 'e', rates)],
        backend=backend(),
    )
    problem = system.prepare()._problem
    derivative = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.programs[0].solves_ == 1
    assert problem.states.read(derivative, 'e.c') == pytest.approx(2)


def test_static_program_solves_once_per_frame():
    """Test that a program alone solves once for its one frame."""
    trajectory = _flows().prepare().simulate(until=1.0)
    assert trajectory.problem.programs[0].solves_ == 1


class FluxesInteraction(Interaction):
    """Program of the uptake and growth of a culture, the yield fixed."""

    def __init__(self, label, *, target: Role = 'fluxes'):
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            self.laws_ = [
                *_constraints(s.growth == 0.1 * s.uptake, s.uptake >= 0),
                ObjectiveLaw(-s.growth, [s.uptake, s.growth], infeasible='zero'),
            ]


class BoundInteraction(Interaction):
    """Michaelis-Menten bound of the uptake of a culture."""

    def __init__(self, label, *, fluxes: Role = 'fluxes', culture: Role = 'culture'):
        super().__init__(label)
        self.fluxes = fluxes
        self.culture = culture

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            uptake, g = context.fluxes.uptake, context.culture.g
            self.laws_ = [ConstraintLaw(uptake <= 10 * g / (0.015 + g), target=uptake)]


class CultureGrowthInteraction(Interaction):
    """Growth of the biomass and consumption of the glucose of a culture at the fluxes of its program."""

    def __init__(self, label, *, fluxes: Role = 'fluxes', culture: Role = 'culture'):
        super().__init__(label)
        self.fluxes = fluxes
        self.culture = culture

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [
                RateLaw(s.culture.x, s.fluxes.growth * s.culture.x),
                RateLaw(s.culture.g, -s.fluxes.uptake * s.culture.x),
            ]


def _culture(*, nested: bool, biomass: float = 0.01, glucose: float = 10.0, solver='auto'):
    """Return a culture whose program runs beside its rates, or in a discrete system beside them."""
    fluxes = Element('fluxes', values={'uptake': 0.0, 'growth': 0.0})
    culture = Element('culture', values={'x': biomass, 'g': glucose})
    if not nested:
        return System(
            elements=[fluxes, culture],
            interactions=[
                FluxesInteraction('program'),
                BoundInteraction('bound'),
                CultureGrowthInteraction('growth'),
            ],
            solver=solver,
        )
    clock, tick = _clock()
    return System(
        systems=[
            System('fba', elements=[fluxes, clock], interactions=[FluxesInteraction('program'), tick]),
            System('medium', elements=[culture]),
        ],
        interactions=[
            BoundInteraction('bound', fluxes='fba/fluxes', culture='medium/culture'),
            CultureGrowthInteraction('growth', fluxes='fba/fluxes', culture='medium/culture'),
        ],
        solver=solver,
    )


def test_cosimulated_program_compiles_to_a_map_and_an_ode_partition():
    """Test that a program in a discrete system beside rates compiles to two partitions."""
    problem = _culture(nested=True).prepare()._problem
    assert problem.form == FormKind.COUPLED
    assert {item.problem.form for item in problem.partitions} == {FormKind.DISCRETE, FormKind.ODE}


def test_cosimulated_program_matches_the_closed_form_of_raman():
    """Test one synchronisation interval against the closed form of Raman's Eqs. 9.8 and 9.9."""
    interval = 0.1
    solver = CoSimulationSolver(sync_dt=interval, solvers={'medium': RK4Solver(dt=0.001)})
    trajectory = _culture(nested=True, solver=solver).prepare().simulate(until=interval, dt=interval)
    uptake = 10 * 10.0 / (0.015 + 10.0)
    mu = 0.1 * uptake
    biomass = 0.01 * np.exp(mu * interval)
    glucose = 10.0 + uptake / mu * 0.01 * (1 - np.exp(mu * interval))
    assert float(trajectory.get_values('medium/culture.x')[-1]) == pytest.approx(biomass, rel=1e-8)
    assert float(trajectory.get_values('medium/culture.g')[-1]) == pytest.approx(glucose, rel=1e-8)


def test_cosimulated_program_approaches_the_direct_approach():
    """Test that the error against the direct approach falls at least linearly with the interval."""
    errors = []
    for interval in (0.1, 0.05, 0.025):
        solver = CoSimulationSolver(sync_dt=interval, solvers={'medium': RK4Solver(dt=0.001)})
        system = _culture(nested=True, glucose=0.5, solver=solver).prepare()
        errors.append(system.simulate(until=0.5, dt=0.1).get_values('medium/culture.x')[-1])
    direct = _culture(nested=False, glucose=0.5, solver=RK4Solver(dt=0.001)).prepare().simulate(until=0.5, dt=0.1)
    reference = float(direct.get_values('culture.x')[-1])
    gaps = [abs(float(item) - reference) for item in errors]
    assert gaps[1] <= gaps[0] * 0.75
    assert gaps[2] <= gaps[1] * 0.75


def _lattice_system(sites, active_at, solver='auto'):
    """Return a lattice of cells whose program is active above a biomass threshold."""
    initial = np.zeros(sites * sites)
    initial[active_at] = 1.0
    space = LatticeSpace((sites, sites))
    return System(
        elements=[
            _cell('cells', biomass=initial, space=space),
            _medium(glucose=np.full(sites * sites, 5.0), space=space),
        ],
        interactions=[CellInteraction('growth', target='cells', active=True), UptakeInteraction('u', cell='cells')],
        space=space,
        solver=solver,
    )


def test_group_program_solves_only_the_active_sites():
    """Test that one evaluation on 25 by 25 sites solves the 9 active ones, and the others have zero variables."""
    active = np.arange(9) * 7
    problem = _lattice_system(25, active).prepare()._problem
    state = problem.algebraic(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.programs[0].solves_ == 9
    values = problem.states.read(state, 'cells.uptake')
    assert np.count_nonzero(values) == 9
    assert np.all(values[active] > 0)


def test_group_program_on_a_single_site_matches_the_well_mixed_culture():
    """Test that a lattice of one site without diffusion matches the well-mixed culture."""
    mixed = System(
        elements=[_cell(biomass=1.0), _medium(glucose=5.0)],
        interactions=[CellInteraction('growth'), UptakeInteraction('u')],
        solver=RK4Solver(dt=0.01),
    )
    single = _lattice_system(1, np.array([0]), solver=RK4Solver(dt=0.01))
    first = mixed.prepare().simulate(until=0.5, dt=0.1)
    second = single.prepare().simulate(until=0.5, dt=0.1)
    assert float(np.ravel(second.get_values('cells.biomass'))[-1]) == pytest.approx(
        float(first.get_values('cell.biomass')[-1]),
        rel=1e-6,
    )


def test_recorded_exchange_is_zero_after_depletion():
    """Test that a trajectory holds the uptake at every frame, and that it is zero once the substrate is gone."""
    system = _culture(nested=False, biomass=1.0, glucose=0.1, solver=RK4Solver(dt=0.001))
    trajectory = system.prepare().simulate(until=2.0, dt=0.1)
    uptake = trajectory.get_values('fluxes.uptake')
    assert len(uptake) == len(trajectory.times)
    assert float(uptake[0]) > 0
    assert float(uptake[-1]) == pytest.approx(0, abs=1e-3)
