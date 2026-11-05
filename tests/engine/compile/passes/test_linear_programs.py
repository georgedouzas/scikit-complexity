"""Test the `_linear_programs` pass of `skcomplex.engine.compile.passes`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import LinearProgramLaw, RateLaw, UpdateLaw
from skcomplex.core.laws.utils import either
from skcomplex.core.spaces import DiscreteTime, LatticeSpace, StaticTime
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import PassCompiler, compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import CoSimulationSolver, RK4Solver
from skcomplex.errors import InfeasibleError, ValidationError
from skcomplex.specs import LinearObjectiveSpec, ParameterSpec, RoleSpec, RunStateSpec, StateSpec
from skcomplex.types import FormKind

Flux = Annotated[float, StateSpec(algebraic=True)]
Amount = Annotated[float, StateSpec()]
Number = Annotated[float, ParameterSpec()]
TOY_NAMES = ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'b1', 'b2', 'vbio')
TOY_REVERSIBLE = ('v4', 'v6', 'b2')


class ToyNetworkElement(Element):
    """Toy network of Table 8.2 of Raman (2021), with the growth objective and optionally the total flux after it."""

    def __init__(
        self,
        label,
        *,
        v1: Flux = 0.0,
        v2: Flux = 0.0,
        v3: Flux = 0.0,
        v4: Flux = 0.0,
        v5: Flux = 0.0,
        v6: Flux = 0.0,
        v7: Flux = 0.0,
        b1: Flux = 0.0,
        b2: Flux = 0.0,
        vbio: Flux = 0.0,
        total: bool = False,
        duals: bool = False,
    ):
        super().__init__(label)
        self.v1, self.v2, self.v3, self.v4, self.v5 = v1, v2, v3, v4, v5
        self.v6, self.v7, self.b1, self.b2, self.vbio = v6, v7, b1, b2, vbio
        self.total = total
        self.duals = duals

    def _get_laws(self, s):
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
        objectives = [LinearObjectiveSpec(s.vbio)]
        if self.total:
            objectives.append(LinearObjectiveSpec(sum(abs(item) for item in variables), sense='min'))
        return [LinearProgramLaw(variables, [*balances, *bounds], objectives, duals=self.duals)]


class FlowElement(Element):
    """Two flows with the capacity of their sum in a parameter, and switches that gate the first."""

    def __init__(
        self,
        label,
        *,
        v: Flux = 0.0,
        w: Flux = 0.0,
        capacity: Number = 10.0,
        g1: Number = 1.0,
        g2: Number = 1.0,
        infeasible: str = 'raise',
        minimum: float = 0.0,
    ):
        super().__init__(label)
        self.v, self.w = v, w
        self.capacity, self.g1, self.g2 = capacity, g1, g2
        self.infeasible = infeasible
        self.minimum = minimum

    def _get_laws(self, s):
        constraints = [
            s.v + s.w <= s.capacity,
            s.w >= self.minimum,
            s.v >= 0,
            s.v <= 1000 * either(s.g1 > 0.5, s.g2 > 0.5),
        ]
        return [LinearProgramLaw([s.v, s.w], constraints, [LinearObjectiveSpec(s.v)], infeasible=self.infeasible)]


class UpperBoundInteraction(Interaction):
    """Upper bound of one variable, which merges into the program of the variable."""

    def __init__(self, label, target: Annotated[str, RoleSpec()] = 'flows', bound: Number = 8.0):
        super().__init__(label)
        self.target = target
        self.bound = bound

    def _get_laws(self, s):
        return [LinearProgramLaw([s.target.v], [s.target.v <= s.bound])]


class CellElement(Element):
    """Cell whose growth and uptake a program sets, the yield fixed and the uptake bound by an interaction."""

    def __init__(
        self,
        label,
        uptake: Flux = 0.0,
        growth: Flux = 0.0,
        biomass: Amount = 1.0,
        *,
        active: object = None,
        infeasible: str = 'zero',
    ):
        super().__init__(label)
        self.uptake, self.growth, self.biomass = uptake, growth, biomass
        self.active = active
        self.infeasible = infeasible

    def _get_laws(self, s):
        return [
            LinearProgramLaw(
                [s.uptake, s.growth],
                [s.growth == 0.1 * s.uptake, s.uptake >= 0],
                [LinearObjectiveSpec(s.growth)],
                infeasible=self.infeasible,
                active=s.biomass > 0.5 if self.active else None,
            ),
            RateLaw(s.biomass, s.growth * s.biomass),
        ]


class ActiveCellElement(CellElement):
    """Cell whose program solves only above a biomass threshold."""

    def __init__(self, label, uptake: Flux = 0.0, growth: Flux = 0.0, biomass: Amount = 1.0):
        super().__init__(label, uptake, growth, biomass, active=True)


class MediumElement(Element):
    """Medium that holds one substrate."""

    def __init__(self, label, glucose: Amount = 10.0):
        super().__init__(label)
        self.glucose = glucose


class UptakeInteraction(Interaction):
    """Michaelis-Menten bound of the uptake of a cell and the consumption of the substrate."""

    def __init__(
        self,
        label,
        cell: Annotated[str, RoleSpec()] = 'cell',
        medium: Annotated[str, RoleSpec()] = 'medium',
        vmax: Number = 10.0,
        km: Number = 0.015,
    ):
        super().__init__(label)
        self.cell = cell
        self.medium = medium
        self.vmax = vmax
        self.km = km

    def _get_laws(self, s):
        return [
            LinearProgramLaw(
                [s.cell.uptake],
                [s.cell.uptake <= s.vmax * s.medium.glucose / (s.km + s.medium.glucose)],
            ),
            RateLaw(s.medium.glucose, -s.cell.uptake * s.cell.biomass),
        ]


def _toy_values(system):
    """Return the values of the program of a system at its one frame."""
    trajectory = Simulation(system).check().prepare().run(1.0).trajectory_
    return {name: float(trajectory.values(f'net.{name}')[0]) for name in TOY_NAMES}, trajectory


def test_lowered_toy_network_reaches_the_optimum_of_raman():
    """Test that the toy network of Raman reaches growth 50 with every balance zero."""
    system = System(elements=[ToyNetworkElement('net')], time=StaticTime())
    fluxes, _ = _toy_values(system)
    assert fluxes['vbio'] == pytest.approx(50, abs=1e-9)
    assert fluxes['v1'] + fluxes['v3'] - fluxes['v4'] - fluxes['v5'] == pytest.approx(0, abs=1e-9)
    assert fluxes['v1'] + fluxes['v3'] - fluxes['v7'] - 10 * fluxes['vbio'] == pytest.approx(0, abs=1e-9)


def test_lowered_flow_network_equals_the_maximum_flow_of_networkx():
    """Test that a transport network written as a program reaches the maximum flow of the same graph."""
    graph = nx.DiGraph()
    graph.add_edge('s', 'a', capacity=4.0)
    graph.add_edge('a', 'b', capacity=3.0)
    graph.add_edge('s', 'b', capacity=1.0)

    class RoutesElement(Element):
        def __init__(self, label, sa: Flux = 0.0, ab: Flux = 0.0, sb: Flux = 0.0):
            super().__init__(label)
            self.sa, self.ab, self.sb = sa, ab, sb

        def _get_laws(self, s):
            constraints = [
                s.sa <= 4,
                s.ab <= 3,
                s.sb <= 1,
                s.sa >= 0,
                s.ab >= 0,
                s.sb >= 0,
                s.sa - s.ab == 0,
            ]
            return [LinearProgramLaw([s.sa, s.ab, s.sb], constraints, [LinearObjectiveSpec(s.ab + s.sb)])]

    trajectory = Simulation(System(elements=[RoutesElement('n')], time=StaticTime())).check().prepare().run(1.0)
    expected, _ = nx.maximum_flow(graph, 's', 'b')
    assert float(trajectory.trajectory_.values('n.objective_0')[0]) == pytest.approx(expected)


def test_lowered_lexicographic_objectives_keep_the_first_optimum():
    """Test that the total flux falls under the second objective, and the first optimum stays."""
    single, _ = _toy_values(System(elements=[ToyNetworkElement('net')], time=StaticTime()))
    lexicographic, trajectory = _toy_values(System(elements=[ToyNetworkElement('net', total=True)], time=StaticTime()))
    assert lexicographic['vbio'] == pytest.approx(single['vbio'], abs=1e-9)
    assert sum(map(abs, lexicographic.values())) <= sum(map(abs, single.values())) + 1e-6
    assert float(trajectory.values('net.objective_1')[0]) == pytest.approx(sum(map(abs, lexicographic.values())))


def test_issues_name_a_variable_that_is_not_algebraic():
    """Test that a variable that is not algebraic raises an error that says how to declare it."""

    class PlainElement(Element):
        def __init__(self, label, x: Amount = 0.0):
            super().__init__(label)
            self.x = x

        def _get_laws(self, s):
            return [LinearProgramLaw([s.x], [s.x <= 1], [LinearObjectiveSpec(s.x)])]

    with pytest.raises(ValidationError, match=r'p\.x.*StateSpec\(algebraic=True\)'):
        compile_system(System(elements=[PlainElement('p')], time=StaticTime()))


def test_issues_report_a_nonlinear_relation_with_both_variables():
    """Test that a constraint with the product of two variables gives the issue that names them."""

    class ProductElement(Element):
        def __init__(self, label, v: Flux = 0.0, w: Flux = 0.0):
            super().__init__(label)
            self.v, self.w = v, w

        def _get_laws(self, s):
            return [LinearProgramLaw([s.v, s.w], [s.v * s.w <= 1], [LinearObjectiveSpec(s.v)])]

    system = System(elements=[ProductElement('p')], time=StaticTime())
    report = PassCompiler().check().prepare(system).report_
    issues = [issue for issue in report.issues if issue.code == 'linear-program-form']
    assert len(issues) == 1
    assert 'v_p' in issues[0].message
    assert 'w_p' in issues[0].message
    assert 'v_p*w_p <= 1' in issues[0].message


def test_issues_report_a_coefficient_that_reads_a_state():
    """Test that a coefficient of a variable that reads a state gives the form issue."""

    class ReadingElement(Element):
        def __init__(self, label, v: Flux = 0.0, x: Amount = 1.0):
            super().__init__(label)
            self.v, self.x = v, x

        def _get_laws(self, s):
            return [LinearProgramLaw([s.v], [s.x * s.v <= 1], [LinearObjectiveSpec(s.v)])]

    with pytest.raises(ValidationError, match='linear-program-form'):
        compile_system(System(elements=[ReadingElement('p')], time=StaticTime()))


def test_issues_report_two_laws_that_hold_objectives():
    """Test that two merged laws with objectives give the issue that names both paths."""

    class HoldingInteraction(Interaction):
        def __init__(self, label, target: Annotated[str, RoleSpec()] = 'flows'):
            super().__init__(label)
            self.target = target

        def _get_laws(self, s):
            return [LinearProgramLaw([s.target.v], [s.target.v <= 1], [LinearObjectiveSpec(s.target.v)])]

    system = System(elements=[FlowElement('flows')], interactions=[HoldingInteraction('second')], time=StaticTime())
    report = PassCompiler().check().prepare(system).report_
    issues = [issue for issue in report.issues if issue.code == 'linear-program-objectives']
    assert len(issues) == 1
    assert 'flows' in issues[0].message
    assert 'second' in issues[0].message


def test_merged_laws_use_the_smaller_of_two_upper_bounds():
    """Test that two interactions that bound one variable by 8 and 6 leave the bound 6."""
    system = System(
        elements=[FlowElement('flows', capacity=100.0)],
        interactions=[UpperBoundInteraction('a', bound=8.0), UpperBoundInteraction('b', bound=6.0)],
        time=StaticTime(),
    )
    trajectory = Simulation(system).check().prepare().run(1.0).trajectory_
    assert float(trajectory.values('flows.v')[0]) == pytest.approx(6)


def test_merged_laws_bound_uptake_by_michaelis_menten():
    """Test that an uptake of 10 S / (0.015 + S) at S = 0.015 is bounded by 5 at that solve."""
    system = System(
        elements=[CellElement('cell'), MediumElement('medium', glucose=0.015)],
        interactions=[UptakeInteraction('u')],
    )
    problem = compile_system(system)
    state = problem.programs[0].run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)).values_
    assert problem.states.read(state, 'cell.uptake') == pytest.approx(5)


def test_switches_keep_a_reaction_with_one_isozyme():
    """Test that a reaction bounded through two genes keeps its bound with one off, and is zero with both off."""
    time = StaticTime()
    one_off = System(elements=[FlowElement('flows', capacity=20.0, g1=0.0)], time=time)
    both_off = System(elements=[FlowElement('flows', capacity=20.0, g1=0.0, g2=0.0)], time=time)
    kept = Simulation(one_off).check().prepare().run(1.0).trajectory_.values('flows.v')[0]
    gone = Simulation(both_off).check().prepare().run(1.0).trajectory_.values('flows.v')[0]
    assert float(kept) == pytest.approx(20)
    assert float(gone) == pytest.approx(0)


def test_switches_set_between_two_runs_change_the_objective():
    """Test that switching off an essential reaction between two runs makes the second objective zero."""
    simulation = Simulation(System(elements=[FlowElement('flows')], time=StaticTime())).check().prepare()
    first = float(simulation.run(1.0).trajectory_.values('flows.objective_0')[0])
    simulation.update_parameters({'flows.g1': 0.0, 'flows.g2': 0.0})
    second = float(simulation.run(1.0).trajectory_.values('flows.objective_0')[0])
    assert first == pytest.approx(10)
    assert second == pytest.approx(0, abs=1e-9)


def test_switches_follow_a_repressing_node_through_an_update_law():
    """Test that a node that is on holds a reaction at zero, and that the reaction may carry flux after it turns off."""

    class SwitchedElement(Element):
        def __init__(
            self,
            label,
            v: Flux = 0.0,
            switch: Annotated[float, ParameterSpec(adaptive=True, bounds=(0, 1))] = 1.0,
        ):
            super().__init__(label)
            self.v, self.switch = v, switch

        def _get_laws(self, s):
            return [LinearProgramLaw([s.v], [s.v >= 0, s.v <= 5 * s.switch], [LinearObjectiveSpec(s.v)])]

    class NodeElement(Element):
        def __init__(self, label, on: Annotated[float, StateSpec()] = 1.0):
            super().__init__(label)
            self.on = on

        def _get_laws(self, s):
            return [UpdateLaw(s.on, 0 * s.on)]

    class RepressionInteraction(Interaction):
        def __init__(
            self,
            label,
            node: Annotated[str, RoleSpec()] = 'node',
            target: Annotated[str, RoleSpec()] = 'cell',
        ):
            super().__init__(label)
            self.node = node
            self.target = target

        def _get_laws(self, s):
            return [UpdateLaw(s.target.switch, 1 - s.node.on)]

    system = System(
        elements=[SwitchedElement('cell', switch=0.0), NodeElement('node')],
        interactions=[RepressionInteraction('r')],
        time=DiscreteTime(),
    )
    trajectory = Simulation(system).check().prepare().run(3).trajectory_
    values = trajectory.values('cell.v')
    assert float(values[0]) == pytest.approx(0)
    assert float(values[-1]) == pytest.approx(5)


def test_infeasible_program_raises_with_the_program_time_and_subset():
    """Test that a required flux above its supply raises an error that names both variables."""

    class MaintenanceElement(Element):
        def __init__(self, label, maintenance: Flux = 0.0, uptake: Flux = 0.0, *, infeasible: str = 'raise'):
            super().__init__(label)
            self.maintenance, self.uptake = maintenance, uptake
            self.infeasible = infeasible

        def _get_laws(self, s):
            return [
                LinearProgramLaw(
                    [s.maintenance, s.uptake],
                    [s.maintenance >= 8, s.uptake >= 0, s.uptake <= 0, s.maintenance - s.uptake <= 0],
                    [LinearObjectiveSpec(s.uptake)],
                    infeasible=self.infeasible,
                ),
            ]

    system = System(elements=[MaintenanceElement('cell')], time=StaticTime())
    with pytest.raises(InfeasibleError, match=r'cell.*time 0\.0.*cell\.maintenance.*cell\.uptake'):
        Simulation(system).check().prepare().run(1.0)
    zero = System(elements=[MaintenanceElement('cell', infeasible='zero')], time=StaticTime())
    trajectory = Simulation(zero).check().prepare().run(1.0).trajectory_
    assert float(trajectory.values('cell.maintenance')[0]) == 0.0
    assert float(trajectory.values('cell.objective_0')[0]) == 0.0


def test_infeasible_program_with_the_hold_policy_keeps_the_last_values():
    """Test that the hold policy keeps the last values through an infeasible solve."""
    system = System(elements=[FlowElement('flows', infeasible='hold', minimum=0.0)], time=StaticTime())
    problem = compile_system(system)
    kernel = problem.programs[0]
    position = problem.parameters.names.index('flows.capacity')
    p = problem.initial_parameters.copy()
    first = kernel.run(RunStateSpec(0.0, problem.initial_state, p)).values_.copy()
    p[position] = -1.0
    second = kernel.run(RunStateSpec(1.0, problem.initial_state, p)).values_
    np.testing.assert_allclose(second, first)


def test_recorded_duals_give_the_change_of_the_objective():
    """Test that the dual of the binding constraint is the change of the optimum with its right side."""
    system = System(elements=[ToyNetworkElement('net', duals=True)], time=StaticTime())
    trajectory = Simulation(system).check().prepare().run(1.0).trajectory_
    relations = list(compile_system(system).programs[0].program.relations)
    shared = [relations.index(f'{name}_net <= 1000') for name in ('v6', 'b2')]
    assert sum(float(trajectory.values(f'net.dual_{item}')[0]) for item in shared) == pytest.approx(0.05)


def test_form_of_programs_keeps_the_form_of_the_laws_beside_them():
    """Test the forms of a program beside rate laws, alone in discrete time, and beside an ODE system."""
    beside = System(elements=[CellElement('cell'), MediumElement('medium')], interactions=[UptakeInteraction('u')])
    assert compile_system(beside).form == FormKind.ODE
    alone = System(elements=[FlowElement('flows')], time=DiscreteTime())
    assert compile_system(alone).form == FormKind.DISCRETE
    assert compile_system(System(elements=[FlowElement('flows')], time=StaticTime())).form == FormKind.STATIC


def test_discrete_program_solves_once_per_step_before_the_updates():
    """Test that a program alone under discrete time solves once per step."""
    system = System(elements=[FlowElement('flows')], time=DiscreteTime())
    problem = compile_system(system)
    kernel = problem.programs[0]
    x, p = problem.initial_state, problem.initial_parameters
    for step in range(3):
        x = problem.update(float(step), x, p)
    assert kernel.solves_ == 3
    assert problem.states.read(x, 'flows.v') == pytest.approx(10)


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_drift_solves_the_program_once_per_evaluation(backend):
    """Test that the drift solves one program once, however many rates read it, on both backends."""

    class ThreeRatesElement(Element):
        def __init__(self, label, v: Flux = 0.0, a: Amount = 1.0, b: Amount = 1.0, c: Amount = 1.0):
            super().__init__(label)
            self.v, self.a, self.b, self.c = v, a, b, c

        def _get_laws(self, s):
            return [
                LinearProgramLaw([s.v], [s.v <= s.a], [LinearObjectiveSpec(s.v)]),
                RateLaw(s.a, -s.v),
                RateLaw(s.b, s.v),
                RateLaw(s.c, 2 * s.v),
            ]

    problem = compile_system(System(elements=[ThreeRatesElement('e')]), backend=backend())
    derivative = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.programs[0].solves_ == 1
    assert problem.states.read(derivative, 'e.c') == pytest.approx(2)


def test_static_program_solves_once_per_frame():
    """Test that a program alone under static time solves once for its one frame."""
    simulation = Simulation(System(elements=[FlowElement('flows')], time=StaticTime())).check().prepare()
    simulation.run(1.0)
    assert simulation.problem_.programs[0].solves_ == 1


def _culture(*, nested: bool, biomass: float = 0.01, glucose: float = 10.0):
    """Return a culture whose program runs beside its rates, or in a discrete system beside them."""

    class FluxesElement(Element):
        def __init__(self, label, uptake: Flux = 0.0, growth: Flux = 0.0):
            super().__init__(label)
            self.uptake, self.growth = uptake, growth

        def _get_laws(self, s):
            return [
                LinearProgramLaw(
                    [s.uptake, s.growth],
                    [s.growth == 0.1 * s.uptake, s.uptake >= 0],
                    [LinearObjectiveSpec(s.growth)],
                    infeasible='zero',
                ),
            ]

    class CultureElement(Element):
        def __init__(self, label, x: Amount = biomass, g: Amount = glucose):
            super().__init__(label)
            self.x, self.g = x, g

    class BoundInteraction(Interaction):
        def __init__(
            self,
            label,
            fluxes: Annotated[str, RoleSpec()] = 'fluxes',
            culture: Annotated[str, RoleSpec()] = 'culture',
        ):
            super().__init__(label)
            self.fluxes = fluxes
            self.culture = culture

        def _get_laws(self, s):
            return [LinearProgramLaw([s.fluxes.uptake], [s.fluxes.uptake <= 10 * s.culture.g / (0.015 + s.culture.g)])]

    class GrowthInteraction(Interaction):
        def __init__(
            self,
            label,
            fluxes: Annotated[str, RoleSpec()] = 'fluxes',
            culture: Annotated[str, RoleSpec()] = 'culture',
        ):
            super().__init__(label)
            self.fluxes = fluxes
            self.culture = culture

        def _get_laws(self, s):
            return [
                RateLaw(s.culture.x, s.fluxes.growth * s.culture.x),
                RateLaw(s.culture.g, -s.fluxes.uptake * s.culture.x),
            ]

    if not nested:
        return System(
            elements=[FluxesElement('fluxes'), CultureElement('culture')],
            interactions=[BoundInteraction('bound'), GrowthInteraction('growth')],
        )
    return System(
        systems=[
            System('fba', elements=[FluxesElement('fluxes')], time=DiscreteTime()),
            System('medium', elements=[CultureElement('culture')]),
        ],
        interactions=[
            BoundInteraction('bound', fluxes='fba/fluxes', culture='medium/culture'),
            GrowthInteraction('growth', fluxes='fba/fluxes', culture='medium/culture'),
        ],
    )


def test_cosimulated_program_compiles_to_a_map_and_an_ode_partition():
    """Test that a program in a discrete system beside rates compiles to two partitions."""
    problem = compile_system(_culture(nested=True))
    assert problem.form == FormKind.COUPLED
    assert {item.problem.form for item in problem.partitions} == {FormKind.DISCRETE, FormKind.ODE}


def test_cosimulated_program_matches_the_closed_form_of_raman():
    """Test one synchronisation interval against the closed form of Raman's Eqs. 9.8 and 9.9."""
    interval = 0.1
    solver = CoSimulationSolver(sync_dt=interval, solvers={'medium': RK4Solver(dt=0.001)})
    simulation = Simulation(_culture(nested=True), solver=solver, dt=interval).check().prepare().run(interval)
    trajectory = simulation.trajectory_
    uptake = 10 * 10.0 / (0.015 + 10.0)
    mu = 0.1 * uptake
    biomass = 0.01 * np.exp(mu * interval)
    glucose = 10.0 + uptake / mu * 0.01 * (1 - np.exp(mu * interval))
    assert float(trajectory.values('medium/culture.x')[-1]) == pytest.approx(biomass, rel=1e-8)
    assert float(trajectory.values('medium/culture.g')[-1]) == pytest.approx(glucose, rel=1e-8)


def test_cosimulated_program_approaches_the_direct_approach():
    """Test that the error against the direct approach falls at least linearly with the interval."""
    errors = []
    for interval in (0.1, 0.05, 0.025):
        solver = CoSimulationSolver(sync_dt=interval, solvers={'medium': RK4Solver(dt=0.001)})
        simulation = Simulation(_culture(nested=True, glucose=0.5), solver=solver, dt=0.1).check().prepare()
        errors.append(simulation.run(0.5).trajectory_.values('medium/culture.x')[-1])
    direct_low = Simulation(_culture(nested=False, glucose=0.5), solver=RK4Solver(dt=0.001), dt=0.1).check().prepare()
    reference = float(direct_low.run(0.5).trajectory_.values('culture.x')[-1])
    gaps = [abs(float(item) - reference) for item in errors]
    assert gaps[1] <= gaps[0] * 0.75
    assert gaps[2] <= gaps[1] * 0.75


def _lattice_system(sites, active_at):
    """Return a lattice of cells whose program is active above a biomass threshold."""
    initial = np.zeros(sites * sites)
    initial[active_at] = 1.0
    return System(
        elements=[
            GroupElement('cells', element=ActiveCellElement, on='sites', values={'biomass': initial}),
            GroupElement('medium', element=MediumElement, on='sites', values={'glucose': 5.0}),
        ],
        interactions=[UptakeInteraction('u', cell='cells', medium='medium')],
        space=LatticeSpace((sites, sites)),
    )


def test_group_program_solves_only_the_active_sites():
    """Test that one evaluation on 25 by 25 sites solves the 9 active ones, and the others have zero variables."""
    active = np.arange(9) * 7
    problem = compile_system(_lattice_system(25, active))
    state = problem.algebraic(0.0, problem.initial_state, problem.initial_parameters)
    assert problem.programs[0].solves_ == 9
    values = problem.states.read(state, 'cells.uptake')
    assert np.count_nonzero(values) == 9
    assert np.all(values[active] > 0)


def test_group_program_on_a_single_site_matches_the_well_mixed_culture():
    """Test that a lattice of one site without diffusion matches the well-mixed culture."""
    mixed = System(
        elements=[CellElement('cell', biomass=1.0), MediumElement('medium', glucose=5.0)],
        interactions=[UptakeInteraction('u')],
    )
    single = _lattice_system(1, np.array([0]))
    first = Simulation(mixed, solver=RK4Solver(dt=0.01), dt=0.1).check().prepare().run(0.5).trajectory_
    second = Simulation(single, solver=RK4Solver(dt=0.01), dt=0.1).check().prepare().run(0.5).trajectory_
    assert float(np.ravel(second.values('cells.biomass'))[-1]) == pytest.approx(
        float(first.values('cell.biomass')[-1]),
        rel=1e-6,
    )


def test_recorded_exchange_is_zero_after_depletion():
    """Test that a trajectory holds the uptake at every frame, and that it is zero once the substrate is gone."""
    system = _culture(nested=False, biomass=1.0, glucose=0.1)
    trajectory = Simulation(system, solver=RK4Solver(dt=0.001), dt=0.1).check().prepare().run(2.0).trajectory_
    uptake = trajectory.values('fluxes.uptake')
    assert len(uptake) == len(trajectory.times)
    assert float(uptake[0]) > 0
    assert float(uptake[-1]) == pytest.approx(0, abs=1e-3)
