"""Test the `_equation_blocks` module of `skcomplex.core.engine.compile.passes`, through the systems it compiles."""

from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import exp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, EventLaw, NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import SolverError, ValidationError
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class BranchInteraction(Interaction):
    """Component between the nodes `a` and `b`, whose current `i`, held by `branch`, adds into both node balances.

    Every node holds its potential `v`, and the currents into a node add up to zero. The current `i` flows into the
    component at `a` and out of it at `b`. A subclass returns the equation of its branch.
    """

    def __init__(self, label: str, *, a: Role, b: Role, branch: Role) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.branch = branch

    def _get_branch_laws(self, s: SimpleNamespace) -> list[ConstraintLaw]:
        """Return the equations of the branch, none here."""
        del s
        return []

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the current into the balances of both nodes and the equations of the branch."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            i = context.branch.i
            self.laws_ = [
                ConstraintLaw(i, 0, target=context.a.v),
                ConstraintLaw(-i, 0, target=context.b.v),
                *self._get_branch_laws(context),
            ]


class ResistorInteraction(BranchInteraction):
    """Resistor that obeys Ohm's law."""

    def __init__(self, label: str, *, a: Role, b: Role, branch: Role, R: float = 1.0) -> None:
        super().__init__(label, a=a, b=b, branch=branch)
        self.R = R

    def _get_branch_laws(self, s):
        """Return Ohm's law."""
        return [ConstraintLaw(s.branch.i, (s.a.v - s.b.v) / s.R, target=s.branch.i)]


class CapacitorInteraction(BranchInteraction):
    """Capacitor of voltage `u`, held by its branch."""

    def __init__(self, label: str, *, a: Role, b: Role, branch: Role, C: float = 1.0) -> None:
        super().__init__(label, a=a, b=b, branch=branch)
        self.C = C

    def _get_branch_laws(self, s):
        """Return the charge balance and the voltage of the capacitor."""
        u, i = s.branch.u, s.branch.i
        return [ConstraintLaw(s.C * u.diff(s.t), i, target=u), ConstraintLaw(u, s.a.v - s.b.v, target=i)]


class VoltageSourceInteraction(BranchInteraction):
    """Source of the constant voltage `V`."""

    def __init__(self, label: str, *, a: Role, b: Role, branch: Role, V: float = 1.0) -> None:
        super().__init__(label, a=a, b=b, branch=branch)
        self.V = V

    def _get_branch_laws(self, s):
        """Return the voltage of the source."""
        return [ConstraintLaw(s.a.v - s.b.v, s.V, target=s.branch.i)]


class CurrentSourceInteraction(BranchInteraction):
    """Source that drives the constant current `current` out of its node `a`."""

    def __init__(self, label: str, *, a: Role, b: Role, branch: Role, current: float = 1.0) -> None:
        super().__init__(label, a=a, b=b, branch=branch)
        self.current = current

    def _get_branch_laws(self, s):
        """Return the current of the source."""
        return [ConstraintLaw(s.branch.i, -s.current, target=s.branch.i)]


class DiodeInteraction(BranchInteraction):
    """Exponential diode."""

    def __init__(self, label: str, *, a: Role, b: Role, branch: Role, Is: float = 1e-6, Vt: float = 0.025) -> None:
        super().__init__(label, a=a, b=b, branch=branch)
        self.Is = Is
        self.Vt = Vt

    def _get_branch_laws(self, s):
        """Return the characteristic of the diode."""
        return [ConstraintLaw(s.branch.i, s.Is * (exp((s.a.v - s.b.v) / s.Vt) - 1), target=s.branch.i)]


class GroundInteraction(Interaction):
    """Reference that holds a node at zero potential, feeding the current `i` of its branch into the node."""

    def __init__(self, label: str, *, node: Role, branch: Role) -> None:
        super().__init__(label)
        self.node = node
        self.branch = branch

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the zero potential and the current into the balance of the node."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            i = context.branch.i
            self.laws_ = [ConstraintLaw(context.node.v, 0, target=i), ConstraintLaw(i, 0, target=context.node.v)]


def _nodes(*labels):
    """Return a node of zero potential for every label."""
    return [Element(label, values={'v': 0.0}) for label in labels]


def _branches(*labels, **values):
    """Return a branch of zero current, and of further values, for every label."""
    return [Element(label, values={'i': 0.0, **values}) for label in labels]


def _ground(node):
    """Return the elements and the interaction of a ground at a node."""
    return _branches('ground'), GroundInteraction('grounding', node=node, branch='ground')


def _driven(source):
    """Return a resistor between the nodes of a source and a ground."""
    ground, grounding = _ground('bottom')
    clock, tick = _ticking()
    return System(
        elements=[*_nodes('top', 'bottom'), *_branches('source', 'load'), *ground, clock],
        interactions=[
            source,
            ResistorInteraction('resistor', a='top', b='bottom', branch='load', R=4.0),
            grounding,
            tick,
        ],
    )


def _ticking():
    """Return a clock with a zero rate, which makes its system run in continuous time and record every frame."""
    tick = Interaction('ticking', roles={'target': 'clock'}, laws=lambda s: [RateLaw(s.target.n, 0.0)])
    return Element('clock', values={'n': 0.0}), tick


def _run(system, until=1.0, dt=0.25):
    """Return the trajectory of a system."""
    return system.prepare().simulate(until=until, dt=dt)


def _issue(system, code):
    """Return the issue of a code that the compiler reports for a system."""
    issues = PassCompiler().prepare(system).report_.issues
    return next(issue for issue in issues if issue.code == code)


def _watcher(nonlinear=False, *interactions, x=0.0):
    """Return an element that follows an algebraic value, with its equation and a rate law that reads it."""

    def laws(s):
        w = s.target
        equation = (
            ConstraintLaw(w.y + exp(w.y), 1 + s.t, target=w.y) if nonlinear else ConstraintLaw(w.y, 2 + s.t, target=w.y)
        )
        return [equation, RateLaw(w.x, w.y)]

    return System(
        elements=[Element('watch', values={'x': x, 'y': 0.0})],
        interactions=[Interaction('watching', roles={'target': 'watch'}, laws=laws), *interactions],
    )


def test_same_resistor_solves_for_the_current_or_the_voltage():
    """Test that a resistor gives the current under a voltage source and the voltage under a current source."""
    voltage = _run(_driven(VoltageSourceInteraction('supply', a='top', b='bottom', branch='source', V=2.0)))
    current = _run(_driven(CurrentSourceInteraction('supply', a='top', b='bottom', branch='source', current=2.0)))
    assert np.allclose(voltage.get_values('load.i').ravel(), 0.5)
    assert np.allclose(voltage.get_values('top.v').ravel(), 2.0)
    assert np.allclose(current.get_values('top.v').ravel(), 8.0)
    assert np.allclose(current.get_values('load.i').ravel(), 2.0)


def test_divider_records_the_voltage_of_its_middle_node_at_every_frame():
    """Test that the trajectory holds the divided voltage of the middle node at every frame."""
    ground, grounding = _ground('bottom')
    clock, tick = _ticking()
    system = System(
        elements=[*_nodes('top', 'middle', 'bottom'), *_branches('source', 'upper', 'lower'), *ground, clock],
        interactions=[
            tick,
            VoltageSourceInteraction('supply', a='top', b='bottom', branch='source', V=12.0),
            ResistorInteraction('upper_resistor', a='top', b='middle', branch='upper', R=1.0),
            ResistorInteraction('lower_resistor', a='middle', b='bottom', branch='lower', R=3.0),
            grounding,
        ],
    )
    trajectory = _run(system)
    assert trajectory.times.size == 5
    assert np.allclose(trajectory.get_values('middle.v').ravel(), 9.0)


def test_node_of_one_component_has_a_zero_flow():
    """Test that the current of a component alone at a node is zero, which fixes the potential it gives the node."""

    def laws(s):
        i = s.branch.i
        return [ConstraintLaw(i, 2.0 * (s.a.v - 1.0), target=i), ConstraintLaw(i, 0, target=s.a.v)]

    channel = Interaction('conductance', roles={'a': 'a', 'branch': 'channel'}, laws=laws)
    system = System(elements=[*_nodes('a'), *_branches('channel')], interactions=[channel])
    trajectory = _run(system)
    assert np.allclose(trajectory.get_values('channel.i').ravel(), 0.0)
    assert np.allclose(trajectory.get_values('a.v').ravel(), 1.0)


def test_rate_law_reads_the_solution_of_an_algebraic_state():
    """Test that a rate law reads an algebraic state as the solution of the equation of the state."""
    trajectory = _run(_watcher(), until=1.0, dt=0.5)
    assert np.allclose(trajectory.get_values('watch.y').ravel(), [2.0, 2.5, 3.0])
    assert trajectory.get_values('watch.x')[-1] == pytest.approx(2.5, rel=1e-4)


def test_rate_law_reads_a_nonlinear_solution_through_a_call():
    """Test that a rate law reads an algebraic state of a nonlinear block, which the run solves at every evaluation."""
    system = _watcher(nonlinear=True)
    assert system.prepare()._problem.has_callables
    trajectory = _run(system, until=1.0, dt=0.5)
    expected = [brentq(lambda y, t=t: y + np.exp(y) - 1 - t, -5, 5, xtol=1e-14) for t in (0.0, 0.5, 1.0)]
    assert np.allclose(trajectory.get_values('watch.y').ravel(), expected, rtol=0, atol=1e-8)
    assert trajectory.get_values('watch.x')[-1] > 0.0


def test_capacitor_discharges_through_a_nonlinear_block():
    """Test that a capacitor discharging through a resistor and a diode follows the equation written by hand."""
    R, C, Is, Vt = 100.0, 1e-3, 1e-6, 0.025
    ground, grounding = _ground('bottom')
    system = System(
        elements=[
            *_nodes('top', 'middle', 'bottom'),
            *_branches('tank', u=1.0),
            *_branches('resistor', 'diode'),
            *ground,
        ],
        interactions=[
            CapacitorInteraction('capacitor', a='top', b='bottom', branch='tank', C=C),
            ResistorInteraction('resistance', a='top', b='middle', branch='resistor', R=R),
            DiodeInteraction('rectifier', a='middle', b='bottom', branch='diode', Is=Is, Vt=Vt),
            grounding,
        ],
    )
    trajectory = _run(system, until=0.2, dt=0.05)

    def current(u):
        return brentq(lambda i: Is * (np.exp((u - R * i) / Vt) - 1) - i, 0.0, max(u / R, 1e-12), xtol=1e-14)

    solution = solve_ivp(lambda _, u: -current(u[0]) / C, (0.0, 0.2), [1.0], t_eval=trajectory.times, rtol=1e-8)
    assert np.allclose(trajectory.get_values('tank.u').ravel(), solution.y[0], rtol=1e-3)


def test_large_linear_block_is_solved_numerically():
    """Test that a resistive ladder whose block exceeds six unknowns matches the solve of its nodal equations."""
    sections = 12
    nodes = [f'node{index}' for index in range(sections + 1)]
    resistors = [
        ResistorInteraction(
            f'series_resistor{index}',
            a=nodes[index],
            b=nodes[index + 1],
            branch=f'series{index}',
            R=1.0 + index,
        )
        for index in range(sections)
    ]
    resistors += [
        ResistorInteraction(
            f'shunt_resistor{index}',
            a=nodes[index + 1],
            b='bottom',
            branch=f'shunt{index}',
            R=2.0 + index,
        )
        for index in range(sections)
    ]
    branches = [*(f'series{index}' for index in range(sections)), *(f'shunt{index}' for index in range(sections))]
    ground, grounding = _ground('bottom')
    system = System(
        elements=[*_nodes(*nodes, 'bottom'), *_branches('source', *branches), *ground],
        interactions=[
            VoltageSourceInteraction('supply', a='node0', b='bottom', branch='source', V=10.0),
            grounding,
            *resistors,
        ],
    )
    matrix = np.zeros((sections, sections))
    right = np.zeros(sections)
    right[0] = 10.0 / 1.0
    for index in range(sections):
        matrix[index, index] += 1 / (1.0 + index) + 1 / (2.0 + index)
        if index + 1 < sections:
            matrix[index, index] += 1 / (2.0 + index)
            matrix[index, index + 1] -= 1 / (2.0 + index)
            matrix[index + 1, index] -= 1 / (2.0 + index)
    voltages = np.linalg.solve(matrix, right)
    trajectory = _run(system)
    for index, voltage in enumerate(voltages):
        assert np.allclose(trajectory.get_values(f'node{index + 1}.v').ravel(), voltage)


def test_structurally_regular_singular_block_raises_an_error():
    """Test that a block whose matrix is singular for all values raises an error that names its unknowns."""

    def laws(s):
        twin = s.target
        return [
            ConstraintLaw(twin.x + twin.y, 1.0, target=twin.x),
            ConstraintLaw(2 * twin.x + 2 * twin.y, 3.0, target=twin.y),
        ]

    clock, tick = _ticking()
    system = System(
        elements=[Element('twin', values={'x': 0.0, 'y': 0.0}), clock],
        interactions=[Interaction('equations', roles={'target': 'twin'}, laws=laws), tick],
    )
    with pytest.raises(ValidationError, match='the system is singular'):
        system.prepare()


def test_nonlinear_block_without_a_solution_raises_an_error():
    """Test that a nonlinear equation with no real solution raises a solver error that names its unknown."""
    hopeless = Interaction(
        'equation',
        roles={'target': 'hopeless'},
        laws=lambda s: [ConstraintLaw(s.target.y**2, -1.0, target=s.target.y)],
    )
    with pytest.raises(SolverError, match='did not converge'):
        _run(System(elements=[Element('hopeless', values={'y': 1.0})], interactions=[hopeless]))


def test_guard_reads_the_symbolic_solution_of_an_algebraic_state():
    """Test that the guard of an event reads an algebraic state as its solution."""
    stop = Interaction('stop', roles={'target': 'watch'}, laws=lambda s: [EventLaw(s.target.y - 3.0, terminal=True)])
    trajectory = _run(_watcher(False, stop), until=5.0, dt=0.25)
    assert trajectory.times[-1] == pytest.approx(1.0, abs=1e-3)


def test_guard_of_a_nonlinear_solution_raises_an_error():
    """Test that a guard that reads an algebraic state of a nonlinear block raises an error."""
    stop = Interaction('stop', roles={'target': 'watch'}, laws=lambda s: [EventLaw(s.target.y - 3.0)])
    with pytest.raises(ValidationError, match='a nonlinear solve determines these algebraic states'):
        _watcher(True, stop).prepare()


def test_rate_law_on_an_algebraic_state_raises_a_named_error():
    """Test that a law that integrates an algebraic state raises an error that names it and its equations."""
    laws = Interaction(
        'integration',
        roles={'target': 'integrated'},
        laws=lambda s: [ConstraintLaw(s.target.x, 0.0, target=s.target.x), RateLaw(s.target.x, 1.0)],
    )
    system = System(elements=[Element('integrated', values={'x': 0.0})], interactions=[laws])
    with pytest.raises(ValidationError, match=r'x_integrated == 0\.0 of integration contains no highest derivative'):
        system.prepare()


def test_parallel_capacitors_report_a_structural_singularity_before_compilation():
    """Test that two capacitors in parallel report `structural-singularity`, which names their voltages."""
    ground, grounding = _ground('bottom')
    system = System(
        elements=[*_nodes('top', 'bottom'), *_branches('first', 'second', u=1.0), *ground],
        interactions=[
            CapacitorInteraction('first_capacitor', a='top', b='bottom', branch='first'),
            CapacitorInteraction('second_capacitor', a='top', b='bottom', branch='second'),
            grounding,
        ],
    )
    compiler = PassCompiler().prepare(system)
    issue = next(item for item in compiler.report_.issues if item.code == 'structural-singularity')
    assert 'first.u' in issue.message
    assert 'second.u' in issue.message
    assert 'resistance' in issue.fix
    assert compiler.compilation_.problem is None


def test_coupled_problem_records_the_algebraic_states_of_its_partition():
    """Test that a circuit in a nested system next to agents records its algebraic states at every frame."""
    ground, grounding = _ground('bottom')
    circuit = System(
        'circuit',
        elements=[*_nodes('top', 'bottom'), *_branches('tank', u=2.0), *_branches('load'), *ground],
        interactions=[
            CapacitorInteraction('capacitor', a='top', b='bottom', branch='tank'),
            ResistorInteraction('resistor', a='top', b='bottom', branch='load', R=4.0),
            grounding,
        ],
    )
    waiting = Interaction(
        'waiting',
        roles={'target': 'farmers'},
        laws=lambda s: [UpdateLaw(s.target, lambda agent, system: None)],
    )
    agents = System('village', elements=[Element('farmers', values={'x': 0.0}, size=2)], interactions=[waiting])
    trajectory = _run(System(systems=[circuit, agents]), until=2, dt=1)
    assert trajectory.problem.form == FormKind.COUPLED
    assert np.allclose(trajectory.get_values('circuit/top.v').ravel(), trajectory.get_values('circuit/tank.u').ravel())
    assert trajectory.get_values('circuit/load.i')[-1] == pytest.approx(0.5 * np.exp(-0.5), rel=1e-3)


def test_noise_law_reads_a_symbolic_solution():
    """Test that a noise law reads an algebraic state as its solution."""
    laws = Interaction(
        'noise',
        roles={'target': 'watch'},
        laws=lambda s: [ConstraintLaw(s.target.y, 0.0, target=s.target.y), NoiseLaw(s.target.x, s.target.y)],
    )
    system = System(elements=[Element('watch', values={'x': 1.0, 'y': 0.0})], interactions=[laws])
    trajectory = system.prepare().simulate(until=0.5, dt=0.1, random_state=0)
    assert np.allclose(trajectory.get_values('watch.x').ravel(), 1.0)


class SquareInteraction(Interaction):
    """Constraint `y = a x^2` of every point of a group."""

    def __init__(self, label: str, *, target: Role = 'points', a: float = 2.0) -> None:
        super().__init__(label)
        self.target = target
        self.a = a

    def _prepare(self, context):
        """Set the constraint of every point."""
        super()._prepare(context)
        self.laws_ = [ConstraintLaw(context.target.y == context.a * context.target.x**2, target=context.target.y)]


class ShiftInteraction(Interaction):
    """Step `x <- x + 1` of every point, or the growth `x' = x` with `rate=True`."""

    def __init__(self, label: str, *, target: Role = 'points', rate: bool = False) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _prepare(self, context):
        """Set the step or the growth of every point."""
        super()._prepare(context)
        x = context.target.x
        self.laws_ = [RateLaw(x.diff(context.t), x) if self.rate else UpdateLaw(x, x + 1)]


@pytest.mark.parametrize(
    ('rate', 'expected'),
    [(False, 2 * np.array([3.0, 4.0, 5.0]) ** 2), (True, 2 * (np.exp(2.0) * np.array([1.0, 2.0, 3.0])) ** 2)],
)
def test_constraint_of_a_group_holds_for_every_member(rate, expected):
    """Test that a constraint of an interaction on a group holds for every point, in a map and in a DAE."""
    points = Element('points', values={'x': np.array([1.0, 2.0, 3.0]), 'y': 0.0}, size=3)
    system = System(elements=[points], interactions=[SquareInteraction('square'), ShiftInteraction('shift', rate=rate)])
    trajectory = system.prepare().simulate(until=2.0)
    np.testing.assert_allclose(trajectory.get_values('points.y')[-1], expected, rtol=1e-5)


def test_constraint_of_a_group_holds_for_every_member_of_a_static_system():
    """Test that a static system solves the constraint of every point."""
    points = Element('points', values={'x': np.array([1.0, 2.0, 3.0]), 'y': 0.0}, size=3)
    system = System(elements=[points], interactions=[SquareInteraction('square')]).prepare()
    assert system.form_ == 'static solution'
    np.testing.assert_allclose(system.simulate(until=1.0).get_values('points.y')[-1], [2.0, 8.0, 18.0])
