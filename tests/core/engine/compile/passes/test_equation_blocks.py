"""Test the `_equation_blocks` module of `skcomplex.core.engine.compile.passes`, through the systems it compiles."""

from typing import Annotated, ClassVar

import numpy as np
import pytest
from sage.all__sagemath_symbolics import exp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import ConnectionInteraction, CrossingInteraction
from skcomplex.core.components.laws import EquationLaw, NoiseLaw, RateLaw, RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import AlgebraicStateError, SolverError, ValidationError
from skcomplex.core.specs import ConnectorSpec, ParameterSpec, StateSpec
from skcomplex.core.types import FormKind
from skcomplex.systems.engineering.common.electrical import (
    ELECTRICAL,
    CapacitorElement,
    CurrentSourceElement,
    DiodeElement,
    GroundElement,
    ResistorElement,
    VoltageSourceElement,
)
from skcomplex.systems.engineering.control import THERMAL, ConvectionElement

PIN = ConnectorSpec('pin', potential='v', flow='i')


class IntegratedElement(Element):
    """Element with a rate law on an algebraic state."""

    def __init__(self, label: str, x: Annotated[float, StateSpec(algebraic=True)] = 0.0) -> None:
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        """Return a rate law on the algebraic state."""
        return [RateLaw(s.x, 1.0)]


class MissingStateElement(Element):
    """Element that declares a port without the state of its flow."""

    ports: ClassVar = {'p': ELECTRICAL}

    def __init__(self, label: str, p_v: Annotated[float, StateSpec(algebraic=True)] = 0.0) -> None:
        super().__init__(label)
        self.p_v = p_v

    def _get_laws(self, s):
        """Return the potential of the port."""
        return [EquationLaw(s.p_v, 0)]


class WatcherElement(Element):
    """Element that follows an algebraic value, with an equation of its own and a rate law that reads it."""

    def __init__(
        self,
        label: str,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        nonlinear: bool = False,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.nonlinear = nonlinear

    def _get_laws(self, s):
        """Return the equation of the algebraic value and the rate of the state that reads it."""
        equation = EquationLaw(s.y + exp(s.y), 1 + s.t) if self.nonlinear else EquationLaw(s.y, 2 + s.t)
        return [equation, RateLaw(s.x, s.y)]


class VectorWatcherElement(Element):
    """Element whose noise reads an algebraic value."""

    def __init__(
        self,
        label: str,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec(algebraic=True)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y

    def _get_laws(self, s):
        """Return the equation of the algebraic value and the noise it scales."""
        return [EquationLaw(s.y, 0.0), NoiseLaw(s.x, s.y)]


class OpenChannelElement(Element):
    """Element with one port whose flow an equation does not fix, for a port no connection names."""

    ports: ClassVar = {'a': PIN}

    def __init__(
        self,
        label: str,
        *,
        a_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        a_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        g: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.a_v = a_v
        self.a_i = a_i
        self.g = g

    def _get_laws(self, s):
        """Return the current through the conductance of the channel from a potential of one."""
        return [EquationLaw(s.a_i, s.g * (s.a_v - 1.0))]


def _connected(*nodes):
    """Return one connection per node of pairs of an element and a port."""
    return [
        ConnectionInteraction(f'node{index}', members=[item for item, _ in node], ports=[port for _, port in node])
        for index, node in enumerate(nodes)
    ]


def _driven(source):
    """Return a resistor between the pins of a source and a ground."""
    return System(
        elements=[source, ResistorElement('load', R=4.0), GroundElement('ground')],
        interactions=_connected(
            [('source', 'p'), ('load', 'p')],
            [('source', 'n'), ('load', 'n'), ('ground', 'p')],
        ),
    )


def _run(system, until=1.0, dt=0.25):
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt).prepare().run(until=until).trajectory_


def _codes(system):
    """Return the codes of the issues the compiler reports for a system."""
    return [issue.code for issue in PassCompiler().prepare(system).report_.issues]


def _issue(system, code):
    """Return the issue of a code that the compiler reports for a system."""
    issues = PassCompiler().prepare(system).report_.issues
    return next(issue for issue in issues if issue.code == code)


def test_same_resistor_solves_for_the_current_or_the_voltage():
    """Test that a resistor gives the current under a voltage source and the voltage under a current source."""
    voltage = _run(_driven(VoltageSourceElement('source', V=2.0)))
    current = _run(_driven(CurrentSourceElement('source', current=2.0)))
    assert np.allclose(voltage.values('load.p_i').ravel(), 0.5)
    assert np.allclose(voltage.values('load.p_v').ravel(), 2.0)
    assert np.allclose(current.values('load.p_v').ravel(), 8.0)
    assert np.allclose(current.values('load.p_i').ravel(), 2.0)


def test_divider_records_the_voltage_of_its_middle_node_at_every_frame():
    """Test that the trajectory holds the divided voltage of the middle node at every frame."""
    system = System(
        elements=[
            VoltageSourceElement('source', V=12.0),
            ResistorElement('upper', R=1.0),
            ResistorElement('lower', R=3.0),
            GroundElement('ground'),
        ],
        interactions=_connected(
            [('source', 'p'), ('upper', 'p')],
            [('upper', 'n'), ('lower', 'p')],
            [('lower', 'n'), ('source', 'n'), ('ground', 'p')],
        ),
    )
    trajectory = _run(system)
    assert trajectory.times.size == 5
    assert np.allclose(trajectory.values('lower.p_v').ravel(), 9.0)
    assert np.allclose(trajectory.values('upper.n_v').ravel(), 9.0)


def test_port_without_a_connection_has_a_zero_flow():
    """Test that the flow of an unconnected port is zero, which fixes the potential the element gives it."""
    trajectory = _run(System(elements=[OpenChannelElement('channel')]))
    assert np.allclose(trajectory.values('channel.a_i').ravel(), 0.0)
    assert np.allclose(trajectory.values('channel.a_v').ravel(), 1.0)


def test_rate_law_reads_the_solution_of_an_algebraic_state():
    """Test that a rate law of an element reads its algebraic state as the solution of the equation of the state."""
    trajectory = _run(System(elements=[WatcherElement('watch')]), until=1.0, dt=0.5)
    assert np.allclose(trajectory.values('watch.y').ravel(), [2.0, 2.5, 3.0])
    assert trajectory.values('watch.x')[-1] == pytest.approx(2.5, rel=1e-4)


def test_rate_law_reads_a_nonlinear_solution_through_a_call():
    """Test that a rate law reads an algebraic state of a nonlinear block, which the run solves at every evaluation."""
    system = System(elements=[WatcherElement('watch', nonlinear=True)])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.has_callables
    trajectory = _run(system, until=1.0, dt=0.5)
    expected = [brentq(lambda y, t=t: y + np.exp(y) - 1 - t, -5, 5, xtol=1e-14) for t in (0.0, 0.5, 1.0)]
    assert np.allclose(trajectory.values('watch.y').ravel(), expected, rtol=0, atol=1e-8)
    assert trajectory.values('watch.x')[-1] > 0.0


def test_capacitor_discharges_through_a_nonlinear_block():
    """Test that a capacitor discharging through a resistor and a diode follows the equation written by hand."""
    R, C, Is, Vt = 100.0, 1e-3, 1e-6, 0.025
    system = System(
        elements=[
            CapacitorElement('tank', C=C, u=1.0),
            ResistorElement('resistor', R=R),
            DiodeElement('diode', Is=Is, Vt=Vt),
            GroundElement('ground'),
        ],
        interactions=_connected(
            [('tank', 'p'), ('resistor', 'p')],
            [('resistor', 'n'), ('diode', 'p')],
            [('diode', 'n'), ('tank', 'n'), ('ground', 'p')],
        ),
    )
    trajectory = _run(system, until=0.2, dt=0.05)

    def current(u):
        return brentq(lambda i: Is * (np.exp((u - R * i) / Vt) - 1) - i, 0.0, max(u / R, 1e-12), xtol=1e-14)

    solution = solve_ivp(lambda _, u: -current(u[0]) / C, (0.0, 0.2), [1.0], t_eval=trajectory.times, rtol=1e-8)
    assert np.allclose(trajectory.values('tank.u').ravel(), solution.y[0], rtol=1e-3)


def test_large_linear_block_is_solved_numerically():
    """Test that a resistive ladder whose block exceeds six unknowns matches the solve of its nodal equations."""
    sections = 12
    resistors = [ResistorElement(f'series{index}', R=1.0 + index) for index in range(sections)]
    resistors += [ResistorElement(f'shunt{index}', R=2.0 + index) for index in range(sections)]
    nodes = [[('source', 'p'), ('series0', 'p')]]
    for index in range(sections):
        following = [(f'series{index + 1}', 'p')] if index + 1 < sections else []
        nodes.append([(f'series{index}', 'n'), (f'shunt{index}', 'p'), *following])
    nodes.append(
        [('source', 'n'), ('ground', 'p'), *((f'shunt{index}', 'n') for index in range(sections))],
    )
    system = System(
        elements=[VoltageSourceElement('source', V=10.0), GroundElement('ground'), *resistors],
        interactions=_connected(*nodes),
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
        assert np.allclose(trajectory.values(f'shunt{index}.p_v').ravel(), voltage)


def test_structurally_regular_singular_block_raises_an_error():
    """Test that a block whose matrix is singular for all values raises an error that names its unknowns."""

    class TwinElement(Element):
        """Element with two equations that are the same in two algebraic states."""

        def __init__(
            self,
            label: str,
            x: Annotated[float, StateSpec(algebraic=True)] = 0.0,
            y: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        ) -> None:
            super().__init__(label)
            self.x = x
            self.y = y

        def _get_laws(self, s):
            """Return two dependent equations."""
            return [EquationLaw(s.x + s.y, 1.0), EquationLaw(2 * s.x + 2 * s.y, 3.0)]

    with pytest.raises(ValidationError, match='the system is singular'):
        PassCompiler().prepare(System(elements=[TwinElement('twin')])).run()


def test_nonlinear_block_without_a_solution_raises_an_error():
    """Test that a nonlinear equation with no real solution raises a solver error that names its unknown."""

    class HopelessElement(Element):
        """Element whose nonlinear equation has no real solution."""

        def __init__(self, label: str, y: Annotated[float, StateSpec(algebraic=True)] = 1.0) -> None:
            super().__init__(label)
            self.y = y

        def _get_laws(self, s):
            """Return an equation with no real solution."""
            return [EquationLaw(s.y**2, -1.0)]

    with pytest.raises(SolverError, match='have no solution near the last values'):
        _run(System(elements=[HopelessElement('hopeless')]))


def test_guard_reads_the_symbolic_solution_of_an_algebraic_state():
    """Test that the guard of an event reads an algebraic state as its solution."""
    system = System(
        elements=[WatcherElement('watch')],
        interactions=[CrossingInteraction('stop', target='watch', guard=lambda s: s.target.y - 3.0, terminal=True)],
    )
    trajectory = Simulation(system, dt=0.25).prepare().run(until=5.0).trajectory_
    assert trajectory.times[-1] == pytest.approx(1.0, abs=1e-3)


def test_guard_of_a_nonlinear_solution_raises_an_error():
    """Test that a guard that reads an algebraic state of a nonlinear block raises an error."""
    system = System(
        elements=[WatcherElement('watch', nonlinear=True)],
        interactions=[CrossingInteraction('stop', target='watch', guard=lambda s: s.target.y - 3.0)],
    )
    with pytest.raises(ValidationError, match='a nonlinear solve determines these algebraic states'):
        PassCompiler().prepare(system).run()


def test_rate_law_on_an_algebraic_state_raises_a_named_error():
    """Test that a law that integrates an algebraic state raises an error that names it and its equations."""
    with pytest.raises(AlgebraicStateError, match=r'integrated\.x.*equations of the system determine'):
        PassCompiler().prepare(System(elements=[IntegratedElement('integrated')])).run()


def test_connector_mismatch_names_both_ports_and_their_connectors():
    """Test that connecting an electrical port to a thermal port reports `port-connector`."""
    system = System(
        elements=[ResistorElement('load'), ConvectionElement('surface'), GroundElement('ground')],
        interactions=_connected([('load', 'p'), ('surface', 'a')], [('load', 'n'), ('ground', 'p')]),
    )
    issue = _issue(system, 'port-connector')
    assert 'load.p' in issue.message
    assert 'surface.a' in issue.message
    assert ELECTRICAL.name in issue.message
    assert THERMAL.name in issue.message


def test_connection_of_an_undeclared_port_reports_a_connector_issue():
    """Test that a connection of a port the element does not declare reports `port-connector`."""
    system = System(
        elements=[ResistorElement('load'), GroundElement('ground')],
        interactions=_connected([('load', 'missing'), ('ground', 'p')]),
    )
    assert 'load.missing' in _issue(system, 'port-connector').message


def test_shared_port_tells_to_list_the_node_in_one_connection():
    """Test that a port in two connections reports `port-connections`."""
    system = System(
        elements=[ResistorElement('load'), ResistorElement('other'), GroundElement('ground')],
        interactions=_connected(
            [('load', 'p'), ('other', 'p')],
            [('load', 'p'), ('ground', 'p')],
            [('load', 'n'), ('other', 'n')],
        ),
    )
    issue = _issue(system, 'port-connections')
    assert 'load.p' in issue.message
    assert issue.fix == 'List the ports of the node in one connection.'


def test_missing_port_state_names_the_state():
    """Test that a declared port without the state of its flow reports `port-states`, which names it."""
    assert 'p_i' in _issue(System(elements=[MissingStateElement('broken')]), 'port-states').message


def test_parallel_capacitors_report_a_structural_singularity_before_compilation():
    """Test that two capacitors in parallel report `structural-singularity`, which names their voltages."""
    system = System(
        elements=[CapacitorElement('first', u=1.0), CapacitorElement('second', u=1.0), GroundElement('ground')],
        interactions=_connected(
            [('first', 'p'), ('second', 'p')],
            [('first', 'n'), ('second', 'n'), ('ground', 'p')],
        ),
    )
    compiler = PassCompiler().prepare(system)
    issue = next(item for item in compiler.report_.issues if item.code == 'structural-singularity')
    assert 'first.u' in issue.message
    assert 'second.u' in issue.message
    assert 'resistance' in issue.fix
    assert compiler.compilation_.problem is None


def test_algebraic_state_without_an_equation_reports_a_structural_singularity():
    """Test that an algebraic state that no equation holds reports `structural-singularity`, which names it."""

    class IdleElement(Element):
        """Element with an algebraic state and no equation."""

        def __init__(self, label: str, y: Annotated[float, StateSpec(algebraic=True)] = 0.0) -> None:
            super().__init__(label)
            self.y = y

    assert 'idle.y' in _issue(System(elements=[IdleElement('idle')]), 'structural-singularity').message


def test_port_under_discrete_time_reports_the_law_time_issue():
    """Test that a connected port under discrete time gives the existing `law-time` issue."""
    system = _driven(VoltageSourceElement('source', V=2.0))
    system.time = DiscreteTime()
    assert 'law-time' in _codes(system)


def test_coupled_problem_records_the_algebraic_states_of_its_partition():
    """Test that a circuit in a nested system next to agents records its algebraic states at every frame."""

    class WaitingElement(Element):
        """Agent that does nothing."""

        def __init__(self, label: str, x: Annotated[float, StateSpec()] = 0.0) -> None:
            super().__init__(label)
            self.x = x

        def _get_laws(self, s):
            """Return a rule that does nothing."""
            return [RuleLaw(lambda agent, system: None)]

    circuit = System(
        'circuit',
        elements=[CapacitorElement('tank', u=2.0), ResistorElement('load', R=4.0), GroundElement('ground')],
        interactions=_connected(
            [('tank', 'p'), ('load', 'p')],
            [('tank', 'n'), ('load', 'n'), ('ground', 'p')],
        ),
    )
    agents = System('village', elements=[GroupElement('farmers', element=WaitingElement, size=2)])
    system = System(systems=[circuit, agents])
    simulation = Simulation(system, dt=1).prepare()
    assert simulation.problem_.form == FormKind.COUPLED
    trajectory = simulation.run(until=2).trajectory_
    assert np.allclose(trajectory.values('circuit/load.p_v').ravel(), trajectory.values('circuit/tank.u').ravel())
    assert trajectory.values('circuit/load.p_i')[-1] == pytest.approx(0.5 * np.exp(-0.5), rel=1e-3)


def test_noise_law_reads_a_symbolic_solution():
    """Test that a noise law reads an algebraic state as its solution."""
    system = System(elements=[VectorWatcherElement('watch', x=1.0)])
    trajectory = Simulation(system, dt=0.1, random_state=0).prepare().run(until=0.5).trajectory_
    assert np.allclose(trajectory.values('watch.x').ravel(), 1.0)
