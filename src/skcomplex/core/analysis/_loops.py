"""Derive the feedback loops of systems of stocks and flows, their dominance, and their transaction-flow matrices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import AnalysisError, ConservationError, Expression, symbols_of
from ..compile import LoopTerm, Problem, StockFlow, compile_system, numpy_function
from ..model import System
from ..model._flows import DELAY_ORDERS
from ..runtime import Trajectory

POLARITIES = {1: 'reinforcing', -1: 'balancing', 0: 'neutral'}


@dataclass(frozen=True)
class Loop:
    """Feedback loop of a system, a cycle of the signed dependency graph.

    Args:
        nodes:
            The states, auxiliaries, and flows of the loop, in the order of their influence, starting at its first
            state by name, or at its first node by name when it holds no state.

        polarity:
            `'reinforcing'` for a positive product of edge signs, `'balancing'` for a negative one, or `'neutral'`
            when an edge has no influence at the point of evaluation.

        gain:
            The product of the partial derivatives along the loop at the point of evaluation.
    """

    nodes: tuple[str, ...]
    polarity: str
    gain: float


@dataclass(frozen=True)
class Loops:
    """Signed dependency graph of a system and its feedback loops.

    Args:
        graph:
            The directed graph from each state, auxiliary, or flow to every node whose value or rate it enters, with
            the edge attributes `sign` and `gain`.

        loops:
            The feedback loops.
    """

    graph: nx.DiGraph
    loops: tuple[Loop, ...]


@dataclass(frozen=True)
class LoopDominance:
    """Loop gains along a trajectory.

    Args:
        times:
            The time of every frame.

        loops:
            The loops, with their polarity at the first frame.

        gains:
            The gain of every loop at every frame, one row per frame.

        dominant:
            The index of the loop with the largest absolute gain at every frame.
    """

    times: npt.NDArray[np.float64]
    loops: tuple[Loop, ...]
    gains: npt.NDArray[np.float64]
    dominant: npt.NDArray[np.intp]


@dataclass(frozen=True)
class FlowMatrix:
    """Transaction-flow matrix of one frame, with flows as rows and holders as columns, whose rows sum to zero.

    Args:
        flows:
            The paths of the flows.

        holders:
            The states and environments the flows connect.

        values:
            The outflow, negative, and the inflow, positive, of every flow for every holder.
    """

    flows: tuple[str, ...]
    holders: tuple[str, ...]
    values: npt.NDArray[np.float64]


def _stock_flow(problem: Problem) -> StockFlow:
    """Return the stock-flow description of a problem, or raise an error for problems without one."""
    if problem.stock_flow is None:
        error_msg = (
            'The problem has no stock-flow description. Compile the system as one partition with compile_system.'
        )
        raise AnalysisError(error_msg)
    return problem.stock_flow


def _without_delays(expression: Expression) -> Expression:
    """Return an expression with every delay and smoothing replaced by its input, which it follows with gain one."""
    calls = {
        item: item.operands()[0]
        for item in _operations(expression)
        if getattr(item.operator(), 'name', lambda: '')() in DELAY_ORDERS
    }
    return _without_delays(SR(expression).subs(calls)) if calls else SR(expression)


def _operations(expression: Expression) -> list[Expression]:
    """Return every subexpression of an expression that applies an operator."""
    found, stack = [], [SR(expression)]
    while stack:
        item = stack.pop()
        if item.operator() is not None:
            found.append(item)
            stack.extend(item.operands())
    return found


class _Graph:
    """Edges of the dependency graph of a problem with the expressions of their partial derivatives."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        stock_flow = _stock_flow(problem)
        self.readers: dict[Expression, tuple[bool, int]] = {}
        self.expanded: dict[Expression, Expression] = {}
        self.edges: dict[tuple[str, str], Expression] = defaultdict(lambda: SR(0))
        for term in stock_flow.terms:
            self._add(term)

    def _node_of(self, item: Expression, term: LoopTerm) -> str | None:
        """Return the node a symbol stands for, an auxiliary or a state, or `None` for parameters and time."""
        if item in term.auxiliaries:
            return term.auxiliaries[item]
        reference = term.references.get(item)
        if reference is None:
            return None
        if reference.entry.members is not None:
            error_msg = f'{term.path} reads the group variable {reference.entry.name}. Loops take single elements only.'
            raise AnalysisError(error_msg)
        self.readers[item] = (reference.is_state, reference.entry.start + reference.column)
        return reference.entry.name if reference.is_state else None

    def _add(self, term: LoopTerm) -> None:
        """Add the edges of one term, from every node its expression reads."""
        expression = _without_delays(term.expression)
        expanded = SR(expression).subs(self.expanded)
        if term.kind == 'auxiliary':
            self.expanded[next(item for item, path in term.auxiliaries.items() if path == term.path)] = expanded
        for item in symbols_of([expression]) - {self.problem.time}:
            node = self._node_of(item, term)
            if node is not None:
                self.edges[(node, term.path)] += SR(expression).diff(item).subs(self.expanded)
        for item in symbols_of([expanded]) - {self.problem.time}:
            self._node_of(item, term)
        if term.kind == 'flow':
            if term.sink is not None:
                self.edges[(term.path, term.sink)] += 1
            if term.source is not None:
                self.edges[(term.path, term.source)] -= 1

    def evaluate(
        self,
        expression: Expression,
        states: npt.NDArray[np.float64],
        parameters: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the values of an expression over frames of states and parameters."""
        arguments = sorted(symbols_of([expression]) - {self.problem.time}, key=str)
        unknown = [item for item in arguments if item not in self.readers]
        if unknown:
            error_msg = f'The loop gain {expression} reads {unknown[0]}, which no term of the system defines.'
            raise AnalysisError(error_msg)
        function = numpy_function(self.problem.time, arguments, [expression])
        values = [
            (states if is_state else parameters)[:, index]
            for is_state, index in map(self.readers.__getitem__, arguments)
        ]
        return np.broadcast_to(np.asarray(function(times, *values)[0], dtype=float), np.shape(times)).copy()


def _signed_loops(graph: _Graph, gains: Mapping[tuple[str, str], float]) -> tuple[nx.DiGraph, tuple[Loop, ...]]:
    """Return the signed graph with edge gains and its loops with polarity and gain."""
    directed = nx.DiGraph()
    for (start, end), derivative in graph.edges.items():
        if SR(derivative).is_trivial_zero():
            continue
        gain = gains[(start, end)]
        directed.add_edge(start, end, sign=int(np.sign(gain)), gain=gain)
    stocks = {entry.name for entry in graph.problem.states.entries}
    loops = []
    for found in nx.simple_cycles(directed):
        anchor = min((node for node in found if node in stocks), default=min(found))
        start = found.index(anchor)
        cycle = [*found[start:], *found[:start]]
        pairs = list(zip(cycle, [*cycle[1:], cycle[0]], strict=True))
        gain = float(np.prod([directed.edges[pair]['gain'] for pair in pairs]))
        loops.append(Loop(tuple(cycle), POLARITIES[int(np.sign(gain))], gain))
    return directed, tuple(sorted(loops, key=lambda loop: (len(loop.nodes), loop.nodes)))


def _problem_of(system: System | Problem) -> Problem:
    """Return the compiled problem of a system, or the problem itself."""
    return system if isinstance(system, Problem) else compile_system(system)


def loops(system: System | Problem, at: npt.ArrayLike | None = None) -> Loops:
    """Return the signed dependency graph of a system and its feedback loops.

    An edge goes from each state, auxiliary, or flow to every auxiliary, flow, or state rate whose expression reads
    it, with the sign of the partial derivative, and from each flow to its sink with sign `+` and its source with sign
    `-`. Delays and smoothings pass their input with gain one. The loops are the cycles of the graph, found with
    Johnson's algorithm, and a loop's polarity is the product of its edge signs.

    Args:
        system:
            The system, or its compiled problem.

        at:
            The state vector the signs are evaluated at, or `None` for the initial state.

    Returns:
        loops:
            The graph and the loops.

    Raises:
        AnalysisError:
            The system is partitioned, or a law reads a group.

    Examples:
        >>> from skcomplex.core.analysis import loops
        >>> from skcomplex.models import create
        >>> [loop.polarity for loop in loops(create('bass_diffusion')).loops]
        ['reinforcing', 'balancing']
    """
    problem = _problem_of(system)
    graph = _Graph(problem)
    state = np.asarray(problem.initial_state if at is None else at, dtype=float)[np.newaxis]
    parameters, times = problem.initial_parameters[np.newaxis], np.zeros(1)
    gains = {edge: float(graph.evaluate(SR(value), state, parameters, times)[0]) for edge, value in graph.edges.items()}
    return Loops(*_signed_loops(graph, gains))


def loop_dominance(trajectory: Trajectory) -> LoopDominance:
    """Return the gains of the loops of a system along a trajectory and the dominant loop at every frame.

    The gain of a loop is the product of the partial derivatives along it, evaluated at every frame, and the dominant
    loop has the largest absolute gain.

    Args:
        trajectory:
            The trajectory of a system with stocks and flows.

    Returns:
        dominance:
            The loops, their gains, and the dominant loop at every frame.

    Raises:
        AnalysisError:
            The system has no feedback loop.

    Examples:
        >>> from skcomplex.core.analysis import loop_dominance
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.models import create
        >>> trajectory = Simulation(create('bass_diffusion'), dt=1).run(2).trajectory_
        >>> loop_dominance(trajectory).gains.shape
        (3, 2)
    """
    problem = trajectory.problem
    graph = _Graph(problem)
    states, parameters = np.atleast_2d(trajectory.states), np.atleast_2d(trajectory.parameters)
    times = np.asarray(trajectory.times, dtype=float)
    values = {edge: graph.evaluate(SR(value), states, parameters, times) for edge, value in graph.edges.items()}
    _, found = _signed_loops(graph, {edge: float(value[0]) for edge, value in values.items()})
    if not found:
        error_msg = 'The system has no feedback loop, so no loop dominates.'
        raise AnalysisError(error_msg)
    gains = np.column_stack(
        [
            np.prod([values[pair] for pair in zip(loop.nodes, [*loop.nodes[1:], loop.nodes[0]], strict=True)], axis=0)
            for loop in found
        ],
    )
    return LoopDominance(times, found, gains, np.argmax(np.abs(gains), axis=1))


def conservation(system: System | Problem) -> dict[str, Expression]:
    """Return the net inflow from the environment of every quantity, after checking that internal flows cancel.

    The sum of the contributions of all flows to the states of a quantity must equal the sum of its flows with the
    environment, so the total changes only through the boundary.

    Args:
        system:
            The system, or its compiled problem.

    Returns:
        boundary:
            The symbolic rate, or amount per step, at which each quantity enters the system from the environment.

    Raises:
        ConservationError:
            The internal flows of a quantity do not cancel.

    Examples:
        >>> from skcomplex.core.analysis import conservation
        >>> from skcomplex.models import create
        >>> conservation(create('bathtub'))
        {'water': inflow_tub*where(t >= 5.0, 1.0, 0) - where(water_tub > 0, outflow_tub, 0)}
    """
    problem = _problem_of(system)
    stock_flow = _stock_flow(problem)
    amounts = {record.path: record.expression for record in stock_flow.recorded}
    net: dict[str, Expression] = defaultdict(lambda: SR(0))
    boundary: dict[str, Expression] = defaultdict(lambda: SR(0))
    for flow in stock_flow.flows:
        if flow.quantity is None:
            continue
        amount = SR(amounts[flow.path])
        net[flow.quantity] += (flow.sink is not None) * amount - (flow.source is not None) * amount
        boundary[flow.quantity] += (flow.source is None) * amount - (flow.sink is None) * amount
    for quantity, value in net.items():
        if not bool((value - boundary[quantity]).simplify_full().is_trivial_zero()):
            error_msg = f'The internal flows of {quantity!r} do not cancel: the total changes by {value}.'
            raise ConservationError(error_msg)
    return {quantity: boundary[quantity] for quantity in stock_flow.quantities if quantity in boundary} | {
        quantity: SR(0) for quantity in stock_flow.quantities if quantity not in boundary
    }


def flow_matrix(trajectory: Trajectory, frame: int = -1, quantity: str | None = None) -> FlowMatrix:
    """Return the transaction-flow matrix of one frame, with flows as rows and holders as columns.

    Every row holds the outflow of its flow at the source, negative, and the inflow at the sink, positive, so every
    row sums to zero. Flows of a group sum over its members.

    Args:
        trajectory:
            The trajectory of a system with flows.

        frame:
            The index of the frame.

        quantity:
            The quantity whose flows form the rows, or `None` for every flow.

    Returns:
        matrix:
            The transaction-flow matrix.

    Examples:
        >>> from skcomplex.core.analysis import flow_matrix
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.models import create
        >>> trajectory = Simulation(create('bathtub', water=20.0), dt=1).run(6).trajectory_
        >>> flow_matrix(trajectory).values.sum(axis=1).tolist()
        [0.0, 0.0]
    """
    stock_flow = _stock_flow(trajectory.problem)
    flows = [flow for flow in stock_flow.flows if quantity is None or flow.quantity == quantity]
    holders = list(dict.fromkeys(name for flow in flows for name in (flow.source_label, flow.sink_label)))
    values = np.zeros((len(flows), len(holders)))
    for row, flow in enumerate(flows):
        amount = float(np.sum(trajectory[flow.path][frame]))
        values[row, holders.index(flow.source_label)] -= amount
        values[row, holders.index(flow.sink_label)] += amount
    return FlowMatrix(tuple(flow.path for flow in flows), tuple(holders), values)
