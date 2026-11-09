"""Derive the feedback loops of the signed dependency graph of a system.

The module also gives their dominance and its flow matrices.

`SETTINGS` holds the settings every kind reads, besides `kind`.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, dirac_delta

from ..components.laws.utils._delays import delay_parts
from ..engine._utils import symbols_of
from ..engine.problems import ProblemSpec
from ..engine.problems._expression_printer import ExpressionPrinter
from ..engine.problems._loop_term_spec import LoopTermSpec
from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import AnalysisTags
from ..specs._loop_spec import _LoopSpec
from ..types import Array, DataKind
from ._base import _check_choice, _check_settings, _composite, _read_system
from ._stability import StabilityResult
from ._symbolic import _transfer_network
from ._utils import compile_problem
from .types import Source

KINDS = ('loops', 'dominance', 'flow-matrix')
SETTINGS = {
    'loops': {'at'},
    'dominance': set(),
    'flow-matrix': {'frame', 'quantity'},
}
POLARITIES = {1: 'reinforcing', -1: 'balancing', 0: 'neutral'}
TAGS = AnalysisTags(DataKind.SYSTEM)


class _Graph:
    """Edges of the signed dependency graph of a problem.

    They come with the expressions of their partial derivatives. The nodes are the state variables and the flows. A
    state whose rate the flows and rate laws of single elements give is linked through them, and every other state
    through its row of the Jacobian.
    """

    def __init__(self, problem: ProblemSpec) -> None:
        self.problem = problem
        self.readers: dict[Expression, tuple[bool, int]] = {}
        self.edges: dict[tuple[str, str], Expression] = defaultdict(lambda: SR(0))
        terms = problem.transfer_network.terms if problem.transfer_network is not None else ()
        self.described = self._get_described(terms)
        self.states = {entry.name for entry in problem.states.entries}
        for term in terms:
            if self._get_holders(term) & self.described:
                self._add(term)
        hidden = self._get_hidden()
        if any(name not in self.described | hidden for name in self.states):
            self._add_jacobian(hidden)

    def _reads_group(self, term: LoopTermSpec) -> bool:
        """Check whether a term reads or changes a variable of a group.

        Its expression does not resolve such a variable.
        """
        entries = [term.references[item].entry for item in symbols_of([term.expression]) if item in term.references]
        entries += [self.problem.states.get_entry(name) for name in self._get_holders(term)]
        return any(entry.members is not None for entry in entries)

    def _get_described(self, terms: tuple[LoopTermSpec, ...]) -> set[str]:
        """Return the states whose rate the terms give.

        The terms of a state of higher order give its highest derivative rather than its rate, and those that read or
        change a group give the rates of all its members at once, so the Jacobian describes these states instead.
        """
        changed = {name for term in terms for name in self._get_holders(term)}
        unresolved = {name for term in terms if self._reads_group(term) for name in self._get_holders(term)}
        higher = {owner for kind, owner, _ in self.problem.companions.values() if kind == 'derivative'}
        return changed - unresolved - higher

    def _get_hidden(self) -> set[str]:
        """Return the states the compiler adds for boundary flows, delays, and smoothings."""
        transfer_network = self.problem.transfer_network
        if transfer_network is None:
            return set()
        outputs = symbols_of([delay.output for delay in transfer_network.delays])
        delays = {name.partition('[')[0] for name, item in self.problem.equation_symbols.items() if item in outputs}
        return set(transfer_network.boundary.values()) | delays

    def _get_node_of(self, item: Expression, term: LoopTermSpec) -> str | None:
        """Return the state a symbol stands for, or `None` for parameters and time."""
        reference = term.references.get(item)
        if reference is None:
            return None
        self.readers[item] = (reference.is_state, reference.entry.start + reference.column)
        return reference.entry.name if reference.is_state else None

    def _add(self, term: LoopTermSpec) -> None:
        """Add the edges of one term, from every node its expression reads."""
        expression = self._copy_without_delays(term.expression)
        for item in symbols_of([expression]) - {self.problem.time}:
            node = self._get_node_of(item, term)
            if node is not None:
                self.edges[(node, term.path)] += self._get_derivative(expression, item)
        if term.kind == 'flow':
            if term.sink in self.described:
                self.edges[(term.path, term.sink)] += 1
            if term.source in self.described:
                self.edges[(term.path, term.source)] -= 1

    def _add_jacobian(self, hidden: set[str]) -> None:
        """Add the edges into every state the terms do not describe.

        The edges come from its row of the Jacobian. The row of a map is that of the change of every state in one step,
        its next value minus itself, so that its edges and those of flows per step have the same meaning.
        """
        dynamics = StabilityResult._get_dynamics_of(self.problem)
        nodes = {}
        for name, item in zip(dynamics.names, dynamics.symbols, strict=True):
            entry_name = name.partition('[')[0]
            if entry_name not in hidden:
                nodes[item] = entry_name if entry_name in self.described else name
                self.readers[item] = (True, self.problem.states.locate(name)[1].start)
        self.states |= set(nodes.values())
        for name, item, rate in zip(dynamics.names, dynamics.symbols, dynamics.rates, strict=True):
            if name.partition('[')[0] in self.described | hidden:
                continue
            change = SR(rate - item if dynamics.discrete else rate).subs(dynamics.parameters)
            for other, node in nodes.items():
                derivative = self._get_derivative(change, other)
                if not derivative.is_trivial_zero():
                    self.edges[(node, nodes[item])] += derivative

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
            error_msg = f'The loop gain {expression} reads {unknown[0]}, which is no state or parameter of the system.'
            raise AnalysisError(error_msg)
        function = ExpressionPrinter(time=self.problem.time, compiled=True).prepare(arguments).run([expression]).code_
        if not callable(function):
            error_msg = f'The printer returned no function of the loop gain {expression}.'
            raise AnalysisError(error_msg)
        values = [
            (states if self.readers[argument][0] else parameters)[:, self.readers[argument][1]]
            for argument in arguments
        ]
        gains = np.asarray(function(times, *values), dtype=np.float64)[0]
        return np.broadcast_to(gains, np.shape(times)).copy()

    @staticmethod
    def _copy_without_delays(expression: Expression) -> Expression:
        """Return an expression with every delay and smoothing replaced by its input.

        A delay or a smoothing follows its input with gain one.
        """
        calls = {
            item: item.operands()[0]
            for item in _Graph._get_operations(expression)
            if delay_parts(getattr(item.operator(), 'name', lambda: '')()) is not None
        }
        return _Graph._copy_without_delays(SR(expression).subs(calls)) if calls else SR(expression)

    @staticmethod
    def _get_derivative(expression: Expression, item: Expression) -> Expression:
        """Return the partial derivative of an expression without the deltas of steps.

        The deltas are Dirac deltas.

        A step's derivative is zero off its jump.
        """
        return SR(expression).diff(item).subs(dirac_delta(SR.wild(0)) == 0)

    @staticmethod
    def _get_holders(term: LoopTermSpec) -> set[str]:
        """Return the states whose rate a term changes."""
        return {term.path} if term.kind == 'rate' else {name for name in (term.source, term.sink) if name is not None}

    @staticmethod
    def _get_operations(expression: Expression) -> list[Expression]:
        """Return every subexpression of an expression that applies an operator."""
        found, stack = [], [SR(expression)]
        while stack:
            item = stack.pop()
            if item.operator() is not None:
                found.append(item)
                stack.extend(item.operands())
        return found


@dataclass(frozen=True)
class LoopsResult:
    """Feedback loops of the signed dependency graph of a system.

    They come with their dominance in time, or its flow matrix.

    Args:
        problem:
            The compiled problem.

        trajectory:
            The trajectory of `'dominance'` and `'flow-matrix'`.

        graph:
            The signed dependency graph from each state or flow to every node whose amount or rate depends on it, with
            the edge attributes `sign` and `gain`, for `'loops'`. The members of a group and the components of a vector
            that the Jacobian describes are nodes of their own, named like `species.x[0]`.

        loops:
            The nodes of every loop, in the order of their influence, starting at its first state by name, or at its
            first node by name when it holds no state.

        polarities:
            The polarity of every loop, at the first frame for `'dominance'`: `'reinforcing'` for a positive product
            of edge signs, `'balancing'` for a negative one, or `'neutral'` when an edge has no influence.

        gains:
            The gain of every loop, with one row per frame for `'dominance'`.

        times:
            The time of every frame of `'dominance'`.

        dominant:
            The index of the loop with the largest absolute gain at every frame of `'dominance'`.

        flows:
            The paths of the flows of the flow matrix.

        holders:
            The states and environments the flows connect.

        matrix:
            The outflow, negative, and the inflow, positive, of every flow for every holder.
    """

    problem: ProblemSpec
    trajectory: Trajectory | None = None
    graph: nx.DiGraph | None = None
    loops: list[tuple[str, ...]] | None = None
    polarities: list[str] | None = None
    gains: Array | None = None
    times: Array | None = None
    dominant: npt.NDArray[np.intp] | None = None
    flows: tuple[str, ...] | None = None
    holders: tuple[str, ...] | None = None
    matrix: Array | None = None


def loops(
    source: Source,
    *,
    kind: str = 'loops',
    at: npt.ArrayLike | None = None,
    frame: int = -1,
    quantity: str | None = None,
) -> LoopsResult:
    """Return the feedback loops of the signed dependency graph of a system.

    They come with their dominance, or its flow matrix. The signed dependency graph of a system has the state variables
    as nodes and an edge from `x_j` to `x_i` when the rate of `x_i` depends on `x_j`, with the sign of the partial
    derivative `df_i/dx_j`, the entry of the Jacobian. The rate of a map is the change of its state in one step, its
    next value minus itself. When a system declares flows, the flows are nodes too: an edge goes from each state to
    every flow whose amount depends on it, with the sign of the partial derivative, and from each flow to its sink with
    sign `+` and to its source with sign `-`, so the graph shows the path of every influence through the flows. Rate
    laws of single elements give the edges into the states they change, and the Jacobian, as `stability` derives it with
    the parameters at their values, gives the edges into every other state, such as the members of a group or the
    velocity of a state of second order. Delays and smoothings pass their input with gain one.

    The kinds read a system or a trajectory:

    - `'loops'`: the signed dependency graph and its feedback loops, the elementary cycles of the graph found with
      Johnson's algorithm. The gain of a loop is the product of the partial derivatives along it, and its polarity
      the sign of the gain.
    - `'dominance'`: the gains of the loops along a trajectory, at every frame, and the dominant loop, with the largest
      absolute gain, at every frame.
    - `'flow-matrix'`: the transaction-flow matrix of one frame of a trajectory, with flows as rows and holders as
      columns. Every row holds the outflow of its flow at the source, negative, and the inflow at the sink, positive,
      so every row sums to zero. Flows of a group sum over its members. The matrix books the amounts the flows move
      between holders, which the Jacobian does not hold, so it is defined only for systems that declare flows.

    Args:
        source:
            The system, its compiled problem, or a trajectory of it, and a trajectory for `'dominance'` and
            `'flow-matrix'`.

        kind:
            What to derive of the loops, one of the kinds above.

        at:
            The state vector the signs of `'loops'` are evaluated at, or `None` for the initial state.

        frame:
            The index of the frame of the flow matrix.

        quantity:
            The quantity whose flows form the rows of the flow matrix, or `None` for every flow.

    Returns:
        result:
            The loops with their graph, their gains along the trajectory, or the flow matrix.

    Raises:
        AnalysisError:
            The kind is unknown, a setting the kind does not read is changed, the kind reads a trajectory and gets
            another source, the system has no loop to dominate, or it declares no flows for a flow matrix.

    Examples:
        >>> from skcomplex.core.analysis import loops
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.economics.innovation import BassDiffusionSystem
        >>> loops(BassDiffusionSystem().prepare().system_).polarities
        ['reinforcing', 'balancing']
        >>> trajectory = simulate(BassDiffusionSystem().prepare().system_, until=2, dt=1)
        >>> loops(trajectory, kind='dominance').gains.shape
        (3, 2)
    """
    source = _composite(source)
    _check_choice('loops', 'kind', kind, KINDS)
    _check_settings(loops, 'kind', SETTINGS[kind], kind=kind, at=at, frame=frame, quantity=quantity)
    if kind != 'loops' and not isinstance(source, Trajectory):
        error_msg = f'loops of kind {kind!r} reads a trajectory, not {type(source).__name__}.'
        raise AnalysisError(error_msg)
    trajectory = source if isinstance(source, Trajectory) else None
    problem = compile_problem(_read_system(source), 'The feedback loop analysis')
    if kind == 'flow-matrix' and not _transfer_network(problem).flows:
        error_msg = (
            'The system declares no flows, so it has no transaction-flow matrix. Read the signed dependency of its '
            "states with the kind 'loops', or its Jacobian with `stability`."
        )
        raise AnalysisError(error_msg)
    if kind == 'loops':
        graph = _Graph(problem)
        state = np.asarray(problem.initial_state if at is None else at, dtype=float)[np.newaxis]
        parameters, times = problem.initial_parameters[np.newaxis], np.zeros(1)
        gains = {
            edge: float(graph.evaluate(SR(value), state, parameters, times)[0]) for edge, value in graph.edges.items()
        }
        directed, found = _signed_loops(graph, gains)
        return LoopsResult(
            problem,
            trajectory=trajectory,
            graph=directed,
            loops=[loop.nodes for loop in found],
            polarities=[loop.polarity for loop in found],
            gains=np.asarray([loop.gain for loop in found], dtype=float),
        )
    if trajectory is None:
        error_msg = f'loops of kind {kind!r} reads a trajectory. Give one.'
        raise AnalysisError(error_msg)
    if kind == 'dominance':
        return _dominance(problem, trajectory)
    return _flow_matrix(problem, trajectory, frame, quantity)


def _dominance(problem: ProblemSpec, trajectory: Trajectory) -> LoopsResult:
    """Return the gains of the loops along a trajectory.

    The dominant loop at every frame is returned too.
    """
    graph = _Graph(problem)
    states, parameters = np.atleast_2d(trajectory.states), np.atleast_2d(trajectory.parameters)
    times = np.asarray(trajectory.times, dtype=float)
    values = {edge: graph.evaluate(SR(value), states, parameters, times) for edge, value in graph.edges.items()}
    _, found = _signed_loops(graph, {edge: float(value[0]) for edge, value in values.items()})
    if not found:
        error_msg = 'The system has no feedback loop, so no loop dominates. Analyse a system with a loop.'
        raise AnalysisError(error_msg)
    gains = np.column_stack(
        [
            np.prod(
                [values[pair] for pair in zip(loop.nodes, [*loop.nodes[1:], loop.nodes[0]], strict=True)],
                axis=0,
            )
            for loop in found
        ],
    )
    return LoopsResult(
        problem,
        trajectory=trajectory,
        loops=[loop.nodes for loop in found],
        polarities=[loop.polarity for loop in found],
        gains=gains,
        times=times,
        dominant=np.argmax(np.abs(gains), axis=1),
    )


def _flow_matrix(problem: ProblemSpec, trajectory: Trajectory, frame: int, quantity: str | None) -> LoopsResult:
    """Return the transaction-flow matrix of one frame."""
    transfer_network = _transfer_network(problem)
    flows = [flow for flow in transfer_network.flows if quantity is None or flow.quantity == quantity]
    holders = list(dict.fromkeys(name for flow in flows for name in (flow.source_label, flow.sink_label)))
    values = np.zeros((len(flows), len(holders)))
    for row, flow in enumerate(flows):
        amount = float(np.sum(trajectory.get_values(flow.path)[frame]))
        values[row, holders.index(flow.source_label)] -= amount
        values[row, holders.index(flow.sink_label)] += amount
    return LoopsResult(
        problem,
        trajectory=trajectory,
        flows=tuple(flow.path for flow in flows),
        holders=tuple(holders),
        matrix=values,
    )


def _signed_loops(graph: _Graph, gains: Mapping[tuple[str, str], float]) -> tuple[nx.DiGraph, list[_LoopSpec]]:
    """Return the signed graph with edge gains and its loops with polarity and gain."""
    directed = nx.DiGraph()
    for (start, end), derivative in graph.edges.items():
        if SR(derivative).is_trivial_zero():
            continue
        gain = gains[(start, end)]
        directed.add_edge(start, end, sign=int(np.sign(gain)), gain=gain)
    loops = []
    for found in nx.simple_cycles(directed):
        anchor = min((node for node in found if node in graph.states), default=min(found))
        start = found.index(anchor)
        cycle = [*found[start:], *found[:start]]
        pairs = list(zip(cycle, [*cycle[1:], cycle[0]], strict=True))
        gain = float(np.prod([directed.edges[pair]['gain'] for pair in pairs]))
        loops.append(_LoopSpec(tuple(cycle), POLARITIES[int(np.sign(gain))], gain))
    return directed, sorted(loops, key=lambda loop: (len(loop.nodes), loop.nodes))
