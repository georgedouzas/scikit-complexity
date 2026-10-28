"""Derive the feedback loops of the signed dependency graph of a system, their dominance, and its flow matrices."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import networkx as nx
import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, dirac_delta

from ..core.errors import AnalysisError
from ..core.laws.utils import delay_parts
from ..engine.compile import compiled_problem
from ..engine.problems import LoopTerm, Problem, numpy_function, symbols_of
from ..engine.runtime import Trajectory
from ._base import BaseAnalysis, Source
from ._kinds import DataKind
from ._stability import StabilityAnalysis
from ._symbolic import SymbolicAnalysis

KINDS = ('loops', 'dominance', 'flow-matrix')
SETTINGS = {
    'loops': {'at'},
    'dominance': set(),
    'flow-matrix': {'frame', 'quantity'},
}
"""Settings every kind reads, besides `kind`."""
POLARITIES = {1: 'reinforcing', -1: 'balancing', 0: 'neutral'}


@dataclass(frozen=True)
class _Loop:
    """Feedback loop: its nodes from its first state, its polarity, and its gain at the point of evaluation."""

    nodes: tuple[str, ...]
    polarity: str
    gain: float


class _Graph:
    """Edges of the signed dependency graph of a problem with the expressions of their partial derivatives.

    The nodes are the state variables and the flows. A state whose rate the flows and rate laws of single elements
    give is linked through them, and every other state through its row of the Jacobian.
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.readers: dict[Expression, tuple[bool, int]] = {}
        self.edges: dict[tuple[str, str], Expression] = defaultdict(lambda: SR(0))
        terms = problem.stock_flow.terms if problem.stock_flow is not None else ()
        self.described = self._described(terms)
        self.states = {entry.name for entry in problem.states.entries}
        for term in terms:
            if self._holders(term) & self.described:
                self._add(term)
        hidden = self._hidden()
        if any(name not in self.described | hidden for name in self.states):
            self._add_jacobian(hidden)

    def _reads_group(self, term: LoopTerm) -> bool:
        """Check whether a term reads or changes a variable of a group, which its expression does not resolve."""
        entries = [term.references[item].entry for item in symbols_of([term.expression]) if item in term.references]
        entries += [self.problem.states.entry(name) for name in self._holders(term)]
        return any(entry.members is not None for entry in entries)

    def _described(self, terms: tuple[LoopTerm, ...]) -> set[str]:
        """Return the states whose rate the terms give.

        The terms of a state of higher order give its highest derivative rather than its rate, and those that read or
        change a group give the rates of all its members at once, so the Jacobian describes these states instead.
        """
        changed = {name for term in terms for name in self._holders(term)}
        unresolved = {name for term in terms if self._reads_group(term) for name in self._holders(term)}
        higher = {owner for kind, owner, _ in self.problem.companions.values() if kind == 'derivative'}
        return changed - unresolved - higher

    def _hidden(self) -> set[str]:
        """Return the states the compiler adds for the boundary flows and for delays and smoothings."""
        stock_flow = self.problem.stock_flow
        if stock_flow is None:
            return set()
        outputs = symbols_of([delay.output for delay in stock_flow.delays])
        delays = {
            name.partition('[')[0] for name, item in self.problem.equation_symbols.items() if item in outputs
        }
        return set(stock_flow.boundary.values()) | delays

    def _node_of(self, item: Expression, term: LoopTerm) -> str | None:
        """Return the state a symbol stands for, or `None` for parameters and time."""
        reference = term.references.get(item)
        if reference is None:
            return None
        self.readers[item] = (reference.is_state, reference.entry.start + reference.column)
        return reference.entry.name if reference.is_state else None

    def _add(self, term: LoopTerm) -> None:
        """Add the edges of one term, from every node its expression reads."""
        expression = self._without_delays(term.expression)
        for item in symbols_of([expression]) - {self.problem.time}:
            node = self._node_of(item, term)
            if node is not None:
                self.edges[(node, term.path)] += self._derivative(expression, item)
        if term.kind == 'flow':
            if term.sink in self.described:
                self.edges[(term.path, term.sink)] += 1
            if term.source in self.described:
                self.edges[(term.path, term.source)] -= 1

    def _add_jacobian(self, hidden: set[str]) -> None:
        """Add the edges into every state the terms do not describe, from its row of the Jacobian.

        The row of a map is that of the change of every state in one step, its next value minus itself, so that its
        edges and those of flows per step have the same meaning.
        """
        dynamics = StabilityAnalysis._dynamics_of(self.problem)
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
                derivative = self._derivative(change, other)
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
        function = numpy_function(self.problem.time, arguments, [expression])
        values = [
            (states if is_state else parameters)[:, index]
            for is_state, index in map(self.readers.__getitem__, arguments)
        ]
        return np.broadcast_to(np.asarray(function(times, *values)[0], dtype=float), np.shape(times)).copy()

    @staticmethod
    def _without_delays(expression: Expression) -> Expression:
        """Return an expression with every delay and smoothing replaced by its input, which it follows with gain one."""
        calls = {
            item: item.operands()[0]
            for item in _Graph._operations(expression)
            if delay_parts(getattr(item.operator(), 'name', lambda: '')()) is not None
        }
        return _Graph._without_delays(SR(expression).subs(calls)) if calls else SR(expression)

    @staticmethod
    def _derivative(expression: Expression, item: Expression) -> Expression:
        """Return the partial derivative of an expression, without the Dirac deltas of steps, zero off their jump."""
        return SR(expression).diff(item).subs(dirac_delta(SR.wild(0)) == 0)

    @staticmethod
    def _holders(term: LoopTerm) -> set[str]:
        """Return the states whose rate a term changes."""
        return {term.path} if term.kind == 'rate' else {name for name in (term.source, term.sink) if name is not None}

    @staticmethod
    def _operations(expression: Expression) -> list[Expression]:
        """Return every subexpression of an expression that applies an operator."""
        found, stack = [], [SR(expression)]
        while stack:
            item = stack.pop()
            if item.operator() is not None:
                found.append(item)
                stack.extend(item.operands())
        return found


class LoopAnalysis(BaseAnalysis):
    """Feedback loops of the signed dependency graph of a system, their dominance in time, or its flow matrix.

    The signed dependency graph of a system has the state variables as nodes and an edge from `x_j` to `x_i` when the
    rate of `x_i` depends on `x_j`, with the sign of the partial derivative `df_i/dx_j`, the entry of the Jacobian. The
    rate of a map is the change of its state in one step, its next value minus itself. When a system declares flows,
    the flows are nodes too: an edge goes from each state to every flow whose amount depends on it, with the sign of
    the partial derivative, and from each flow to its sink with sign `+` and to its source with sign `-`, so the
    graph shows the path of every influence through the flows. Rate laws of single elements give the edges into the
    states they change, and the Jacobian, as `StabilityAnalysis` derives it with the parameters at their values,
    gives the edges into every other state, such as the members of a group or the velocity of a state of second
    order. Delays and smoothings pass their input with gain one.

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
        kind:
            The analysis of the loops, one of the kinds above.

        at:
            The state vector the signs of `'loops'` are evaluated at, or `None` for the initial state.

        frame:
            The index of the frame of the flow matrix.

        quantity:
            The quantity whose flows form the rows of the flow matrix, or `None` for every flow.

    Attributes:
        problem_:
            The compiled problem.

        trajectory_:
            The trajectory of `'dominance'` and `'flow-matrix'`.

        graph_:
            The signed dependency graph from each state or flow to every node whose amount or rate depends on it, with
            the edge attributes `sign` and `gain`, for `'loops'`. The members of a group and the components of a vector
            that the Jacobian describes are nodes of their own, named like `species.x[0]`.

        loops_:
            The nodes of every loop, in the order of their influence, starting at its first state by name, or at its
            first node by name when it holds no state.

        polarities_:
            The polarity of every loop, at the first frame for `'dominance'`: `'reinforcing'` for a positive product
            of edge signs, `'balancing'` for a negative one, or `'neutral'` when an edge has no influence.

        gains_:
            The gain of every loop, with one row per frame for `'dominance'`.

        times_:
            The time of every frame of `'dominance'`.

        dominant_:
            The index of the loop with the largest absolute gain at every frame of `'dominance'`.

        flows_:
            The paths of the flows of the flow matrix.

        holders_:
            The states and environments the flows connect.

        matrix_:
            The outflow, negative, and the inflow, positive, of every flow for every holder.

    Examples:
        >>> from skcomplex.analysis import LoopAnalysis
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models import create
        >>> LoopAnalysis().check().prepare(create('bass_diffusion')).run().polarities_
        ['reinforcing', 'balancing']
        >>> trajectory = Simulation(create('bass_diffusion'), dt=1).run(2).trajectory_
        >>> LoopAnalysis('dominance').check().prepare(trajectory).run().gains_.shape
        (3, 2)
    """

    def __init__(
        self,
        kind: str = 'loops',
        *,
        at: npt.ArrayLike | None = None,
        frame: int = -1,
        quantity: str | None = None,
    ) -> None:
        self.kind = kind
        self.at = at
        self.frame = frame
        self.quantity = quantity

    def _check(self) -> Self:
        """Validate the kind and that the kind reads every changed setting."""
        self._check_choice('kind', self.kind, KINDS)
        self._check_settings('kind', SETTINGS[self.kind])
        return self

    def _get_data_kind(self) -> DataKind:
        """Return the system kind."""
        return DataKind.SYSTEM

    def _prepare(self, source: Source) -> Self:
        """Compile the system of the source, and keep the trajectory that dominance and the flow matrix read."""
        if self.kind != 'loops' and not isinstance(source, Trajectory):
            error_msg = f'{type(self).__name__} of kind {self.kind!r} reads a trajectory, not {type(source).__name__}.'
            raise AnalysisError(error_msg)
        self.trajectory_ = source if isinstance(source, Trajectory) else None
        self.problem_ = compiled_problem(self._read_system(source), 'The feedback loop analysis')
        if self.kind == 'flow-matrix' and not SymbolicAnalysis._stock_flow(self.problem_).flows:
            error_msg = (
                'The system declares no flows, so it has no transaction-flow matrix. Read the signed dependency of its '
                "states with the kind 'loops', or its Jacobian with StabilityAnalysis."
            )
            raise AnalysisError(error_msg)
        return self

    def _set_loops(self, found: list[_Loop]) -> None:
        """Set the nodes and the polarities of loops."""
        self.loops_ = [loop.nodes for loop in found]
        self.polarities_ = [loop.polarity for loop in found]

    def _run_loops(self) -> None:
        """Set the signed graph and its loops at a state."""
        problem = self.problem_
        graph = _Graph(problem)
        state = np.asarray(problem.initial_state if self.at is None else self.at, dtype=float)[np.newaxis]
        parameters, times = problem.initial_parameters[np.newaxis], np.zeros(1)
        gains = {
            edge: float(graph.evaluate(SR(value), state, parameters, times)[0]) for edge, value in graph.edges.items()
        }
        self.graph_, found = self._signed_loops(graph, gains)
        self._set_loops(found)
        self.gains_ = np.asarray([loop.gain for loop in found], dtype=float)

    def _run_dominance(self, trajectory: Trajectory) -> None:
        """Set the gains of the loops along a trajectory and the dominant loop at every frame."""
        graph = _Graph(self.problem_)
        states, parameters = np.atleast_2d(trajectory.states), np.atleast_2d(trajectory.parameters)
        times = np.asarray(trajectory.times, dtype=float)
        values = {edge: graph.evaluate(SR(value), states, parameters, times) for edge, value in graph.edges.items()}
        _, found = self._signed_loops(graph, {edge: float(value[0]) for edge, value in values.items()})
        if not found:
            error_msg = 'The system has no feedback loop, so no loop dominates. Analyse a system with a loop.'
            raise AnalysisError(error_msg)
        self._set_loops(found)
        self.times_ = times
        self.gains_ = np.column_stack(
            [
                np.prod(
                    [values[pair] for pair in zip(loop.nodes, [*loop.nodes[1:], loop.nodes[0]], strict=True)],
                    axis=0,
                )
                for loop in found
            ],
        )
        self.dominant_ = np.argmax(np.abs(self.gains_), axis=1)

    def _run_flow_matrix(self, trajectory: Trajectory) -> None:
        """Set the transaction-flow matrix of one frame."""
        stock_flow = SymbolicAnalysis._stock_flow(self.problem_)
        flows = [flow for flow in stock_flow.flows if self.quantity is None or flow.quantity == self.quantity]
        holders = list(dict.fromkeys(name for flow in flows for name in (flow.source_label, flow.sink_label)))
        values = np.zeros((len(flows), len(holders)))
        for row, flow in enumerate(flows):
            amount = float(np.sum(trajectory.values(flow.path)[self.frame]))
            values[row, holders.index(flow.source_label)] -= amount
            values[row, holders.index(flow.sink_label)] += amount
        self.flows_ = tuple(flow.path for flow in flows)
        self.holders_ = tuple(holders)
        self.matrix_ = values

    def _run(self) -> Self:
        """Derive the loops, their dominance, or the flow matrix."""
        if self.kind == 'loops':
            self._run_loops()
        elif self.kind == 'dominance':
            self._run_dominance(self.trajectory_)  # type: ignore[arg-type]
        else:
            self._run_flow_matrix(self.trajectory_)  # type: ignore[arg-type]
        return self

    @staticmethod
    def _signed_loops(graph: _Graph, gains: Mapping[tuple[str, str], float]) -> tuple[nx.DiGraph, list[_Loop]]:
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
            loops.append(_Loop(tuple(cycle), POLARITIES[int(np.sign(gain))], gain))
        return directed, sorted(loops, key=lambda loop: (len(loop.nodes), loop.nodes))
