"""Draw the variables of a linear program as bars, its objective values as series, and its structure as a graph."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import networkx as nx
import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb

from ...components.specs import CompiledAgentsSpec, CompiledElementSpec
from ...errors import ValidationError
from ..problems import BaseProblemSpec, FrameSpec, LinearProgramKernel, ProblemSpec
from ._base import BaseVisual

GRAPH_LIMIT = 100
LAYOUT_SEED = 0
LAYOUT_SCALE = 10.0
MAXIMUM_WIDTH = 0.2
MINIMUM_WIDTH = 0.005


class LinearProgramVisual(BaseVisual):
    """Visual that draws the linear program of an element.

    It logs a bar chart of the variables under `<path>/variables`, the objective values as scalars under
    `<path>/objective`, and with a graph one node per row of the matrix and per variable under `<path>/graph`, laid out
    once with a seeded layout. It also logs the edges under `<path>/flows` as lines whose widths are proportional to the
    absolute values of their variables.

    Args:
        variables:
            The dotted names of the variables to draw, or `None` for every variable in declared order.

        graph:
            Whether to log the graph, or `None` for a graph up to 100 variables.

        seed:
            The seed of the layout of the graph.

        path:
            The entity path of the element to draw, which also names it in the problem.

        recording:
            The Rerun recording to log to, or `None` for the active recording of the thread.

    Attributes:
        kernel_:
            The kernel of the linear program of the element.

        names_:
            The dotted names of the variables the visual draws.

        positions_:
            The position of every node of the graph, by its name, or nothing without a graph.

    Examples:
        >>> from skcomplex.core.engine.viz import LinearProgramVisual
        >>> LinearProgramVisual(graph=False)
        LinearProgramVisual(graph=False)
    """

    def __init__(
        self,
        variables: Sequence[str] | None = None,
        *,
        graph: bool | None = None,
        seed: int = LAYOUT_SEED,
        path: str | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.variables = variables
        self.graph = graph
        self.seed = seed
        self.path = path
        self.recording = recording

    def _kernel_of(self, problem: BaseProblemSpec | None) -> LinearProgramKernel:
        """Return the kernel of the linear program of the element at the path of the visual."""
        kernels = problem.programs if isinstance(problem, ProblemSpec) else ()
        prefix = f'{self.path}.'
        for kernel in kernels:
            if any(name.startswith(prefix) for name in kernel.program.names):
                return kernel
        error_msg = f'{type(self).__name__} draws a linear program, but no program sets a state of {self.path}.'
        raise ValidationError(error_msg)

    def _prepare(self, context: BaseProblemSpec | None) -> None:
        """Keep the element and its program, lay out the graph, and set the views.

        Args:
            context:
                The compiled problem.

        Raises:
            ValidationError:
                No linear program sets a state of the element.
        """
        self.kernel_ = self._kernel_of(context)
        program = self.kernel_.program
        self.names_ = tuple(program.names if self.variables is None else self.variables)
        self.positions_ = self._layout_graph() if self._draws_graph() else {}
        self._logged_graph = False
        super()._prepare(context)

    def _draws_graph(self) -> bool:
        """Check whether the visual logs a graph."""
        return len(self.kernel_.program.names) <= GRAPH_LIMIT if self.graph is None else bool(self.graph)

    def _edges(self) -> list[tuple[str, str, int]]:
        """Return an edge per nonzero coefficient: its row, its variable, and the column of the variable."""
        program = self.kernel_.program
        matrix = program.matrix.tocoo()
        return [
            (self._row_name(int(row)), program.names[int(column)], int(column))
            for row, column in zip(matrix.row, matrix.col, strict=True)
        ]

    @staticmethod
    def _row_name(row: int) -> str:
        """Return the name of the node of a row of the matrix."""
        return f'row_{row}'

    def _layout_graph(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return the position of every node from a seeded spring layout."""
        program = self.kernel_.program
        graph = nx.Graph()
        graph.add_nodes_from(self._row_name(row) for row in range(len(program.constraints)))
        graph.add_nodes_from(program.names)
        graph.add_edges_from((row, name) for row, name, _ in self._edges())
        layout = nx.spring_layout(graph, seed=self.seed, scale=LAYOUT_SCALE)
        return {name: np.asarray(position, dtype=float) for name, position in layout.items()}

    def _log_graph(self, recording: rr.RecordingStream, path: str) -> None:
        """Log the nodes at their positions and the edges of the graph once."""
        names = list(self.positions_)
        program = self.kernel_.program
        labels = [*program.constraints, *(item.rpartition('.')[2] for item in program.names)]
        positions = np.stack([self.positions_[name] for name in names])
        nodes = rr.GraphNodes(names, positions=positions, labels=labels, show_labels=False)
        recording.log(f'{path}/graph', nodes, static=True)
        edges = rr.GraphEdges([(row, name) for row, name, _ in self._edges()], graph_type='undirected')
        recording.log(f'{path}/graph', edges, static=True)

    def _log_widths(self, recording: rr.RecordingStream, path: str, values: npt.NDArray[np.float64]) -> None:
        """Log the edges as lines whose widths are proportional to the absolute values of their variables."""
        edges = self._edges()
        if not edges:
            return
        amounts = np.asarray([abs(values[column]) for _, _, column in edges], dtype=float)
        largest = float(np.max(amounts))
        widths = amounts / largest * MAXIMUM_WIDTH if largest > 0 else np.zeros_like(amounts)
        strips = [np.stack([self.positions_[row], self.positions_[name]]) for row, name, _ in edges]
        recording.log(f'{path}/flows', rr.LineStrips2D(strips, radii=np.maximum(widths, MINIMUM_WIDTH)))

    def _log(self, recording: rr.RecordingStream, element: CompiledElementSpec, frame: FrameSpec) -> None:
        """Log the bars, the objective values, and the graph of the program.

        Args:
            recording:
                The Rerun recording.

            element:
                The compiled element, with its path.

            frame:
                The frame.
        """
        path = element.path
        program = self.kernel_.program
        chart = np.asarray([float(np.ravel(frame.value(name))[0]) for name in self.names_], dtype=float)
        recording.log(f'{path}/variables', rr.BarChart(chart))
        for position in range(len(program.objectives)):
            value = frame.value(f'{program.path}.objective_{position}')
            name = 'objective' if position == 0 else f'objective_{position}'
            BaseVisual._log_scalars(recording, f'{path}/{name}', value)
        if not self.positions_:
            return
        if not self._logged_graph:
            self._log_graph(recording, path)
            self._logged_graph = True
        values = np.asarray([float(np.ravel(frame.value(name))[0]) for name in program.names], dtype=float)
        self._log_widths(recording, path, values)

    def _views(self, element: CompiledElementSpec | CompiledAgentsSpec, problem: BaseProblemSpec) -> list[rrb.View]:
        """Return the bar chart, the series of the objective values, and with a graph its two views.

        Args:
            element:
                The compiled element, with its path.

            problem:
                The compiled problem the element belongs to.

        Returns:
            views:
                The views, the bar chart first.
        """
        path = element.path
        views: list[rrb.View] = [
            rrb.BarChartView(origin=f'{path}/variables', name=f'{path} variables'),
            rrb.TimeSeriesView(origin=f'{path}/objective', name=f'{path} objective'),
        ]
        if self._draws_graph():
            views += [
                rrb.GraphView(origin=f'{path}/graph', name=f'{path} graph'),
                rrb.Spatial2DView(origin=f'{path}/flows', name=f'{path} flows'),
            ]
        return views
