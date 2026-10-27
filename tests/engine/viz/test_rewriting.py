"""Test the geometry visual of rewriting systems."""

import networkx as nx
import rerun.blueprint as rrb

from skcomplex.core.elements import HostGraphElement, WordElement
from skcomplex.core.laws import GraphProductionLaw, ProductionLaw
from skcomplex.core.specs import GeometrySpec
from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import GeometryVisual, RerunObserver

INTERFACE = nx.Graph()
INTERFACE.add_nodes_from('ab')
SUBDIVISION = GraphProductionLaw(nx.Graph([('a', 'b')]), INTERFACE, nx.Graph([('a', 'c'), ('c', 'b')]))


class SubdividedElement(HostGraphElement):
    """Graph whose edges are subdivided at every generation."""

    def _get_laws(self, s):
        """Return the subdivision of an edge."""
        return [SUBDIVISION]


class TreeElement(WordElement):
    """Word of a branching tree."""

    def _get_laws(self, s):
        """Return the branching of a segment."""
        return [ProductionLaw('F', 'F[&F][/^F]')]


def _views(system):
    """Return the view classes of the geometry visual of the first element of a system."""
    problem = compile_system(system)
    return [type(view) for view in GeometryVisual().prepare(problem.elements[0], problem).views_]


def test_views_and_logs_of_graphs_and_word_geometries(tmp_path):
    """Test the views of a host graph, a 3D word geometry, and an unbounded word, and that each is logged."""
    graph = System(elements=[SubdividedElement('graph', graph=nx.cycle_graph(3))])
    tree = System(elements=[TreeElement('tree', axiom='F', geometry=GeometrySpec(angle=30.0, dimension=3))])
    plain = System(elements=[WordElement('word', axiom='F', geometry=GeometrySpec())])
    assert _views(graph) == [rrb.GraphView, rrb.TimeSeriesView]
    assert _views(tree) == [rrb.Spatial3DView, rrb.TimeSeriesView]
    assert _views(plain) == [rrb.Spatial2DView, rrb.TimeSeriesView]
    for name, system in [('graph', graph), ('tree', tree), ('plain', plain)]:
        observer = RerunObserver(name, save_path=tmp_path / f'{name}.rrd', blueprint='default')
        Simulation(system, observers=[observer]).run(until=2)
        assert observer.logged_frames_ == 3
