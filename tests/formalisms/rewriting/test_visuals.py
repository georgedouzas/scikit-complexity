"""Test the geometry visual of rewriting systems."""

import networkx as nx
import rerun.blueprint as rrb

from skcomplex.core.system import System
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.formalisms.rewriting import (
    GeometryVisual,
    GraphProductionLaw,
    HostGraphElement,
    ProductionLaw,
    RewriteLaw,
    WordElement,
    WordGeometry,
)


def _views(system):
    """Return the view classes of the geometry visual of the first element of a system."""
    problem = compile_system(system)
    return [type(view) for view in GeometryVisual().prepare(problem.elements[0], problem).views_]


def test_views_and_logs_of_graphs_and_word_geometries(tmp_path):
    """Test the views of a host graph, a 3D word geometry, and an unbounded word, and that each is logged."""
    interface = nx.Graph()
    interface.add_nodes_from('ab')
    production = GraphProductionLaw(nx.Graph([('a', 'b')]), interface, nx.Graph([('a', 'c'), ('c', 'b')]))
    SubdividedElement = HostGraphElement.define(
        'SubdividedElement',
        laws=lambda s: [RewriteLaw([production])],
    )
    TreeElement = WordElement.define(
        'TreeElement',
        laws=lambda s: [RewriteLaw([ProductionLaw('F', 'F[&F][/^F]')])],
    )
    graph = System(components=[SubdividedElement('graph', graph=nx.cycle_graph(3))])
    tree = System(components=[TreeElement('tree', axiom='F', geometry=WordGeometry(angle=30.0, dimension=3))])
    plain = System(components=[WordElement('word', axiom='F', geometry=WordGeometry())])
    assert _views(graph) == [rrb.GraphView, rrb.TimeSeriesView]
    assert _views(tree) == [rrb.Spatial3DView, rrb.TimeSeriesView]
    assert _views(plain) == [rrb.Spatial2DView, rrb.TimeSeriesView]
    for name, system in [('graph', graph), ('tree', tree), ('plain', plain)]:
        observer = RerunObserver(name, save_path=tmp_path / f'{name}.rrd', blueprint='default')
        Simulation(system, observers=[observer]).run(until=2)
        assert observer.logged_frames_ == 3
