"""Test the geometry visual of rewriting systems."""

import networkx as nx
import rerun.blueprint as rrb

from skcomplex.core.compile import compile_system
from skcomplex.core.model import GraphProduction, HostGraph, Production, Rewrite, System, Turtle, Word
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import GeometryVisual, RerunObserver


def _views(system):
    """Return the view classes of the geometry visual of the first element of a system."""
    problem = compile_system(system)
    return [type(view) for view in GeometryVisual().views(problem.elements[0], problem)]


def test_views_and_logs_of_graphs_and_turtles(tmp_path):
    """Test the views of a host graph, a 3D turtle, and an unbounded word, and that each is logged."""
    interface = nx.Graph()
    interface.add_nodes_from('ab')
    production = GraphProduction(nx.Graph([('a', 'b')]), interface, nx.Graph([('a', 'c'), ('c', 'b')]))
    graph = System(
        components=[HostGraph('graph', graph=nx.cycle_graph(3))],
        interactions=[Rewrite('rules', target='graph', rules=[production])],
    )
    tree = System(
        components=[Word('tree', axiom='F', turtle=Turtle(angle=30.0, dimension=3))],
        interactions=[Rewrite('rules', target='tree', rules=[Production('F', 'F[&F][/^F]')])],
    )
    plain = System(components=[Word('word', axiom='F', turtle=Turtle())])
    assert _views(graph) == [rrb.GraphView, rrb.TimeSeriesView]
    assert _views(tree) == [rrb.Spatial3DView, rrb.TimeSeriesView]
    assert _views(plain) == [rrb.Spatial2DView, rrb.TimeSeriesView]
    for name, system in [('graph', graph), ('tree', tree), ('plain', plain)]:
        observer = RerunObserver(name, save_path=tmp_path / f'{name}.rrd', blueprint='default')
        Simulation(system, observers=[observer]).run(until=2)
        assert observer.logged_frames_ == 3
