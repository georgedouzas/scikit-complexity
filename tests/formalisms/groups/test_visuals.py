"""Test the Cayley graph visual of lattice groups on group spaces."""

import numpy as np

from skcomplex.core.base import NumericDomain, state
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import Totalistic
from skcomplex.core.space import Memory
from skcomplex.core.system import System
from skcomplex.engine.runtime import Frame, Simulation
from skcomplex.engine.viz import default_blueprint
from skcomplex.formalisms.groups import CayleySpace, CayleyVisual, finitely_presented_group
from skcomplex.formalisms.lattices import LocalRuleLaw


class ParityComponent(ElementComponent):
    """Cell that takes the parity of itself and the cells at its generators."""

    alive: float = state(default=0, domain=NumericDomain.BOOLEAN)

    def laws(self, s):
        """Return the parity rule."""
        return [LocalRuleLaw(s.alive, Memory(('1', 'a', 'b')), Totalistic({1: 1, 3: 1}))]


class Logs:
    """Recording stand-in that keeps the entity paths, archetypes, and static flags it receives."""

    def __init__(self):
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity."""
        self.logs.append((path, archetype, static))


def _simulation():
    """Return a simulation of the parity rule on a ball of radius 3 of the free group of rank 2."""
    space = CayleySpace(finitely_presented_group(['a', 'b']), ball=3).prepare()
    cells = np.zeros(space.size_)
    cells[0] = 1.0
    return Simulation(System(components=[ParityComponent.lattice('cells', space, alive=cells)])).run(until=2)


def test_cayley_visual_logs_nodes_on_rings_and_edges_once():
    """Test that the nodes sit on rings by word length, colored by state, and the edges are logged once as static."""
    simulation = _simulation()
    problem = simulation.problem_
    visual, logs = CayleyVisual().prepare(problem.elements[0], problem), Logs()
    for _ in range(2):
        visual.run(Frame(0, 0.0, simulation.state_, simulation.parameters_, problem), logs)
    nodes = [archetype for path, archetype, static in logs.logs if not static]
    edges = [archetype for path, archetype, static in logs.logs if static]
    assert len(nodes) == 2
    assert len(edges) == 1
    positions = np.asarray(nodes[0].positions.as_arrow_array().to_pylist())
    space = problem.elements[0].element.space
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1) / 40.0, space.lengths_)
    assert len(edges[0].edges.as_arrow_array().to_pylist()) == len(space.edges_)


def test_cayley_visual_views_are_graph_views():
    """Test that a group on a group space defaults to the Cayley visual, whose view is a graph view of its state."""
    simulation = _simulation()
    problem = simulation.problem_
    views = CayleyVisual().prepare(problem.elements[0], problem).views_
    assert [(type(view).__name__, str(view.origin)) for view in views] == [('GraphView', 'cells/alive')]
    assert simulation.system.elements[0].visual.kind == 'cayley'
    assert default_blueprint(simulation.system) is not None
