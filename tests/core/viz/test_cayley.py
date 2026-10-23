"""Test the Cayley graph visual of lattice groups on group spaces."""

import numpy as np

from skcomplex.core.base import Domain, state
from skcomplex.core.components import Element
from skcomplex.core.laws import LocalRule, Totalistic
from skcomplex.core.runtime import Frame, Simulation
from skcomplex.core.space import GroupSpace, Memory, finitely_presented_group
from skcomplex.core.system import System
from skcomplex.core.viz import CayleyVisual, default_blueprint


class Parity(Element):
    """Cell that takes the parity of itself and the cells at its generators."""

    alive: float = state(default=0, domain=Domain.BOOLEAN)

    def laws(self, s):
        """Return the parity rule."""
        return [LocalRule(s.alive, Memory(('1', 'a', 'b')), Totalistic({1: 1, 3: 1}))]


class Logs:
    """Recording stand-in that keeps the entity paths, archetypes, and static flags it receives."""

    def __init__(self):
        """Start with no logs."""
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity."""
        self.logs.append((path, archetype, static))


def _simulation():
    """Return a simulation of the parity rule on a ball of radius 3 of the free group of rank 2."""
    space = GroupSpace(finitely_presented_group(['a', 'b']), ball=3)
    cells = np.zeros(space.size)
    cells[0] = 1.0
    return Simulation(System(components=[Parity.lattice('cells', space, alive=cells)])).run(until=2)


def test_cayley_visual_logs_nodes_on_rings_and_edges_once():
    """Test that the nodes sit on rings by word length, colored by state, and the edges are logged once as static."""
    simulation = _simulation()
    problem = simulation.problem_
    visual, logs = CayleyVisual(), Logs()
    for _ in range(2):
        visual.log(logs, problem.elements[0], Frame(0, 0.0, simulation.state_, simulation.parameters_, problem))
    nodes = [archetype for path, archetype, static in logs.logs if not static]
    edges = [archetype for path, archetype, static in logs.logs if static]
    assert len(nodes) == 2
    assert len(edges) == 1
    positions = np.asarray(nodes[0].positions.as_arrow_array().to_pylist())
    space = problem.elements[0].element.space
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1) / 40.0, space.lengths)
    assert len(edges[0].edges.as_arrow_array().to_pylist()) == len(space.edges)


def test_cayley_visual_views_are_graph_views():
    """Test that a group on a group space defaults to the Cayley visual, whose view is a graph view of its state."""
    simulation = _simulation()
    problem = simulation.problem_
    views = CayleyVisual().views(problem.elements[0], problem)
    assert [(type(view).__name__, str(view.origin)) for view in views] == [('GraphView', 'cells/alive')]
    assert simulation.system.elements[0].visual.kind == 'cayley'
    assert default_blueprint(simulation.system) is not None
