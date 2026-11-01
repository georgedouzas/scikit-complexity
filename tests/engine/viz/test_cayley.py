"""Test the Cayley graph visual of lattice groups on group spaces."""

from typing import Annotated

import numpy as np
from sage.all__sagemath_gap import libgap
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.elements import Element, LatticeElement
from skcomplex.core.laws import LocalRuleLaw
from skcomplex.core.neighborhoods import MemoryNeighborhood
from skcomplex.core.spaces import CayleySpace
from skcomplex.core.system import System
from skcomplex.engine.problems import FrameSpec
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import CayleyVisual, RerunObserver
from skcomplex.specs import StateSpec, TotalisticSpec, VisualSpec


class ParityElement(Element):
    """Cell that takes the parity of itself and the cells at its generators."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.alive = alive

    def _get_laws(self, s):
        """Return the parity rule."""
        return [LocalRuleLaw(s.alive, MemoryNeighborhood(('1', 'a', 'b')), TotalisticSpec({1: 1, 3: 1}))]


class Logs:
    """Recording stand-in that keeps the entity paths, archetypes, and static flags it receives."""

    def __init__(self):
        self.logs = []

    def log(self, path, archetype, static=False):
        """Keep a logged entity."""
        self.logs.append((path, archetype, static))


def _simulation():
    """Return a simulation of the parity rule on a ball of radius 3 of the free group of rank 2."""
    space = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=3).check().prepare()
    cells = np.zeros(space.size_)
    cells[0] = 1.0
    return (
        Simulation(
            System(elements=[LatticeElement('cells', element=ParityElement, space=space, values={'alive': cells})]),
        )
        .check()
        .prepare()
        .run(until=2)
    )


def test_cayley_visual_logs_nodes_on_rings_and_edges_once():
    """Test that the nodes sit on rings by word length, colored by state, and the edges are logged once as static."""
    simulation = _simulation()
    problem = simulation.problem_
    logs = Logs()
    visual = CayleyVisual(path=problem.elements[0].path, recording=logs).check().prepare(problem)
    for _ in range(2):
        visual.run(FrameSpec(0, 0.0, simulation.state_, simulation.parameters_, problem))
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
    views = CayleyVisual(path=problem.elements[0].path).check().prepare(problem).views_
    assert [(type(view).__name__, str(view.origin)) for view in views] == [('GraphView', 'cells/alive')]
    assert problem.elements[0].visual.kind == 'cayley'
    assert RerunObserver(blueprint='default').check().prepare(problem).blueprint_ is not None
