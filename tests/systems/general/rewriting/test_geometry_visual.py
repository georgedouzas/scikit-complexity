"""Test the geometry visual of rewriting systems."""

import rerun.blueprint as rrb

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.systems import System
from skcomplex.systems.general.rewriting import (
    GeometrySpec,
    GeometryVisual,
    ProductionLaw,
    RewritingInteraction,
    WordElement,
)
from skcomplex.systems.general.rewriting._rewriting_compiler import RewritingCompiler


def _views(system):
    """Return the view classes of the geometry visual of the first element of a system."""
    problem = RewritingCompiler().prepare(system).run().problem_
    return [type(view) for view in GeometryVisual(path=problem.elements[0].path).prepare(problem).views_]


def test_views_and_logs_of_word_geometries(tmp_path):
    """Test the views of a 3D word geometry and an unbounded word, and that each is logged."""
    branching = RewritingInteraction('branching', target='tree', laws=[ProductionLaw('F', 'F[&F][/^F]')])
    tree = System(
        elements=[WordElement('tree', axiom='F', geometry=GeometrySpec(angle=30.0, dimension=3))],
        interactions=[branching],
    )
    plain = System(elements=[WordElement('word', axiom='F', geometry=GeometrySpec())])
    assert _views(tree) == [rrb.Spatial3DView, rrb.TimeSeriesView]
    assert _views(plain) == [rrb.Spatial2DView, rrb.TimeSeriesView]
    for name, system in [('tree', tree), ('plain', plain)]:
        observer = RerunObserver(name, save_path=tmp_path / f'{name}.rrd', blueprint='default')
        system.prepare().simulate(until=2, observers=[observer])
        assert observer.logged_frames_ == 3
