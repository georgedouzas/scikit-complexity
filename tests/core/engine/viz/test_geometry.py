"""Test the geometry visual of rewriting systems."""

import rerun.blueprint as rrb

from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._rewriting_compiler import RewritingCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import GeometryVisual, RerunObserver
from skcomplex.core.specs import GeometrySpec


class TreeElement(WordElement):
    """Word of a branching tree."""

    def _get_laws(self, s):
        """Return the branching of a segment."""
        return [ProductionLaw('F', 'F[&F][/^F]')]


def _views(system):
    """Return the view classes of the geometry visual of the first element of a system."""
    problem = RewritingCompiler().prepare(system).run().problem_
    return [type(view) for view in GeometryVisual(path=problem.elements[0].path).prepare(problem).views_]


def test_views_and_logs_of_word_geometries(tmp_path):
    """Test the views of a 3D word geometry and an unbounded word, and that each is logged."""
    tree = CompositeSystem(elements=[TreeElement('tree', axiom='F', geometry=GeometrySpec(angle=30.0, dimension=3))])
    plain = CompositeSystem(elements=[WordElement('word', axiom='F', geometry=GeometrySpec())])
    assert _views(tree) == [rrb.Spatial3DView, rrb.TimeSeriesView]
    assert _views(plain) == [rrb.Spatial2DView, rrb.TimeSeriesView]
    for name, system in [('tree', tree), ('plain', plain)]:
        observer = RerunObserver(name, save_path=tmp_path / f'{name}.rrd', blueprint='default')
        simulate(system, until=2, observers=[observer])
        assert observer.logged_frames_ == 3
