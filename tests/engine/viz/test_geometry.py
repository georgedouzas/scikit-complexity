"""Test the geometry visual of rewriting systems."""

import rerun.blueprint as rrb

from skcomplex.core.elements import WordElement
from skcomplex.core.laws import ProductionLaw
from skcomplex.core.system import System
from skcomplex.engine.compile import RewritingCompiler, compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import GeometryVisual, RerunObserver
from skcomplex.specs import GeometrySpec


class TreeElement(WordElement):
    """Word of a branching tree."""

    def _get_laws(self, s):
        """Return the branching of a segment."""
        return [ProductionLaw('F', 'F[&F][/^F]')]


def _views(system):
    """Return the view classes of the geometry visual of the first element of a system."""
    problem = compile_system(system, compiler=RewritingCompiler())
    return [type(view) for view in GeometryVisual(path=problem.elements[0].path).check().prepare(problem).views_]


def test_views_and_logs_of_word_geometries(tmp_path):
    """Test the views of a 3D word geometry and an unbounded word, and that each is logged."""
    tree = System(elements=[TreeElement('tree', axiom='F', geometry=GeometrySpec(angle=30.0, dimension=3))])
    plain = System(elements=[WordElement('word', axiom='F', geometry=GeometrySpec())])
    assert _views(tree) == [rrb.Spatial3DView, rrb.TimeSeriesView]
    assert _views(plain) == [rrb.Spatial2DView, rrb.TimeSeriesView]
    for name, system in [('tree', tree), ('plain', plain)]:
        observer = RerunObserver(name, save_path=tmp_path / f'{name}.rrd', blueprint='default')
        Simulation(system, observers=[observer], compiler=RewritingCompiler()).check().prepare().run(until=2)
        assert observer.logged_frames_ == 3
