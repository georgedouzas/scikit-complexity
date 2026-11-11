"""Test the `_sites` module of `skcomplex.core.engine.viz`."""

from skcomplex.core.components.elements import FieldElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.specs import VisualSpec
from skcomplex.systems.mathematics.networks import NetworkDiffusionSystem
from skcomplex.systems.physics.heat import HeatElement

from .conftest import _views, default_blueprint


def test_fields_on_networks_are_drawn_as_colored_graphs():
    """Test that a field on a network without a visual is drawn as a colored graph, not one series per node."""
    problem = PassCompiler().prepare(NetworkDiffusionSystem().prepare().system_).run().problem_
    assert problem.elements[0].visual.kind == 'sites'
    assert any(type(view).__name__ == 'GraphView' for view in _views(default_blueprint(problem)))


def test_fields_keep_a_given_visual():
    """Test that a field keeps the visual it is given."""
    field = FieldElement('plate', element=HeatElement, space=LatticeSpace((4, 4)), visual=VisualSpec('sites'))
    assert PassCompiler().prepare(CompositeSystem(elements=[field])).run().problem_.elements[0].visual.kind == 'sites'
