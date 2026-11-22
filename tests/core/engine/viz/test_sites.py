"""Test the `_sites` module of `skcomplex.core.engine.viz`."""

from skcomplex.core.components.elements import VisualSpec
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System
from skcomplex.systems.mathematics.networks import NetworkDiffusionSystem
from skcomplex.systems.physics.heat import HeatElement

from .conftest import _views, default_blueprint


def test_fields_on_networks_are_drawn_as_colored_graphs():
    """Test that a field on a network without a visual is drawn as a colored graph, not one series per node."""
    problem = NetworkDiffusionSystem().prepare().system_._problem
    assert problem.elements[0].visual.kind == 'sites'
    assert any(type(view).__name__ == 'GraphView' for view in _views(default_blueprint(problem)))


def test_fields_keep_the_visual_of_their_member_class():
    """Test that a field is drawn as its member class says, not as the image of its lattice."""

    class SitedHeatElement(HeatElement):
        """Heat drawn on sites."""

        visual = VisualSpec('sites')

    field = SitedHeatElement('plate', space=LatticeSpace((4, 4)))
    assert System(elements=[field]).prepare()._problem.elements[0].visual.kind == 'sites'
