"""Test the `_brusselator` module of `skcomplex.models.chemistry.common.kinetics`."""

from skcomplex.models.chemistry.common.kinetics import BrusselatorReactionElement


def test_brusselator_reaction_element_holds_intermediates_and_supplies():
    """Test that the reactor holds the intermediates `u` and `v` and the supplies `a` and `b`."""
    reactor = BrusselatorReactionElement('reactor', u=0.5, b=2.0)
    assert (reactor.u, reactor.v, reactor.a, reactor.b) == (0.5, 1.0, 1.0, 2.0)
