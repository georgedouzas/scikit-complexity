"""Test the `_base` module of `skcomplex.core.neighborhoods`."""

from types import SimpleNamespace

from skcomplex.core.neighborhoods import BaseNeighborhood


class SidesNeighborhood(BaseNeighborhood):
    """Neighbourhood of the sites on either side of a site of a ring."""

    def __init__(self, steps=(-1, 1), *, label='neighborhood'):
        super().__init__(label)
        self.steps = steps


def test_neighbourhood_contributes_no_laws():
    """Test that a neighbourhood checks, prepares, and runs, contributing no laws."""
    assert not hasattr(SidesNeighborhood().check().prepare().run(SimpleNamespace()), 'laws_')
