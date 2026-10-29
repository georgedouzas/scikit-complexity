"""Test the `_base` module of `skcomplex.core.spaces`."""

from types import SimpleNamespace

import numpy as np
import pytest

from skcomplex.core.spaces import BaseNeighborhood, BaseSpace, BaseTime
from skcomplex.errors import NotCheckedError, SpaceError


class SidesNeighborhood(BaseNeighborhood):
    """Neighbourhood of the sites on either side of a site of a ring."""

    def __init__(self, steps=(-1, 1), *, label='neighborhood'):
        super().__init__(label)
        self.steps = steps


class RingSpace(BaseSpace):
    """Ring of sites whose neighbours are the sites on either side."""

    def __init__(self, n=4, *, label='space'):
        super().__init__(label)
        self.n = n

    def _prepare(self, context=None):
        """Set the facts of the ring."""
        self.size_, self.n_dim_, self.shape_ = self.n, 1, (self.n,)
        self.positions_ = np.arange(self.n, dtype=float)[:, None]
        self.periodic_, self.default_visual_ = True, 'sites'

    def _default_neighborhood(self):
        """Return the two sides."""
        return SidesNeighborhood()

    def _index_table(self, neighborhood):
        """Return the site at every step of every site."""
        return np.array([[(site + step) % self.n for step in neighborhood.steps] for site in range(self.n)])


class EmptySpace(BaseSpace):
    """Space without sites that implements none of the neighbourhood hooks."""

    def __init__(self, *, label='space'):
        super().__init__(label)

    def _prepare(self, context=None):
        """Set no sites."""
        self.size_ = None


class TickTime(BaseTime):
    """Time that takes the frames of the base."""

    def __init__(self, *, label='time'):
        super().__init__(label)

    def _prepare(self, context=None):
        """Set no schedule."""
        self.schedule_ = None


def test_space_reads_the_neighbours_of_a_site_through_its_hooks():
    """Test that a space written with the table hooks gives the neighbours of the default or a given neighbourhood."""
    space = RingSpace().check().prepare()
    assert (space.size_, space.n_dim_, space.periodic_) == (4, 1, True)
    assert space._get_neighbors(0).tolist() == [3, 1]
    assert space._get_neighbors(1, SidesNeighborhood((2,))).tolist() == [3]


def test_space_counts_its_sites_only():
    """Test that a space counts its sites as parts, and has no edges."""
    space = RingSpace().check().prepare()
    assert space._count_parts('sites') == 4
    with pytest.raises(SpaceError, match='no edges'):
        space._count_parts('edges')


def test_space_contributes_no_laws_and_lacks_nothing():
    """Test that a space returns no laws when run and lacks no option of lattices by default."""
    space = RingSpace().check().prepare()
    assert space.run(SimpleNamespace()).laws_ == []
    assert space._get_lacks() == frozenset()


@pytest.mark.parametrize(
    ('hook', 'arguments'),
    [('_default_neighborhood', ()), ('_index_table', (None,)), ('_ball', (1,)), ('_neighbor_pairs', ())],
)
def test_space_without_sites_answers_no_neighbourhood_hook(hook, arguments):
    """Test that a space without sites raises a space error for every neighbourhood hook."""
    with pytest.raises(SpaceError, match='no sites'):
        getattr(EmptySpace().check().prepare(), hook)(*arguments)


def test_space_without_sites_has_no_parts():
    """Test that a space without sites has no sites to count."""
    with pytest.raises(SpaceError, match='no sites'):
        EmptySpace().check().prepare()._count_parts('sites')


def test_space_must_be_checked_before_it_is_prepared():
    """Test that preparing a space that was not checked raises an error."""
    with pytest.raises(NotCheckedError):
        RingSpace().prepare()


def test_neighbourhood_contributes_no_laws():
    """Test that a neighbourhood checks, prepares, and runs into no laws."""
    assert SidesNeighborhood().check().prepare().run(SimpleNamespace()).laws_ == []


def test_base_time_runs_into_no_laws_and_frames_its_ends():
    """Test that the base time contributes no laws and frames the start and the end of a run."""
    time = TickTime().check().prepare()
    assert time.schedule_ is None
    assert time.run(SimpleNamespace()).laws_ == []
    assert time._get_frames(2.0).tolist() == [0.0, 2.0]
