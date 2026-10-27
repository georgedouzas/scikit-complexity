"""Test the `_bases` module of `skcomplex.core.spaces`."""

import numpy as np
import pytest

from skcomplex.core.base import SpaceKind, TimeKind
from skcomplex.core.errors import SpaceError
from skcomplex.core.spaces import BaseSpace, BaseTime


class RingSpace(BaseSpace):
    """Ring of sites whose neighbours are the sites on either side."""

    space_kind = SpaceKind.LATTICE

    def __init__(self, n=4):
        self.n = n

    def _prepare(self):
        """Set the facts of the ring."""
        self.size_, self.n_dim_, self.shape_ = self.n, 1, (self.n,)
        self.positions_ = np.arange(self.n, dtype=float)[:, None]
        self.periodic_, self.default_visual_ = True, 'sites'

    def _default_neighborhood(self):
        """Return the two sides."""
        return (-1, 1)

    def _index_table(self, neighborhood):
        """Return the site at every offset of every site."""
        return np.array([[(site + step) % self.n for step in neighborhood] for site in range(self.n)])


class EmptySpace(BaseSpace):
    """Space that implements none of the neighbourhood hooks."""

    space_kind = SpaceKind.CONTINUOUS


class TickTime(BaseTime):
    """Time that takes the base's fit and run."""

    kind = TimeKind.CONTINUOUS


def test_child_space_runs_through_the_hooks_of_the_base():
    """Test that a space written with `_prepare` and the table hooks is prepared and run by the verbs of the base."""
    space = RingSpace().prepare()
    assert (space.size_, space.n_dim_, space.periodic_) == (4, 1, True)
    assert space.run(0).tolist() == [3, 1]
    assert space.run(1, (2,)).tolist() == [3]


@pytest.mark.parametrize(
    ('hook', 'argument'),
    [('_default_neighborhood', ()), ('_index_table', (None,)), ('_ball', (1,)), ('_neighbor_pairs', ())],
)
def test_space_without_sites_answers_no_neighbourhood_hook(hook, argument):
    """Test that a space without sites raises a space error for every neighbourhood hook."""
    with pytest.raises(SpaceError, match='no sites'):
        getattr(EmptySpace().prepare(), hook)(*argument)


def test_base_time_prepares_its_kind_and_runs_its_ends():
    """Test that the base time prepares its kind without a schedule and runs the start and the end."""
    time = TickTime().prepare()
    assert (time.time_kind_, time.schedule_) == (TimeKind.CONTINUOUS, None)
    assert time.run(2.0).tolist() == [0.0, 2.0]
