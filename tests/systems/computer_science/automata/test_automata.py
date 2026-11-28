"""Test the `_automata` module of `skcomplex.systems.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.systems.computer_science.automata import LangtonsAntSystem

PERIOD = 104


def test_langtons_ant_builds_the_highway():
    """Test that Langton's ant moves two cells along a diagonal and adds 12 black cells every 104 steps at the end."""
    until = 116 * PERIOD
    trajectory = LangtonsAntSystem((120, 120)).prepare().simulate(until=until, dt=PERIOD)
    rows = trajectory.get_values('ant.row').ravel()
    columns = trajectory.get_values('ant.column').ravel()
    black = np.array([colors.sum() for colors in trajectory.get_values('ant.colors')[:, 0]])
    steps = np.column_stack([np.diff(rows[-4:]), np.diff(columns[-4:])])
    assert np.all(np.abs(steps) == 2)
    assert np.all(steps == steps[0])
    assert np.all(np.diff(black[-4:]) == 12)


def test_langtons_ant_flips_the_cell_it_leaves():
    """Test that every step flips the colour of the cell the ant stood on and of no other cell."""
    trajectory = LangtonsAntSystem((15, 15)).prepare().simulate(until=200, dt=1)
    colors = np.stack(trajectory.get_values('ant.colors')[:, 0])
    rows = trajectory.get_values('ant.row').ravel()
    columns = trajectory.get_values('ant.column').ravel()
    changed = np.argwhere(np.diff(colors.astype(int), axis=0) != 0)
    assert changed.tolist() == [[step, rows[step], columns[step]] for step in range(200)]


def test_langtons_ant_starts_on_given_colors():
    """Test that the ant turns left on a black start cell and right on a white one, from its start and heading."""
    colors = np.zeros((10, 10))
    colors[2, 3] = 1
    black = LangtonsAntSystem((10, 10), colors=colors, start=(2, 3), heading=1).prepare()
    white = LangtonsAntSystem((10, 10), start=(2, 3), heading=1).prepare()
    left = black.simulate(until=1, dt=1)
    right = white.simulate(until=1, dt=1)
    assert left.get_values('ant.heading').ravel().tolist() == [1, 0]
    assert (left.get_values('ant.row')[-1, 0], left.get_values('ant.column')[-1, 0]) == (1, 3)
    assert right.get_values('ant.heading').ravel().tolist() == [1, 2]
    assert (right.get_values('ant.row')[-1, 0], right.get_values('ant.column')[-1, 0]) == (3, 3)


def test_langtons_ant_wraps_around_the_torus():
    """Test that the ant leaving the first row enters the last one."""
    colors = np.ones((5, 5))
    trajectory = LangtonsAntSystem((5, 5), colors=colors, start=(0, 0), heading=1).prepare().simulate(until=1, dt=1)
    assert trajectory.get_values('ant.row')[-1, 0] == 4


@pytest.mark.parametrize(
    'arguments',
    [{'colors': np.zeros((4, 5))}, {'colors': np.full((5, 5), 2)}, {'start': (5, 0)}],
)
def test_langtons_ant_rejects_bad_colors_and_starts(arguments):
    """Test that colours of another shape or of other values and a start outside the torus raise an error."""
    with pytest.raises(ValidationError):
        LangtonsAntSystem((5, 5), **arguments).prepare()
