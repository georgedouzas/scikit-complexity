"""Test the `_variants` module of `skcomplex.systems.computer_science.automata`."""

from math import comb

import numpy as np
import pytest

from skcomplex.core.errors import ParameterError
from skcomplex.systems import create_simulation
from skcomplex.systems.computer_science.automata import (
    diffusion_automaton,
    elementary_automaton,
    game_of_life,
    highlife,
    noisy_automata,
    unbounded_alphabet_automaton,
)

from ...conftest import _run


def test_highlife_gives_birth_on_six_neighbours():
    """Test that HighLife, unlike Life, brings a dead cell with six live neighbours to life."""
    cells = np.zeros((7, 7))
    cells[2, 2:5] = cells[4, 2:5] = 1
    life = _run(game_of_life((7, 7), initial=cells), 1).lattice('cells.alive')[-1]
    high = _run(highlife((7, 7), initial=cells), 1).lattice('cells.alive')[-1]
    assert life[3, 3] == 0
    assert high[3, 3] == 1


def test_highlife_rejects_a_malformed_rule():
    """Test that a rule outside the notation `B<births>/S<survivals>` raises a parameter error."""
    with pytest.raises(ParameterError, match='B36/S23'):
        highlife((5, 5), rule='36/23')


def test_noisy_automata_follow_the_rule_without_noise_and_randomize_with_full_noise():
    """Test that the noisy automaton is the elementary one without noise and a coin flip at noise 1/2."""
    noiseless = _run(noisy_automata(18, noise=0.0), 20).values('cells.alive')
    exact = _run(elementary_automaton(18, initial=noiseless[0]), 20).values('cells.alive')
    np.testing.assert_array_equal(noiseless, exact)
    random = _run(noisy_automata(18, noise=0.5, size=201), 50).values('cells.alive')[1:]
    assert random.mean() == pytest.approx(0.5, abs=0.02)
    assert 0 < random.mean(axis=1).std() < 0.1


def test_diffusion_automaton_conserves_and_spreads():
    """Test that discrete diffusion conserves the total content on the torus and flattens the square."""
    levels = _run(diffusion_automaton((30, 30), width=6), 50).values('cells.level')
    np.testing.assert_allclose(levels.sum(axis=1), 36.0)
    assert levels[-1].max() < 0.5 * levels[0].max()


def test_unbounded_alphabet_automaton_builds_pascals_triangle():
    """Test that the additive automaton over the integers holds the binomial coefficients of its step."""
    values = _run(unbounded_alphabet_automaton(81), 40).values('cells.value')[-1]
    np.testing.assert_array_equal(values[40:81], [comb(40, k) for k in range(41)])


def test_fredkin_replicator_copies_the_pattern_four_times_at_powers_of_two():
    """Test that at the times 16 and 32 the state is the sum modulo 2 of the pattern shifted by that many cells."""
    trajectory = create_simulation('fredkin_replicator').prepare().run().trajectory_
    cells = trajectory.lattice('cells.alive')
    for time in (16, 32):
        shifted = [np.roll(cells[0], shift, axis) for shift in (time, -time) for axis in (0, 1)]
        assert np.array_equal(cells[time], sum(shifted) % 2)
        assert cells[time].sum() == 4 * cells[0].sum()
