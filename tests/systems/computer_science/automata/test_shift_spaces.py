"""Test the `_shift_spaces` module of `skcomplex.systems.computer_science.automata`."""

import re

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ParameterError
from skcomplex.systems.computer_science.automata import (
    AddingMachineSystem,
    ContextFreeSubshiftSystem,
    GapSubshiftSystem,
    MorseSubshiftSystem,
)

from ..conftest import _run

GOLDEN_RATIO = (1 + 5**0.5) / 2
BLOCK = re.compile('1*2*')


def _gaps(cells):
    """Return the runs of zeros between consecutive ones of a window."""
    text = ''.join(str(int(cell)) for cell in cells).strip('0')
    return [len(run) for run in text.split('1')[1:-1]] if text else []


@pytest.mark.parametrize('gaps', [None, (1, 4), (0, 3, 5)])
def test_gap_subshift_keeps_its_gaps_in_the_set(gaps):
    """Test that every window the shift visits has its gaps between ones in the set, the even numbers by default."""
    values = _run(GapSubshiftSystem(80, gaps=gaps), 120).values('cells.alive')
    allowed = (lambda gap: gap % 2 == 0) if gaps is None else (lambda gap: gap in gaps)
    for cells in values:
        assert all(allowed(gap) for gap in _gaps(cells))


def test_gap_subshift_shifts_the_window():
    """Test that every step moves the window one cell to the left."""
    values = _run(GapSubshiftSystem(60), 5).values('cells.alive')
    np.testing.assert_array_equal(values[1:, :-1], values[:-1, 1:])


def test_even_subshift_has_the_density_of_its_maximal_entropy_measure():
    """Test that the density of ones of the even subshift is `1 / E[h + 1]` for gaps of probability `φ^-(h + 1)`."""
    values = _run(GapSubshiftSystem(200, random_state=1), 2000).values('cells.alive')
    lengths = np.arange(0, 400, 2)
    mean = np.sum((lengths + 1) * GOLDEN_RATIO ** -(lengths + 1.0))
    assert values.mean() == pytest.approx(1 / mean, abs=0.02)


def test_gap_subshift_rejects_an_empty_set():
    """Test that a set without any gap below the cap raises a parameter error."""
    with pytest.raises(ParameterError, match='at least one allowed gap'):
        GapSubshiftSystem(10, gaps=(1000,)).prepare()


def _blocks_balanced(ring):
    """Return whether every block between consecutive zeros of a ring is no `1^h 2^k` with `h != k`."""
    zeros = np.flatnonzero(ring == 0)
    if zeros.size == 0:
        return True
    text = ''.join(str(int(symbol)) for symbol in np.roll(ring, -zeros[0]))
    blocks = text[1:].split('0')
    return all(block.count('1') == block.count('2') for block in blocks if BLOCK.fullmatch(block))


def test_context_free_subshift_stays_admissible_and_moves():
    """Test that the chain only visits rings without unbalanced blocks, and that it changes them."""
    values = _run(ContextFreeSubshiftSystem(40), 30).values('cells.symbol')
    assert all(_blocks_balanced(ring) for ring in values)
    assert (values[-1] != values[0]).any()
    with pytest.raises(ParameterError, match='h != k'):
        ContextFreeSubshiftSystem(5, initial=[0, 1, 0, 0, 0]).prepare()


def test_morse_subshift_builds_the_thue_morse_prefixes():
    """Test that generation `n` is `T(0) ... T(2^n - 1)`, the parity of the ones of every binary expansion."""
    trajectory = simulate(MorseSubshiftSystem(), until=8)
    assert trajectory.values('morse.size').tolist() == [2.0**n for n in range(9)]
    word = ''.join(str(code) for code in trajectory.structure('morse', 8).codes)
    assert word == ''.join(str(bin(k).count('1') % 2) for k in range(256))


def test_adding_machine_counts_through_every_residue():
    """Test that the digits read `t mod q^L` at step `t`, so the orbit visits every vertex of level `L` in turn."""
    for base, digits in ((2, 5), (3, 3)):
        values = _run(AddingMachineSystem(digits, base=base), base**digits).values('cells.digit')
        numbers = values @ base ** np.arange(digits)
        np.testing.assert_array_equal(numbers, np.arange(base**digits + 1) % base**digits)
    with pytest.raises(ParameterError, match='at least 2'):
        AddingMachineSystem(base=1).prepare()
