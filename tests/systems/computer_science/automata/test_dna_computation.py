"""Test the `_dna_computation` module of `skcomplex.systems.computer_science.automata`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ParameterError
from skcomplex.systems.computer_science.automata import DNAComputationSystem
from skcomplex.systems.computer_science.automata._dna_computation import (
    ADLEMAN_EDGES,
    _filtered_counts,
    _get_solutions,
)


def test_dna_computation_finds_the_only_hamiltonian_path_of_adlemans_graph():
    """Test that the strands left by Adleman's filters are all the path 0 to 6 in order, and that some are left."""
    trajectory = simulate(DNAComputationSystem(), random_state=0)
    paths = trajectory.get_agents('strands', 60).values['path']
    solved = paths[_get_solutions(paths, 0, 6, 7)]
    assert len(solved) >= 1
    assert np.all(solved == np.arange(7))
    counts = _filtered_counts(paths, 0, 6, 7)
    assert counts[0] >= counts[1] >= counts[2] == len(solved)
    assert trajectory.get_values('solutions')[-1] == len(solved)


def test_dna_computation_leaves_no_strand_without_a_hamiltonian_path():
    """Test that removing the link from 3 to 4 leaves no strand after the filters."""
    edges = [edge for edge in ADLEMAN_EDGES if edge != (3, 4)]
    trajectory = simulate(DNAComputationSystem(edges=edges), random_state=0)
    assert not _get_solutions(trajectory.get_agents('strands', 60).values['path'], 0, 6, 7).any()


def test_dna_computation_rejects_a_start_outside_the_graph():
    """Test that a start that is not a node raises a parameter error."""
    with pytest.raises(ParameterError, match='start'):
        DNAComputationSystem(start=9).prepare()
