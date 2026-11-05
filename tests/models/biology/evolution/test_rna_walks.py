"""Test the `_rna_walks`, `_rna_folds`, and `_neutral_network_analysis` modules of the evolution models."""

import itertools
import multiprocessing

import networkx as nx
import numpy as np
import pytest
from scipy.stats import spearmanr

from skcomplex.engine.runtime import Simulation
from skcomplex.errors import AnalysisError, ValidationError
from skcomplex.models import create
from skcomplex.models.biology.evolution import (
    NeutralNetworkAnalysis,
    fold_sequence,
    mutate_point,
    read_base_pair_distance,
    read_plane_layout,
    read_sequence,
)

START = 'GGGGAAACCCCAUCGAUAGCGGGAAACCGC'
TARGET = '((((...)))).......((((....))))'
HIDE = """
import sys
for name in ('Bio', 'RNA', 'mdtraj'):
    sys.modules[name] = None
"""


def _fold_of(sequence):
    """Return the structure of a sequence."""
    return fold_sequence(sequence)[0]


def _sequences(trajectory):
    """Return the sequence of every walker at every frame."""
    return [[''.join(row) for row in frame] for frame in trajectory.lattice('walkers.base')]


def _run(until=300, **arguments):
    """Return a trajectory of the model."""
    return Simulation(create('rna_neutral_networks', **arguments), random_state=0).prepare().run(until).trajectory_


def test_fold_sequence_gives_the_hairpin():
    """Test that the hairpin folds into its structure at a negative free energy."""
    structure, energy = fold_sequence('GGGGAAACCCC')
    assert structure == '((((...))))'
    assert energy < 0


def test_fold_sequence_is_memoised():
    """Test that a repeated fold comes from the memo."""
    fold_sequence.cache_clear()
    fold_sequence('GGGGAAACCCC')
    fold_sequence('GGGGAAACCCC')
    assert fold_sequence.cache_info().hits == 1


def test_read_base_pair_distance_counts_differing_pairs():
    """Test that the distance counts the pairs that one structure holds and the other does not."""
    assert read_base_pair_distance('((((...))))', '((((...))))') == 0
    assert read_base_pair_distance('((((...))))', '...........') == 4


def test_read_plane_layout_gives_one_point_per_base():
    """Test that bases paired in a stem lie apart from each other, one point per base."""
    positions = read_plane_layout('((((...))))')
    assert positions.shape == (11, 2)
    assert len(np.unique(positions, axis=0)) == 11


def test_mutate_point_changes_one_site_to_the_other_letters_equally():
    """Test that each proposal differs at one site, and each other letter comes with frequency 1/3 within 0.01."""
    rng = np.random.default_rng(0)
    row = rng.integers(0, 4, size=30).astype(float)
    counts = np.zeros(3)
    for _ in range(30000):
        proposal = mutate_point(row.copy(), rng)
        changed = np.flatnonzero(proposal != row)
        assert changed.size == 1
        counts[int((proposal[changed[0]] - row[changed[0]]) % 4) - 1] += 1
    assert counts / counts.sum() == pytest.approx(1 / 3, abs=0.01)


def test_read_sequence_decodes_the_indices():
    """Test that a row of indices reads as its letters."""
    assert read_sequence(np.array([0.0, 1.0, 2.0, 3.0])) == 'ACGU'


def test_neutral_walk_keeps_the_fold():
    """Test that every walker folds into the target at every frame, and drifts more than 8 sites on average."""
    frames = _sequences(_run())
    assert {_fold_of(sequence) for frame in frames for sequence in frame} == {TARGET}
    distances = [sum(a != b for a, b in zip(sequence, START, strict=True)) for sequence in frames[-1]]
    assert np.mean(distances) > 8


def test_neutral_walk_records_a_score_of_zero():
    """Test that the recorded scores are those of the target at every frame."""
    assert np.all(_run(until=50).values('walk.score') == 0)


def test_adaptive_walk_approaches_the_target():
    """Test that the base-pair distance of every walker never grows, and that its mean falls."""
    trajectory = _run(start=None, regime='adaptive')
    scores = trajectory.values('walk.score')
    assert np.all(np.diff(scores, axis=0) <= 0)
    assert scores[-1].mean() < scores[0].mean()
    frames = _sequences(trajectory)
    expected = [read_base_pair_distance(_fold_of(sequence), TARGET) for sequence in frames[-1]]
    assert scores[-1].tolist() == expected


def test_robustness_against_evolvability_of_a_neutral_run():
    """Test that genotype robustness and evolvability pooled over walkers and frames correlate below -0.4."""
    frames = np.array(_sequences(_run()), dtype=object)[::10]
    robustness = NeutralNetworkAnalysis(_fold_of, 'genotype-robustness').prepare(frames).run().values_
    evolvability = NeutralNetworkAnalysis(_fold_of, 'genotype-evolvability').prepare(frames).run().values_
    assert spearmanr(robustness.ravel(), evolvability.ravel()).statistic < -0.4


def test_neutral_network_analysis_reads_a_trajectory():
    """Test that the analysis reads the sequences of a trajectory by the labels of its state."""
    trajectory = _run(until=3, walkers=2)
    analysis = NeutralNetworkAnalysis(_fold_of, 'network', state='walkers.base').prepare(trajectory).run()
    assert list(analysis.networks_) == [TARGET]
    assert START in analysis.networks_[TARGET]


def _toy(kind):
    """Return the analysis of the binary sequences of length 4 mapped to their first letter."""
    genotypes = [''.join(letters) for letters in itertools.product('01', repeat=4)]
    return NeutralNetworkAnalysis(lambda sequence: sequence[0], kind, alphabet='01').prepare(genotypes).run()


def test_toy_map_has_one_component_per_phenotype():
    """Test that each phenotype of the first letter has 8 sequences in one component."""
    networks = _toy('network').networks_
    assert set(networks) == {'0', '1'}
    for graph in networks.values():
        assert graph.number_of_nodes() == 8
        assert nx.number_connected_components(graph) == 1


def test_toy_map_has_robustness_three_and_evolvability_two():
    """Test the robustness and evolvability of the genotypes and phenotypes of the first letter."""
    assert set(_toy('genotype-robustness').values_.tolist()) == {3.0}
    assert set(_toy('genotype-evolvability').values_.tolist()) == {2.0}
    assert _toy('phenotype-robustness').values_ == {'0': 3.0, '1': 3.0}
    assert _toy('phenotype-evolvability').values_ == {'0': 2.0, '1': 2.0}


def test_neutral_network_analysis_joins_letters_of_an_array():
    """Test that an array of letters is read as one sequence per row."""
    analysis = NeutralNetworkAnalysis(lambda sequence: sequence[0], 'genotype-robustness', alphabet='01')
    assert analysis.prepare(np.array([['0', '1'], ['1', '1']])).run().values_.tolist() == [1.0, 1.0]


def test_neutral_network_analysis_rejects_foreign_letters():
    """Test that a letter outside the alphabet raises an error that names it."""
    with pytest.raises(AnalysisError, match='X'):
        NeutralNetworkAnalysis(_fold_of).prepare(['AXGU'])


def test_neutral_network_analysis_rejects_bad_settings():
    """Test that an unknown kind and a map that is not callable raise errors."""
    with pytest.raises(AnalysisError, match='waist'):
        NeutralNetworkAnalysis(_fold_of, 'waist').prepare()
    with pytest.raises(AnalysisError, match='function'):
        NeutralNetworkAnalysis('fold').prepare()


def test_model_rejects_bad_settings():
    """Test that an unknown regime and sequences of another length raise errors."""
    with pytest.raises(ValidationError, match='regime'):
        create('rna_neutral_networks', regime='random')
    with pytest.raises(ValidationError, match='length'):
        create('rna_neutral_networks', start='GGG')


def test_model_without_the_extra_names_it():
    """Test that creating the model without the extra raises an error that names the model and the extra."""
    code = HIDE + """
from skcomplex.errors import MissingExtraError
from skcomplex.models import create
try:
    create('rna_neutral_networks')
except MissingExtraError as error:
    assert 'rna_neutral_networks' in str(error) and "pip install 'scikit-complexity[rna]'" in str(error)
else:
    raise AssertionError
"""
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(code,))
    process.start()
    process.join()
    assert process.exitcode == 0
