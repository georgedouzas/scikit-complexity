"""Test the `_fuzzy_oil_drop`, `_backbone_visual`, `_ellipse_path`, and `_early_stage` modules of the systems."""

import multiprocessing
import warnings

import numpy as np
import pytest
from rerun.chunk import RrdReader

from skcomplex.core.analysis import chain_shape
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.evolution import RNANeutralNetworksSystem
from skcomplex.systems.biology.systems_biology import (
    AMINO_ACIDS,
    BackboneGeometry,
    EarlyStageFoldingIntermediateSystem,
    EllipsePath,
    FuzzyOilDropSystem,
    fuzzy_oil_drop_measures,
    read_cell_probabilities,
    read_native_pairs,
    read_structure,
)

HIDE = """
import sys
EXTRAS = ('Bio', 'RNA', 'mdtraj')
for name in EXTRAS:
    sys.modules[name] = None
"""
SYSTEM_WITHOUT_EXTRA = HIDE + """
from skcomplex.core.errors import MissingExtraError
from skcomplex.systems.biology.systems_biology import {system}
try:
    {system}().prepare()
except MissingExtraError as error:
    message = str(error)
    assert {name!r} in message and "'structures'" in message
    assert "pip install 'scikit-complexity[structures]'" in message
else:
    raise AssertionError
"""


def _exit_code_of(code):
    """Return the exit code of a program in a fresh interpreter."""
    process = multiprocessing.get_context('spawn').Process(target=exec, args=(code,))
    process.start()
    process.join()
    return process.exitcode


def test_ensemble_of_the_trp_cage():
    """Test 38 frames, every `RD` in `[0, 1]`, equal within 1e-12 to the `RD` of the analysis of each system."""
    trajectory = simulate(FuzzyOilDropSystem())
    assert len(trajectory.times) == 38
    recorded = trajectory.values('RD')
    assert recorded.min() >= 0
    assert recorded.max() <= 1
    expected = fuzzy_oil_drop_measures(structure=read_structure('1L2Y')).values
    assert recorded == pytest.approx(expected, abs=1e-12)


def test_ensemble_records_the_environment_factor():
    """Test that the recorded `K` of each frame equals the one of the analysis."""
    structure = read_structure('1L2Y')
    recorded = simulate(FuzzyOilDropSystem(), until=3).values('K')
    expected = fuzzy_oil_drop_measures(structure=structure, kind='environment-factor').values[:4]
    assert recorded == pytest.approx(expected, abs=1e-12)


def test_ensemble_positions_replay_the_effective_atoms():
    """Test that the positions at frame 10 are the effective atoms of system 10."""
    frames = simulate(FuzzyOilDropSystem(), until=12).lattice('chain.x')
    assert frames[10] == pytest.approx(read_structure('1L2Y').effective_atoms[10])


def test_analysis_reads_the_replay_of_a_trajectory():
    """Test that the analysis reads a trajectory of the replay."""
    structure = read_structure('1L2Y')
    trajectory = simulate(FuzzyOilDropSystem(), until=2)
    values = fuzzy_oil_drop_measures(trajectory, structure=structure, state='chain.x').values
    assert values == pytest.approx(trajectory.values('RD'), abs=1e-12)


def test_folding_lowers_the_relative_distance_and_compacts_the_chain():
    """Test that `RD` never grows at temperature zero, and that the radius of gyration falls from an extended start."""
    system = FuzzyOilDropSystem(dynamics='folding').prepare().system_
    trajectory = simulate(system, random_state=0, until=300)
    relative_distance = trajectory.values('folding.score')[:, 0]
    assert np.all(np.diff(relative_distance) <= 1e-12)
    assert relative_distance[-1] < relative_distance[0]
    structure = read_structure('1L2Y')
    geometry = BackboneGeometry(structure).prepare()
    angles = trajectory.lattice('chain.angle').astype(float)[[0, -1], 0]
    chains = np.array([geometry.run(row).read_effective_atoms() for row in angles])
    radii = chain_shape(chains).values
    assert radii[1] < radii[0]


def test_folding_at_a_positive_temperature_records_rd():
    """Test that the folding at a positive temperature records the `RD` of its angles."""
    trajectory = simulate(FuzzyOilDropSystem(dynamics='folding', temperature=0.1, proposals=2), until=2, random_state=0)
    assert trajectory.values('RD') == pytest.approx(trajectory.values('folding.score')[:, 0], abs=1e-9)


def test_system_rejects_unknown_dynamics():
    """Test that unknown dynamics raise an error."""
    with pytest.raises(ValidationError, match='dynamics'):
        FuzzyOilDropSystem(dynamics='melting').prepare()


def _column_of(path, entity, component):
    """Return the rows that a recording holds of a component of an entity, in the order of the log calls."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        chunks = RrdReader(path).stream().to_chunks()
    for chunk in chunks:
        if chunk.entity_path != entity:
            continue
        batch = chunk.to_record_batch()
        rows += [
            row
            for name in batch.schema.names
            if name.endswith(component)
            for row in batch.column(name).to_pylist()
            if row is not None
        ]
    return rows


def test_protein_chain_streams_points_and_one_strip(tmp_path):
    """Test that the recording holds 20 three-dimensional points and one line strip at every frame."""
    path = tmp_path / 'run.rrd'
    system = FuzzyOilDropSystem().prepare().system_
    observer = RerunObserver('fuzzy_oil_drop', save_path=path)
    simulate(system, until=4, observers=[observer], random_state=0)
    points = _column_of(path, '/chain', ':positions')
    strips = _column_of(path, '/chain/strip', ':strips')
    assert len(points) == 5
    assert len(strips) == 5
    for frame, strip in zip(points, strips, strict=True):
        assert np.shape(frame) == (20, 3)
        assert len(strip) == 1
        assert np.shape(strip[0]) == (20, 3)


def test_folding_chain_streams_rebuilt_points(tmp_path):
    """Test that the folding draws 20 points of the chain rebuilt from its angles."""
    path = tmp_path / 'run.rrd'
    system = FuzzyOilDropSystem(dynamics='folding').prepare().system_
    observer = RerunObserver('fuzzy_oil_drop', save_path=path)
    simulate(system, until=2, observers=[observer], random_state=0)
    points = _column_of(path, '/chain', ':positions')
    assert len(points) == 3
    assert np.shape(points[0]) == (20, 3)


def test_sequences_stream_as_an_image_of_four_colors(tmp_path):
    """Test that 8 walkers of length 30 stream as an 8 by 30 image with four colors at every frame."""
    path = tmp_path / 'run.rrd'
    observer = RerunObserver('rna_neutral_networks', save_path=path)
    system = RNANeutralNetworksSystem(start='GGGGAAACCCCAUCGAUAGCGGGAAACCGC').prepare().system_
    simulate(system, until=40, observers=[observer], random_state=0)
    frames = _column_of(path, '/walkers/base', ':buffer')
    assert len(frames) == 41
    colors = set()
    for frame in frames:
        pixels = np.asarray(frame[0], dtype=np.uint8)
        assert pixels.size == 8 * 30 * 3
        colors |= {tuple(pixel) for pixel in pixels.reshape(-1, 3).tolist()}
    assert len(colors) == 4


@pytest.mark.parametrize(
    ('system', 'name'),
    [
        ('FuzzyOilDropSystem', 'fuzzy_oil_drop'),
        ('EarlyStageFoldingIntermediateSystem', 'early_stage_folding_intermediate'),
    ],
)
def test_protein_system_without_the_extra_names_it(system, name):
    """Test that the error of a missing extra names the system, the extra, and the install command."""
    assert _exit_code_of(SYSTEM_WITHOUT_EXTRA.format(system=system, name=name)) == 0


def test_default_install_lists_every_system_without_the_extras():
    """Test that importing the package and listing its systems without the extras imports none of the extras."""
    code = HIDE + """
import skcomplex.core.analysis, skcomplex.core.components, skcomplex.core.engine
from skcomplex.systems import all_systems
names = {name for name, _ in all_systems(field='biology/systems_biology')}
names |= {name for name, _ in all_systems(field='biology/evolution')}
assert {'FuzzyOilDropSystem', 'RNANeutralNetworksSystem', 'EarlyStageFoldingIntermediateSystem'} <= names
loaded = [name for name in sys.modules if name.split('.')[0] in EXTRAS and sys.modules[name] is not None]
assert loaded == [], loaded
"""
    assert _exit_code_of(code) == 0


def test_projection_lies_on_the_path_and_is_the_closest_point():
    """Test that each projected pair of BPTI lies on the path, and no point of 3600 samples is closer to the native."""
    _, native = read_native_pairs(read_structure('5PTI'))
    path = EllipsePath().prepare()
    path.run(native)
    samples = path.read_points(np.linspace(0, 2 * np.pi, 3600, endpoint=False))
    on_path = path.read_points(path.parameters_)
    assert np.abs(on_path - path.angles_).max() < 1e-9
    for pair, projected in zip(native, path.angles_, strict=True):
        wrapped = (pair - samples + np.pi) % (2 * np.pi) - np.pi
        nearest = np.hypot(*wrapped.T).min()
        difference = (pair - projected + np.pi) % (2 * np.pi) - np.pi
        assert np.hypot(*difference) <= nearest + 1e-12


def test_ellipse_starts_at_the_lower_right_and_turns_clockwise():
    """Test the orientation of Fig. 3.43 of the book and the convention of this class for the origin of the path."""
    points = EllipsePath().prepare().read_points(np.array([0.0, np.pi / 2]))
    assert points[0, 0] > 0 > points[0, 1]
    assert points[1, 0] < 0
    assert points[1, 1] < 0


def test_ellipse_rejects_bad_settings_and_pairs():
    """Test that a diagonal of zero and angles that are not pairs raise errors."""
    with pytest.raises(ValidationError, match='positive'):
        EllipsePath(long_diagonal=0.0).prepare()
    with pytest.raises(ValidationError, match='pairs'):
        EllipsePath().prepare().run(np.zeros(3))


def test_sampled_cells_follow_the_probabilities_of_the_sequence():
    """Test that each residue's frequency of each cell over the last 200 frames matches the probability within 0.15."""
    trajectory = simulate(EarlyStageFoldingIntermediateSystem(), random_state=0, until=300)
    cells = trajectory.lattice('residues.cell').astype(int)[-200:, 0]
    probabilities = read_cell_probabilities()
    names = [residue for _, _, residue in read_structure('5PTI').residues]
    for site, name in enumerate(names):
        frequencies = np.bincount(cells[:, site], minlength=72) / 200
        assert np.abs(frequencies - probabilities[AMINO_ACIDS.index(name)]).max() < 0.15


def test_early_stage_rejects_a_temperature_of_zero():
    """Test that the system takes a positive temperature."""
    with pytest.raises(ValidationError, match='temperature'):
        EarlyStageFoldingIntermediateSystem(temperature=0.0).prepare()
