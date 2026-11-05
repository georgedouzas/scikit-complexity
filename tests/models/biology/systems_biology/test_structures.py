"""Test the modules of structures, backbone geometry, and the fuzzy oil drop analysis of the systems biology models."""

import dataclasses
import warnings

import numpy as np
import pytest

from skcomplex.analysis import ShapeAnalysis
from skcomplex.errors import AnalysisError, ValidationError
from skcomplex.models.biology.systems_biology import (
    BackboneGeometry,
    FuzzyOilDropAnalysis,
    StructureSpec,
    read_kyte_doolittle_scale,
    read_structure,
    read_trajectory,
    write_structure,
)

# The curves T and O of Fig. 1.41 of Konieczny et al. (2023), read in pixels above the axis at 29 even steps across the
# plot. One pixel is added to each, since the drawn curve T touches the axis, where a divergence has no value.
FIGURE_THEORETICAL = np.array(
    [1, 2, 3, 4, 4, 10, 17, 30, 46, 74, 98, 129, 156, 176, 182, 176, 156, 128, 103, 72, 45, 30, 18, 10, 5, 4, 3, 2, 2],
    dtype=float,
)
FIGURE_OBSERVED = np.array(
    [
        14,
        18,
        24,
        40,
        52,
        63,
        67,
        66,
        64,
        63,
        68,
        78,
        91,
        102,
        113,
        116,
        116,
        106,
        94,
        80,
        65,
        52,
        43,
        28,
        23,
        18,
        14,
        12,
        11,
    ],
    dtype=float,
)


@pytest.mark.parametrize(
    ('entry', 'models', 'residues', 'atoms'),
    [('1L2Y', 38, 20, 304), ('1UBQ', 1, 76, 602), ('5PTI', 1, 58, None)],
)
def test_read_structure_holds_the_polymer_of_every_model(entry, models, residues, atoms):
    """Test the models, residues, and atoms of the bundled entries, with 1L2Y at 38 models, 20 residues, 304 atoms."""
    structure = read_structure(entry)
    assert structure.coordinates.shape[0] == models
    assert len(structure.residues) == residues
    if atoms is not None:
        assert structure.coordinates.shape[1] == atoms


@pytest.mark.parametrize('suffix', ['.cif', '.pdb'])
def test_write_structure_round_trip(tmp_path, suffix):
    """Test that a written copy has the same residues, atom names, and coordinates within 0.001 angstrom."""
    structure = read_structure('1L2Y')
    path = tmp_path / f'copy{suffix}'
    write_structure(structure, path)
    copy = read_structure(path)
    assert copy.residues == structure.residues
    assert copy.atom_names == structure.atom_names
    assert np.abs(copy.coordinates - structure.coordinates).max() < 0.001


def test_write_structure_rejects_an_unknown_format(tmp_path):
    """Test that a path of another suffix raises an error."""
    with pytest.raises(ValidationError, match='suffix'):
        write_structure(read_structure('1UBQ'), tmp_path / 'copy.txt')


def test_read_structure_rejects_an_unknown_source():
    """Test that a name that is no entry and no file raises an error."""
    with pytest.raises(ValidationError, match='bundled'):
        read_structure('9XYZ')


def test_structure_spec_rejects_wrong_shapes():
    """Test that coordinates of another shape, and names of another length, raise errors."""
    with pytest.raises(ValidationError, match='models by atoms'):
        StructureSpec(np.zeros((2, 3)), (), (), (), ()).check()
    with pytest.raises(ValidationError, match='one entry'):
        StructureSpec(np.zeros((1, 2, 3)), ('N',), ('GLY',), (1,), ('A',)).check()


def test_read_trajectory_reads_a_dcd_file(tmp_path):
    """Test that a DCD file written by MDTraj reads back in angstrom as the models of a structure."""
    mdtraj = pytest.importorskip('mdtraj')
    structure = read_structure('1L2Y')
    topology = tmp_path / 'top.pdb'
    write_structure(dataclasses.replace(structure, coordinates=structure.coordinates[:1]), topology)
    trajectory = mdtraj.Trajectory(structure.coordinates[:3] / 10.0, mdtraj.load(str(topology)).topology)
    path = tmp_path / 'run.dcd'
    trajectory.save_dcd(str(path))
    frames = read_trajectory(path, topology)
    assert frames.coordinates.shape == (3, 304, 3)
    assert np.abs(frames.coordinates - structure.coordinates[:3]).max() < 0.01
    assert frames.atom_names[:2] == structure.atom_names[:2]


def test_backbone_round_trip_through_internal_coordinates():
    """Test that a chain rebuilt from its own angles matches the read one within 0.01 angstrom after superposition."""
    structure = read_structure('1UBQ')
    geometry = BackboneGeometry(structure).check().prepare()
    positions = geometry.run(geometry.angles_).positions_
    deviation = ShapeAnalysis('rmsd', reference=structure.coordinates[0]).check().prepare(positions[None]).run().values_
    assert deviation[0] < 0.01


def test_backbone_rejects_angles_of_another_length():
    """Test that angles of another length raise an error."""
    geometry = BackboneGeometry(read_structure('1L2Y')).check().prepare()
    with pytest.raises(ValidationError, match='38 angles'):
        geometry.run(np.zeros(4))


def test_backbone_rejects_two_chains():
    """Test that a structure of two chains raises an error."""
    structure = StructureSpec(np.zeros((1, 2, 3)), ('N', 'N'), ('GLY', 'GLY'), (1, 1), ('A', 'B')).check()
    with pytest.raises(ValidationError, match='one chain'):
        BackboneGeometry(structure).check()


def test_backbone_filters_only_the_warning_of_vectors():
    """Test that a rebuild shows no warning of Biopython's vectors, and that another warning stays visible."""
    geometry = BackboneGeometry(read_structure('1L2Y')).check().prepare()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        geometry.run(geometry.angles_)
        warnings.warn("'where' used elsewhere", UserWarning, stacklevel=1)
    assert [str(item.message) for item in caught] == ["'where' used elsewhere"]


def test_kyte_doolittle_scale_is_rescaled():
    """Test that the scale runs from 0 at arginine to 1 at isoleucine over twenty amino acids."""
    scale = read_kyte_doolittle_scale()
    assert len(scale) == 20
    assert min(scale.values()) == scale['ARG'] == 0.0
    assert max(scale.values()) == scale['ILE'] == 1.0


def test_relative_distance_of_the_ideal_micelle_and_of_uniform_hydrophobicity():
    """Test that `RD` is 0 for an observed distribution equal to the theoretical one, and 1 for a uniform one."""
    structure = read_structure('1L2Y')
    analysis = FuzzyOilDropAnalysis(structure).check().prepare().run()
    ideal = FuzzyOilDropAnalysis(structure, observed=analysis.theoretical_[0]).check()
    assert ideal.prepare(structure.effective_atoms[:1]).run().values_[0] == pytest.approx(0.0, abs=1e-12)
    uniform = FuzzyOilDropAnalysis(structure, observed=analysis.uniform_).check()
    assert uniform.prepare(structure.effective_atoms[:1]).run().values_[0] == pytest.approx(1.0, abs=1e-12)


def test_distributions_are_normalised():
    """Test that the profile sets the three distributions, each with a sum of one."""
    analysis = FuzzyOilDropAnalysis(read_structure('1L2Y'), 'profile').check().prepare().run()
    assert analysis.theoretical_.sum(axis=1) == pytest.approx(1.0)
    assert analysis.observed_.sum(axis=1) == pytest.approx(1.0)
    assert analysis.uniform_.sum() == pytest.approx(1.0)


def test_environment_factor_recovers_the_factor_of_a_profile():
    """Test that a distribution built by Eq. 1.9, with its complement normalised first, gives its factor back."""
    structure = read_structure('1L2Y')
    theoretical = FuzzyOilDropAnalysis(structure).check().prepare().run().theoretical_[0]
    complement = (theoretical.max() - theoretical) / (theoretical.max() - theoretical).sum()
    profile = theoretical + 0.5 * complement
    analysis = FuzzyOilDropAnalysis(structure, 'environment-factor', observed=profile / profile.sum())
    factor = analysis.check().prepare(structure.effective_atoms[:1]).run().values_[0]
    assert factor == pytest.approx(0.5, abs=1e-3)


def test_curves_of_fig_1_41_give_the_values_the_book_prints_for_them():
    """Test that the curves of Fig. 1.41 give its `K = 0.5` and, within reading error, the `RD = 0.7` of Fig. 1.40."""

    class FigureAnalysis(FuzzyOilDropAnalysis):
        def _read_theoretical(self, positions):
            return FIGURE_THEORETICAL / FIGURE_THEORETICAL.sum()

    count = len(FIGURE_THEORETICAL)
    numbers = tuple(range(count))
    structure = StructureSpec(np.zeros((1, count, 3)), ('CA',) * count, ('GLY',) * count, numbers, ('A',) * count)
    observed = FIGURE_OBSERVED / FIGURE_OBSERVED.sum()
    factor = FigureAnalysis(structure, 'environment-factor', observed=observed).check().prepare().run().values_[0]
    assert factor == pytest.approx(0.5, abs=0.02)
    distance = FigureAnalysis(structure, observed=observed).check().prepare().run().values_[0]
    assert distance == pytest.approx(0.7, abs=0.1)


def test_native_ubiquitin_is_closer_to_the_micelle_than_its_extended_chain():
    """Test that the native `RD` of ubiquitin is lower than that of its chain with all phi -120 and psi 130."""
    structure = read_structure('1UBQ')
    geometry = BackboneGeometry(structure).check().prepare()
    native = FuzzyOilDropAnalysis(structure).check().prepare().run().values_[0]
    angles = np.full(geometry.angles_.size, -120.0)
    angles[angles.size // 2 :] = 130.0
    positions = geometry.run(angles).read_effective_atoms()
    extended = FuzzyOilDropAnalysis(structure).check().prepare(positions[None]).run().values_[0]
    assert native < extended


def test_unknown_residue_raises_an_error_that_names_it():
    """Test that a residue outside the scale raises an error with its name, its chain and number, and the scale."""
    structure = read_structure('1L2Y')
    names = tuple(
        'SEP' if number == 5 else name
        for name, number in zip(structure.residue_names, structure.residue_numbers, strict=True)
    )
    changed = dataclasses.replace(structure, residue_names=names)
    with pytest.raises(AnalysisError, match=r'SEP at number 5 of chain A is not on the Kyte-Doolittle scale'):
        FuzzyOilDropAnalysis(changed).check().prepare()


def test_analysis_rejects_bad_settings_and_positions():
    """Test that an unknown kind, a cutoff of zero, and positions of another shape raise errors."""
    structure = read_structure('1L2Y')
    with pytest.raises(AnalysisError, match='waist'):
        FuzzyOilDropAnalysis(structure, 'waist').check()
    with pytest.raises(AnalysisError, match='cutoff'):
        FuzzyOilDropAnalysis(structure, cutoff=0.0).check()
    with pytest.raises(AnalysisError, match='residues'):
        FuzzyOilDropAnalysis(structure).check().prepare(np.zeros((1, 3, 3)))
