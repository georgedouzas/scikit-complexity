"""Test the defects of a field of axes or vectors."""

import numpy as np
import pytest

from skcomplex.analysis import DefectAnalysis
from skcomplex.errors import AnalysisError


def _defect_pair(size):
    """Return a periodic nematic field with a +1/2 defect at a plaquette and a -1/2 defect at another one."""
    rows, columns = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    images = range(-2, 3)
    angle = np.zeros((size, size))
    for shift_row in images:
        for shift_column in images:
            for centre, charge in ((size / 4 - 0.5, 0.5), (3 * size / 4 - 0.5, -0.5)):
                row = rows - (centre + shift_row * size)
                column = columns - (size / 2 - 0.5 + shift_column * size)
                angle += charge * np.arctan2(row, column)
    return angle


def test_run_finds_a_pair_of_nematic_defects():
    """A periodic nematic field holds one +1/2 and one -1/2 defect within one spacing of their sites."""
    size = 32
    defects = DefectAnalysis(period=np.pi).check().prepare(_defect_pair(size)).run()
    charges = defects.charges_[0]
    assert sorted(charges.tolist()) == [-0.5, 0.5]
    plus = defects.positions_[0][charges > 0][0]
    minus = defects.positions_[0][charges < 0][0]
    assert np.max(np.abs(plus - [size / 4 - 0.5, size / 2 - 0.5])) <= 1.0
    assert np.max(np.abs(minus - [3 * size / 4 - 0.5, size / 2 - 0.5])) <= 1.0


def test_run_finds_one_vortex_of_a_vector_field():
    """A vector field with one vortex holds one defect of charge 1."""
    rows, columns = np.meshgrid(np.arange(16.0), np.arange(16.0), indexing='ij')
    defects = DefectAnalysis(period=2 * np.pi, periodic=False).check().prepare(np.arctan2(rows - 7.5, columns - 7.5))
    assert defects.run().charges_[0].tolist() == [1.0]


def test_run_reads_every_frame():
    """An array of three dimensions holds one list of defects per frame."""
    frames = np.stack([np.zeros((6, 6)), _defect_pair(6)])
    defects = DefectAnalysis().check().prepare(frames).run()
    assert [len(item) for item in defects.charges_] == [0, 2]


def test_check_rejects_a_period_that_is_not_positive():
    """A period of 0 raises an error."""
    with pytest.raises(AnalysisError, match='period'):
        DefectAnalysis(period=0.0).check()


def test_prepare_rejects_a_field_of_another_rank():
    """A field of one dimension raises an error."""
    with pytest.raises(AnalysisError, match='two-dimensional'):
        DefectAnalysis().check().prepare(np.zeros(4))
