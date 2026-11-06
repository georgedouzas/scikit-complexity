"""Test the `_metabolic_perturbations` module of `skcomplex.systems.biology.systems_biology`."""

import pytest

from skcomplex.systems import create_simulation
from skcomplex.systems.biology.common.metabolism import CobraAnalysis
from skcomplex.systems.biology.systems_biology import metabolic_perturbations


def test_essential_gene_stops_growth():
    """Test that knocking out `gapA` leaves no growth."""
    trajectory = create_simulation('metabolic_perturbations', genes=('b1779',)).prepare().run().trajectory_
    assert float(trajectory.values('cell.objective_0')[0]) == pytest.approx(0.0, abs=1e-6)


def test_single_deletion_scan_finds_seven_essential_genes():
    """Test that deleting every gene in turn leaves exactly 7 with growth below 1e-6."""
    growth = CobraAnalysis('deletions').prepare(metabolic_perturbations()).run().growth_
    assert int((growth < 1e-6).sum()) == 7
