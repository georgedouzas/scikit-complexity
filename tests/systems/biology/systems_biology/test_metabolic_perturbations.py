"""Test the `_metabolic_perturbations` module of `skcomplex.systems.biology.systems_biology`."""

import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.common.metabolism import cobra_measures
from skcomplex.systems.biology.systems_biology import MetabolicPerturbationsSystem


def test_essential_gene_stops_growth():
    """Test that knocking out `gapA` leaves no growth."""
    trajectory = simulate(MetabolicPerturbationsSystem(genes=('b1779',)))
    assert float(trajectory.get_values('cell.objective_0')[0]) == pytest.approx(0.0, abs=1e-6)


def test_single_deletion_scan_finds_seven_essential_genes():
    """Test that deleting every gene in turn leaves exactly 7 with growth below 1e-6."""
    growth = cobra_measures(MetabolicPerturbationsSystem().prepare().system_, kind='deletions').growth
    assert int((growth < 1e-6).sum()) == 7
