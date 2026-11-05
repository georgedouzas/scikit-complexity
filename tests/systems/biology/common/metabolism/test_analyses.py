"""Test the `_analyses` module of `skcomplex.systems.biology.common.metabolism`."""

import sys

import numpy as np
import pytest

from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import MissingExtraError
from skcomplex.core.specs import RunStateSpec
from skcomplex.systems.biology.common.metabolism import (
    CobraAnalysis,
    FluxVariabilityAnalysis,
    ProductionScanAnalysis,
    ReactionSpec,
    StrainDesignAnalysis,
    load_network,
    metabolic_network_type,
    to_cobra,
)

GROWTH = 'Biomass_Ecoli_core'
AEROBIC = 0.8739215


def _core_system(**values):
    """Return the system of the E. coli core system."""
    return System(elements=[load_network('textbook', 'cell').set_params(**values)], time=StaticTime())


def _objective(system):
    """Return the objective of the only program of a static system."""
    trajectory = Simulation(system).prepare().run(0.0).trajectory_
    return float(trajectory.values('cell.objective_0')[0])


def test_cobra_variability_fixes_growth_at_the_optimum():
    """Test that flux variability at the optimum fixes the biomass flux at both ends."""
    ranges = CobraAnalysis('variability').prepare(_core_system()).run().ranges_
    assert ranges.loc[GROWTH].tolist() == pytest.approx([AEROBIC, AEROBIC], abs=1e-4)


def test_cobra_parity_with_the_core_solve():
    """Test that COBRApy and the core agree on growth and on the parsimonious total flux."""
    system = _core_system(parsimonious=True)
    problem = compile_system(system)
    solved = problem.programs[0].run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)).values_
    reconstruction = to_cobra(system.elements[0])
    assert problem.states.read(solved, 'cell.objective_0') == pytest.approx(reconstruction.slim_optimize(), abs=1e-7)
    total = CobraAnalysis('parsimonious').prepare(system).run().total_
    assert problem.states.read(solved, 'cell.objective_1') == pytest.approx(total, abs=1e-6)


def test_cobra_deletions_find_the_essential_genes():
    """Test that deleting every gene in turn leaves 7 with no growth."""
    growth = CobraAnalysis('deletions').prepare(_core_system()).run().growth_
    assert int((growth < 1e-6).sum()) == 7
    assert growth['b1779'] < 1e-6


@pytest.mark.parametrize(
    ('kind', 'linear'),
    [('moma', True), ('moma', False), ('room', True), pytest.param('room', False, marks=pytest.mark.slow)],
)
def test_cobra_adjustment_grows_below_the_optimum_of_the_knock_out(kind, linear):
    """Test that MoMA and ROOM after `pgi` give positive growth below the FBA of the knock-out."""
    knocked = _core_system(g_b4025=0.0)
    optimum = _objective(knocked)
    assert optimum == pytest.approx(0.8632, abs=1e-3)
    analysis = CobraAnalysis(kind, knockouts=('b4025',), linear=linear).prepare(_core_system()).run()
    assert 0 < analysis.growth_ < optimum


def test_cobra_finds_synthetic_lethal_reactions():
    """Test that double deletions of reactions find lethal pairs, neither lethal alone."""
    pairs = CobraAnalysis('synthetic-lethals', target='reaction').prepare(_core_system()).run().pairs_
    assert pairs
    assert ('ENO', 'PGK') not in pairs


def test_strain_design_couples_succinate_to_growth():
    """Test that OptKnock finds a design that secretes at least 9 succinate at maximal growth, unlike the wild type."""
    analysis = StrainDesignAnalysis(
        'optknock',
        product='EX_succ_e',
        constraints=f'{GROWTH} >= 0.1',
        interventions=3,
    )
    designs = analysis.prepare(_core_system()).run().designs_
    assert designs
    reconstruction = to_cobra(_core_system().elements[0])
    cobra = pytest.importorskip('cobra')
    wild = cobra.flux_analysis.flux_variability_analysis(
        reconstruction,
        ['EX_succ_e'],
        fraction_of_optimum=1.0,
        processes=1,
    )
    assert wild.loc['EX_succ_e', 'minimum'] == pytest.approx(0.0, abs=1e-6)
    with reconstruction:
        for name in designs[0]:
            reconstruction.reactions.get_by_id(name).knock_out()
        secreted = cobra.flux_analysis.flux_variability_analysis(
            reconstruction,
            ['EX_succ_e'],
            fraction_of_optimum=1.0,
            processes=1,
        )
    assert secreted.loc['EX_succ_e', 'minimum'] >= 9


def test_production_scan_reports_rising_fluxes_and_falling_growth():
    """Test that scanning the succinate exchange in 10 steps reports rising fluxes while growth falls."""
    analysis = ProductionScanAnalysis('EX_succ_e', steps=10).prepare(_core_system()).run()
    assert analysis.reactions_
    amounts = analysis.fluxes_.loc[analysis.reactions_].abs().to_numpy()
    assert np.all(np.diff(amounts, axis=1) > 0)
    assert analysis.objectives_[0] == pytest.approx(AEROBIC, abs=1e-6)
    assert np.all(np.diff(analysis.objectives_) < 0)


def test_flux_variability_reports_ranges_without_the_extra():
    """Test the ranges of a small network with two parallel routes."""
    reactions = {
        'in': ReactionSpec({'A': 1.0}, upper=4.0),
        'one': ReactionSpec({'A': -1.0}),
        'two': ReactionSpec({'A': -1.0}),
    }
    system = System(elements=[metabolic_network_type('Parallel', reactions, {'in': 1.0})('net')], time=StaticTime())
    ranges = FluxVariabilityAnalysis().prepare(system).run().ranges_
    assert ranges.loc['one'].tolist() == [0.0, 4.0]
    assert ranges.loc['in'].tolist() == [4.0, 4.0]


def test_missing_extra_names_the_class_and_the_command(monkeypatch):
    """Test that an analysis that needs the hidden extra raises an error that names the command."""
    monkeypatch.setitem(sys.modules, 'cobra', None)
    with pytest.raises(
        MissingExtraError,
        match=r"CobraAnalysis.*metabolism.*pip install 'scikit-complexity\[metabolism\]'",
    ):
        CobraAnalysis('variability').prepare()


@pytest.mark.parametrize(
    ('kind', 'settings'),
    [
        ('robustknock', {'product': 'EX_succ_e', 'constraints': f'{GROWTH} >= 0.1'}),
        ('optcouple', {'product': 'EX_succ_e', 'constraints': f'{GROWTH} >= 0.1'}),
        ('mcs', {'constraints': f'{GROWTH} >= 0.1'}),
    ],
)
def test_strain_design_computes_every_kind(kind, settings):
    """Test that the other kinds of strain design return their designs as sets of reaction names."""
    analysis = StrainDesignAnalysis(kind, interventions=2, **settings).prepare(_core_system()).run()
    assert all(isinstance(design, set) for design in analysis.designs_)
    names = set(_core_system().elements[0].reactions)
    assert all(design <= names for design in analysis.designs_)


def test_flux_variability_keeps_an_objective_of_several_reactions():
    """Test that an objective of two reactions is kept as a whole, as Raman Eq. 8.11d gives."""
    reactions = {
        'in': ReactionSpec({'A': 1.0}, upper=4.0),
        'one': ReactionSpec({'A': -1.0}),
        'two': ReactionSpec({'A': -1.0}),
    }
    element_type = metabolic_network_type('TwoGoalsElement', reactions, {'one': 1.0, 'two': 1.0})
    system = System(elements=[element_type('net')], time=StaticTime())
    analysis = FluxVariabilityAnalysis().prepare(system).run()
    assert analysis.optimum_ == pytest.approx(4.0)
    assert analysis.ranges_.loc['in'].tolist() == pytest.approx([4.0, 4.0])
    assert analysis.ranges_.loc['one'].tolist() == pytest.approx([0.0, 4.0])
    assert analysis.ranges_.loc['two'].tolist() == pytest.approx([0.0, 4.0])
