"""Test the `_analyses` module of `skcomplex.systems.biology.common.metabolism`."""

import sys

import numpy as np
import pytest

from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import MissingExtraError
from skcomplex.core.specs import RunStateSpec
from skcomplex.systems.biology.common.metabolism import (
    ReactionSpec,
    cobra_measures,
    flux_variability,
    load_network,
    metabolic_network_type,
    production_scan,
    strain_design,
    to_cobra,
)

GROWTH = 'Biomass_Ecoli_core'
AEROBIC = 0.8739215


def _core_system(**values):
    """Return the system of the E. coli core system."""
    return System(elements=[load_network('textbook', 'cell').set_params(**values)], time=StaticTime())


def _objective(system):
    """Return the objective of the only program of a static system."""
    trajectory = simulate(system, until=0.0)
    return float(trajectory.values('cell.objective_0')[0])


def test_cobra_variability_fixes_growth_at_the_optimum():
    """Test that flux variability at the optimum fixes the biomass flux at both ends."""
    ranges = cobra_measures(_core_system(), kind='variability').ranges
    assert ranges.loc[GROWTH].tolist() == pytest.approx([AEROBIC, AEROBIC], abs=1e-4)


def test_cobra_parity_with_the_core_solve():
    """Test that COBRApy and the core agree on growth and on the parsimonious total flux."""
    system = _core_system(parsimonious=True)
    problem = PassCompiler().prepare(system).run().problem_
    solved = problem.programs[0].run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)).values_
    reconstruction = to_cobra(system.elements[0])
    assert problem.states.read(solved, 'cell.objective_0') == pytest.approx(reconstruction.slim_optimize(), abs=1e-7)
    total = cobra_measures(system, kind='parsimonious').total
    assert problem.states.read(solved, 'cell.objective_1') == pytest.approx(total, abs=1e-6)


def test_cobra_deletions_find_the_essential_genes():
    """Test that deleting every gene in turn leaves 7 with no growth."""
    growth = cobra_measures(_core_system(), kind='deletions').growth
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
    analysis = cobra_measures(_core_system(), kind=kind, knockouts=('b4025',), linear=linear)
    assert 0 < analysis.growth < optimum


def test_cobra_finds_synthetic_lethal_reactions():
    """Test that double deletions of reactions find lethal pairs, neither lethal alone."""
    pairs = cobra_measures(_core_system(), kind='synthetic-lethals', target='reaction').pairs
    assert pairs
    assert ('ENO', 'PGK') not in pairs


def test_strain_design_couples_succinate_to_growth():
    """Test that OptKnock finds a design that secretes at least 9 succinate at maximal growth, unlike the wild type."""
    designs = strain_design(
        _core_system(),
        kind='optknock',
        product='EX_succ_e',
        constraints=f'{GROWTH} >= 0.1',
        interventions=3,
    ).designs
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
    analysis = production_scan(_core_system(), target='EX_succ_e', steps=10)
    assert analysis.reactions
    amounts = analysis.fluxes.loc[analysis.reactions].abs().to_numpy()
    assert np.all(np.diff(amounts, axis=1) > 0)
    assert analysis.objectives[0] == pytest.approx(AEROBIC, abs=1e-6)
    assert np.all(np.diff(analysis.objectives) < 0)


def test_flux_variability_reports_ranges_without_the_extra():
    """Test the ranges of a small network with two parallel routes."""
    reactions = {
        'in': ReactionSpec({'A': 1.0}, upper=4.0),
        'one': ReactionSpec({'A': -1.0}),
        'two': ReactionSpec({'A': -1.0}),
    }
    system = System(elements=[metabolic_network_type('Parallel', reactions, {'in': 1.0})('net')], time=StaticTime())
    ranges = flux_variability(system).ranges
    assert ranges.loc['one'].tolist() == [0.0, 4.0]
    assert ranges.loc['in'].tolist() == [4.0, 4.0]


def test_missing_extra_names_the_function_and_the_command(monkeypatch):
    """Test that a function that needs the hidden extra raises an error that names the command."""
    monkeypatch.setitem(sys.modules, 'cobra', None)
    with pytest.raises(
        MissingExtraError,
        match=r"cobra_measures.*metabolism.*pip install 'scikit-complexity\[metabolism\]'",
    ):
        cobra_measures(None, kind='variability')


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
    analysis = strain_design(_core_system(), kind=kind, interventions=2, **settings)
    assert all(isinstance(design, set) for design in analysis.designs)
    names = set(_core_system().elements[0].reactions)
    assert all(design <= names for design in analysis.designs)


def test_flux_variability_keeps_an_objective_of_several_reactions():
    """Test that an objective of two reactions is kept as a whole, as Raman Eq. 8.11d gives."""
    reactions = {
        'in': ReactionSpec({'A': 1.0}, upper=4.0),
        'one': ReactionSpec({'A': -1.0}),
        'two': ReactionSpec({'A': -1.0}),
    }
    element_type = metabolic_network_type('TwoGoalsElement', reactions, {'one': 1.0, 'two': 1.0})
    system = System(elements=[element_type('net')], time=StaticTime())
    analysis = flux_variability(system)
    assert analysis.optimum == pytest.approx(4.0)
    assert analysis.ranges.loc['in'].tolist() == pytest.approx([4.0, 4.0])
    assert analysis.ranges.loc['one'].tolist() == pytest.approx([0.0, 4.0])
    assert analysis.ranges.loc['two'].tolist() == pytest.approx([0.0, 4.0])
