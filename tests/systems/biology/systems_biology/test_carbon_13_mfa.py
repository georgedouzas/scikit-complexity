"""Test the `_carbon_13_mfa` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np
import pytest

from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.system import System
from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.common.metabolism import ReactionSpec, metabolic_network_type
from skcomplex.systems.biology.systems_biology import (
    AtomMapSpec,
    Carbon13MFASystem,
    FragmentSpec,
    LabelMeasurementSpec,
    TracerSpec,
    labelling,
    tca_labelling_analysis,
)

TRUE = {'v1': 100.0, 'v2': 100.0, 'v3': 50.0, 'v4': 50.0, 'v5': 50.0, 'v6': 50.0, 'v8': 50.0, 'v9': 50.0}


def test_estimation_recovers_the_split_of_alpha_ketoglutarate():
    """Test the estimated split, that every interval holds its true value, and that the test accepts the fit."""
    analysis = tca_labelling_analysis(noise=0.002, random_state=0)(Carbon13MFASystem().prepare().system_)
    assert analysis.fluxes['v3'] == pytest.approx(50, abs=2)
    assert analysis.fluxes['v4'] == pytest.approx(50, abs=2)
    assert analysis.accepted
    for name, value in TRUE.items():
        low, high = analysis.intervals[name]
        assert low - 1e-6 <= value <= high + 1e-6, name


def test_simulation_gives_the_published_pattern():
    """Test the glutamate of the true fluxes and the tracer of Antoniewicz et al. (2007)."""
    analysis = tca_labelling_analysis('simulation')(Carbon13MFASystem().prepare().system_)
    found = analysis.distributions['published']['Glu']
    np.testing.assert_allclose(found, [0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039], atol=5e-5)


def test_origins_of_a_valid_map():
    """Test that the map records where atoms of citrate come from."""
    atoms = AtomMapSpec('v1', [('AcCoA', 'ab'), ('OAC', 'cdef')], [('Cit', 'fedbac')])
    assert atoms.origins[('Cit', 1)] == ('OAC', 4)
    assert atoms.origins[('Cit', 5)] == ('AcCoA', 1)


def test_check_names_an_unbalanced_letter():
    """Test that a product letter no reactant carries raises an error that names the reaction and the letter."""
    with pytest.raises(ValidationError, match=r"reaction v1.*'z'"):
        AtomMapSpec('v1', [('A', 'ab')], [('B', 'az')])._check()


BOUND = 300.0
PUBLISHED = [0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039]
ATOMS = (
    AtomMapSpec('v1', [('AcCoA', 'ab'), ('OAC', 'cdef')], [('Cit', 'fedbac')]),
    AtomMapSpec('v2', [('Cit', 'abcdef')], [('AKG', 'abcde')]),
    AtomMapSpec('v3', [('AKG', 'abcde')], [('Glu', 'abcde')]),
    AtomMapSpec('v4', [('AKG', 'abcde')], [('Suc', 'bcde')]),
    AtomMapSpec('v5', [('Suc', 'abcd')], [('Fum', 'abcd')]),
    AtomMapSpec('v6', [('Fum', 'abcd')], [('OAC', 'abcd')]),
    AtomMapSpec('v8', [('Asp', 'abcd')], [('OAC', 'abcd')]),
)
REACTIONS = {
    'v1': ReactionSpec({'OAC': -1.0, 'Cit': 1.0}, upper=BOUND),
    'v2': ReactionSpec({'Cit': -1.0, 'AKG': 1.0}, upper=BOUND),
    'v3': ReactionSpec({'AKG': -1.0, 'Glu': 1.0}, upper=BOUND),
    'v4': ReactionSpec({'AKG': -1.0, 'Suc': 1.0}, upper=BOUND),
    'v5': ReactionSpec({'Suc': -1.0, 'Fum': 1.0}, upper=BOUND),
    'v6': ReactionSpec({'Fum': -1.0, 'OAC': 1.0}, lower=-BOUND, upper=BOUND),
    'v8': ReactionSpec({'OAC': 1.0}, upper=BOUND),
    'v9': ReactionSpec({'Glu': -1.0}, upper=BOUND),
}
TRUE_FLUXES = {
    'v1': 100.0,
    'v2': 100.0,
    'v3': 50.0,
    'v4': 50.0,
    'v5': 50.0,
    'v6': (125.0, 75.0),
    'v8': 50.0,
    'v9': 50.0,
}
GLUTAMATE = FragmentSpec('Glu', (1, 2, 3, 4, 5))
ASPARTATE = TracerSpec('Asp', {'0000': 1.0})
PUBLISHED_TRACER = (TracerSpec('AcCoA', {'00': 0.5, '01': 0.25, '11': 0.25}), ASPARTATE)
FIRST_TRACER = (TracerSpec('AcCoA', {'00': 0.5, '10': 0.5}), ASPARTATE)
SECOND_TRACER = (TracerSpec('AcCoA', {'00': 0.5, '11': 0.5}), ASPARTATE)


def _system():
    """Return the system of the TCA example of Antoniewicz et al. (2007)."""
    return System(elements=[metabolic_network_type('TcaElement', REACTIONS)('net')], time=StaticTime())


def _simulated(tracers):
    """Return the distributions of glutamate of the true fluxes under some tracers."""
    return labelling(
        _system(),
        atoms=ATOMS,
        tracers=tracers,
        fragments=[GLUTAMATE],
        fluxes=TRUE_FLUXES,
        symmetric=('Suc', 'Fum'),
    ).distributions


def _measurements(noise=0.0, seed=0):
    """Return the measurements of glutamate in two parallel experiments."""
    tracers = {'first': FIRST_TRACER, 'second': SECOND_TRACER}
    rng = np.random.default_rng(seed)
    found = []
    for experiment, values in _simulated(tracers).items():
        drawn = values['Glu'] + (rng.normal(0.0, noise, 6) if noise else 0.0)
        found.append(LabelMeasurementSpec(experiment, 'Glu', drawn, 0.002))
    return tracers, found


def test_check_names_an_inconsistent_atom_count():
    """Test that two reactions with different atom counts of glutamate raise an error that names all three."""
    atoms = [
        AtomMapSpec('v1', [('A', 'abcde')], [('Glu', 'abcde')]),
        AtomMapSpec('v2', [('B', 'abcd')], [('Glu', 'abcd')]),
    ]
    with pytest.raises(ValidationError, match=r'v1 gives Glu 5 atoms.*v2'):
        labelling(_system(), atoms=atoms)


def test_simulation_gives_the_published_distribution():
    """Test the distribution of glutamate of the TCA example against the published one."""
    found = _simulated({'published': PUBLISHED_TRACER})['published']['Glu']
    np.testing.assert_allclose(found, PUBLISHED, atol=5e-5)
    assert np.all(found >= 0)
    assert found.sum() == pytest.approx(1)


def test_simulation_matches_mfapy(tmp_path):
    """Test the distribution of glutamate against mfapy within 1e-10."""
    mfapy = pytest.importorskip('mfapy')
    system = tmp_path / 'system.txt'
    system.write_text(
        '//Reactions\n'
        'v1\tAcCoA + OAC --> Cit\tAcCoA + OAC --> Cit\tAB + CDEF --> FEDBAC\t(kegg:R00351)\t0.0\t300\n'
        'v2\tCit --> AKG + CO2ex\tCit --> AKG + CO2ex\tABCDEF --> ABCDE + F\t(kegg:R00709)\t0.0\t300\n'
        'v3\tAKG --> Glu\tAKG --> Glu\tABCDE --> ABCDE \t(kegg:R00243)\t0.0\t300\n'
        'v4\tAKG --> Suc + CO2ex\tAKG --> Suc + CO2ex\tABCDE --> BCDE + A\t(kegg:R01197)\t0.0\t300\n'
        'v5\tSuc --> Fum\tSuc --> Fum\tABCD --> ABCD\t(kegg:R02164)\t0.0\t300\n'
        'v6\tFum --> OAC\tFum --> OAC\tABCD --> ABCD\t(kegg:R01082)\t0.0\t300\n'
        'v7\tOAC --> Fum\tOAC --> Fum\tABCD --> ABCD\t(kegg:R01082)\t0.0\t300\n'
        'v8\tAsp --> OAC\tAsp --> OAC\tABCD --> ABCD\t(kegg:R00355)\t0.0\t300\n'
        'v9\tGlu --> Gluex\tnd\tnd\t(kegg:R00243)\t0.0\t300\n'
        '//Metabolites\n'
        'CO2ex\t1\tno\tno\texcreted\t(kegg:C00011)\t0\t300\n'
        'AcCoA\t2\tno\tcarbonsource\tno\t(kegg:C00024)\t0\t300\n'
        'OAC\t4\tno\tno\tno\t(kegg:C00036)\t0\t300\n'
        'Cit\t6\tno\tno\tno\t(kegg:C00158)\t0\t300\n'
        'AKG\t5\tno\tno\tno\t(kegg:C00026)\t0\t300\n'
        'Suc\t4\tsymmetry\tno\tno\t(kegg:C00042)\t0\t300\n'
        'Fum\t4\tsymmetry\tno\tno\t(kegg:C00122)\t0\t300\n'
        'Glu\t5\tno\tno\tno\t(kegg:C00025)\t0\t300\n'
        'Gluex\t5\tno\tno\texcreted\t(kegg:C00025)\t0\t300\n'
        'Asp\t4\tno\tcarbonsource\tno\t(kegg:C00049)\t0\t300\n'
        '//Reversible_reactions\n'
        'FUM\tv6\tv7\t(kegg:R01082)\t-300\t300\n'
        '//Target_fragments\n'
        'Glue\tgcms\tGlu_1:2:3:4:5\tuse\tC5H10N2O3\n'
        '//End\n',
    )
    reactions, reversible, metabolites, fragments = mfapy.mfapyio.load_metabolic_model(str(system), format='text')
    metabolic = mfapy.metabolicmodel.MetabolicModel(reactions, reversible, metabolites, fragments)
    metabolic.set_configuration(callbacklevel=0)
    metabolic.set_constraint('reaction', 'v1', 'fixed', 100.0, 1)
    metabolic.update()
    status = tmp_path / 'status.csv'
    status.write_text(
        'State,Id,type,value,stdev,lb,ub\n'
        + ''.join(
            f'reaction,{name},free,{value},1,0,300\n'
            for name, value in (
                ('v1', 100),
                ('v2', 100),
                ('v3', 50),
                ('v4', 50),
                ('v5', 50),
                ('v6', 125),
                ('v7', 75),
                ('v8', 50),
                ('v9', 50),
            )
        )
        + 'reversible,FUM,free,50,1,0,300\n',
    )
    flux = metabolic.load_states(str(status), format='csv')
    source = metabolic.generate_carbon_source_templete()
    source.set_all_isotopomers('AcCoA', [0.5, 0.0, 0.25, 0.25])
    source.set_each_isotopomer('Asp', {'#0000': 1.0}, correction='no')
    reference = metabolic.generate_mdv(flux, source).get_fragment_mdv('Glue')
    found = _simulated({'published': PUBLISHED_TRACER})['published']['Glu']
    np.testing.assert_allclose(found, reference, atol=1e-10)


def test_simulation_of_an_unlabelled_tracer_is_unlabelled():
    """Test that unlabelled tracers give a distribution with all its mass at zero labels."""
    tracers = {'none': (TracerSpec('AcCoA', {'00': 1.0}), ASPARTATE)}
    np.testing.assert_allclose(_simulated(tracers)['none']['Glu'], [1, 0, 0, 0, 0, 0], atol=1e-12)


def test_simulation_scrambles_a_symmetric_metabolite():
    """Test that a label at atom 1 of a symmetric succinate reaches atoms 1 and 4 of fumarate by halves."""
    reactions = {
        'in': ReactionSpec({'Suc': 1.0}),
        'on': ReactionSpec({'Suc': -1.0, 'Fum': 1.0}),
        'out': ReactionSpec({'Fum': -1.0}),
    }
    atoms = [
        AtomMapSpec('in', [('Sub', 'abcd')], [('Suc', 'abcd')]),
        AtomMapSpec('on', [('Suc', 'abcd')], [('Fum', 'abcd')]),
    ]
    fragments = [FragmentSpec('Fum', (atom,), f'atom{atom}') for atom in (1, 2, 3, 4)]
    system = System(elements=[metabolic_network_type('SymmetricElement', reactions)('net')], time=StaticTime())
    found = labelling(
        system,
        atoms=atoms,
        tracers={'first': [TracerSpec('Sub', {'1000': 1.0})]},
        fragments=fragments,
        fluxes={'in': 1.0, 'on': 1.0, 'out': 1.0},
        symmetric=('Suc',),
    ).distributions['first']
    assert found['atom1'][1] == pytest.approx(0.5)
    assert found['atom4'][1] == pytest.approx(0.5)
    assert found['atom2'][1] == pytest.approx(0.0)


def _estimated(measurements, tracers, *, measured=None, intervals='linear', random_state=0):
    """Return the estimation of the TCA example."""
    return labelling(
        _system(),
        kind='estimation',
        atoms=ATOMS,
        tracers=tracers,
        fragments=[GLUTAMATE],
        measurements=measurements,
        measured_fluxes=measured or {'v1': (100.0, 0.0)},
        symmetric=('Suc', 'Fum'),
        intervals=intervals,
        random_state=random_state,
    )


def test_estimation_recovers_the_fluxes_from_noise_free_data():
    """Test that every net flux is recovered within 1e-4 relative error, with a residual below 1e-10."""
    tracers, measurements = _measurements()
    analysis = _estimated(measurements, tracers)
    assert analysis.residual < 1e-10
    for name in ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v8', 'v9'):
        expected = 50.0 if name == 'v6' else TRUE_FLUXES[name]
        assert analysis.fluxes[name] == pytest.approx(expected, rel=1e-4)


def test_estimation_repeats_with_one_seed():
    """Test that two runs with one seed give the same estimates and the same residual."""
    tracers, measurements = _measurements(noise=0.002)
    first = _estimated(measurements, tracers)
    second = _estimated(measurements, tracers)
    assert first.fluxes == second.fluxes
    assert first.residual == second.residual


def test_estimation_rejects_data_that_contradict_the_balances():
    """Test that a measured flux against the balances puts the residual above the acceptance range."""
    tracers, measurements = _measurements(noise=0.002)
    analysis = _estimated(measurements, tracers, measured={'v1': (100.0, 0.0), 'v2': (150.0, 1.0)})
    assert analysis.residual > analysis.chi2[1]
    assert not analysis.accepted


@pytest.fixture(scope='module')
def profiled():
    """Return the estimation of noisy data with profile intervals."""
    tracers, measurements = _measurements(noise=0.002)
    return _estimated(measurements, tracers, intervals='profile')


def test_estimation_of_noisy_data_has_intervals_that_hold_the_truth(profiled):
    """Test that every profile interval of a net flux holds its true value, and that the test accepts the fit."""
    assert profiled.accepted
    for name in ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v8', 'v9'):
        low, high = profiled.intervals[name]
        expected = 50.0 if name == 'v6' else TRUE_FLUXES[name]
        assert low - 1e-3 <= expected <= high + 1e-3


def test_estimation_marks_an_unidentifiable_exchange(profiled):
    """Test that the fumarase exchange, which the data do not fix, reaches a bound and is not identifiable."""
    low, high = profiled.intervals['v6.exchange']
    assert low == pytest.approx(0.0, abs=1e-3) or high == pytest.approx(BOUND)
    assert not profiled.identifiable['v6.exchange']
