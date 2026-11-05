"""Test the translation of Vensim and XMILE files into model modules against the runs of PySD."""

import importlib
import shutil

import numpy as np
import pandas as pd
import pysd
import pytest

from skcomplex.analysis import LoopAnalysis
from skcomplex.core.interactions import RelaxationInteraction, TransferInteraction
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerSolver
from skcomplex.engine.translators import SystemDynamicsTranslator
from skcomplex.errors import ConservationError, IncompatibleLawsError, ValidationError
from skcomplex.specs import ObservableSpec, RunSettingsSpec

REFERENCES = {
    'teacup.mdl': 'teacup_mdl.csv',
    'teacup.xmile': 'teacup_xmile.csv',
    'sir.mdl': 'sir_mdl.csv',
    'sir.xmile': 'sir_xmile.csv',
    'goal_seeking.mdl': 'goal_seeking_mdl.csv',
    'population.mdl': 'population_mdl.csv',
    'third_order_delay.mdl': 'third_order_delay_mdl.csv',
    'regions.mdl': 'regions_mdl.csv',
    'signals.mdl': 'signals_mdl.csv',
    'xmile_signals.xmile': 'xmile_signals_xmile.csv',
}
TOLERANCE = 1e-6
CONTROL = """
INITIAL TIME = 0 ~ ~ |
FINAL TIME = 4 ~ ~ |
TIME STEP = 0.25 ~ ~ |
SAVEPER = 1 ~ ~ |
"""


def _load_reference(fixtures_folder, file_name):
    """Return the stored run of PySD of a fixture, indexed by time."""
    return pd.read_csv(fixtures_folder / 'references' / REFERENCES[file_name], index_col=0)


def _simulate(module):
    """Return the trajectory of a translated module with the integration and the time step of its file."""
    system = module.create_system()
    simulation = Simulation(
        system,
        solver=EulerSolver(dt=module.DT),
        dt=module.DT,
        observables=module.observables(system),
    )
    return simulation.check().prepare().run(until=module.UNTIL).trajectory_


def _at_saved_times(trajectory, values, module, reference):
    """Return values of a trajectory at the times of a stored run of PySD."""
    times = np.asarray(trajectory.times) + module.START
    return np.asarray(values)[np.searchsorted(times, reference.index.to_numpy())]


def _translate_text(tmp_path, text, name='model.mdl'):
    """Return the translator of a model file with the given text and a control section."""
    path = tmp_path / name
    path.write_text(text + CONTROL)
    return SystemDynamicsTranslator(path).check().prepare()


def _total(s):
    """Return the people of the three stocks of the SIR model."""
    return s.susceptible + s.infected + s.recovered


def _growing_total(s):
    """Return a total that the births of a model break."""
    return s.population


@pytest.mark.parametrize('file_name', REFERENCES)
def test_references_load(fixtures_folder, file_name):
    """Test that the stored run of every runnable fixture loads with times and values."""
    reference = _load_reference(fixtures_folder, file_name)
    assert len(reference) > 1
    assert not reference.isna().to_numpy().any()


@pytest.mark.parametrize('file_name', ['teacup.mdl', 'sir.xmile', 'third_order_delay.mdl', 'regions.mdl'])
def test_references_match_pysd(fixtures_folder, tmp_path, file_name):
    """Test that the stored runs are the runs of the installed PySD."""
    path = shutil.copy(fixtures_folder / file_name, tmp_path / file_name)
    model = pysd.read_vensim(str(path)) if file_name.endswith('.mdl') else pysd.read_xmile(str(path))
    run = model.run()
    reference = _load_reference(fixtures_folder, file_name)
    for column in reference.columns:
        assert np.max(np.abs(run[column].to_numpy() - reference[column].to_numpy())) < TOLERANCE


@pytest.mark.parametrize('file_name', ['teacup.mdl', 'teacup.xmile'])
def test_run_teacup_runs_like_pysd(fixtures_folder, translate, file_name):
    """Test that the temperature at every saved time is within the tolerance of PySD's run with the same time step."""
    module = translate(file_name)
    reference = _load_reference(fixtures_folder, file_name)
    trajectory = _simulate(module)
    values = _at_saved_times(trajectory, trajectory.values('teacup.teacup_temperature'), module, reference)
    assert np.max(np.abs(values - reference['Teacup Temperature'].to_numpy())) < TOLERANCE


def test_run_keeps_settings_of_the_file(fixtures_folder):
    """Test that the settings hold the control section, whose length runs from the initial time."""
    translator = SystemDynamicsTranslator(fixtures_folder / 'signals.mdl').check().prepare().run()
    assert translator.settings_ == RunSettingsSpec(start=10.0, until=20.0, dt=0.125, save_every=0.25)
    assert 'START = 10.0' in translator.source_
    assert 'UNTIL = 20.0' in translator.source_
    assert 'SAVE_EVERY = 0.25' in translator.source_


def test_run_reads_saving_interval_from_the_time_step(fixtures_folder):
    """Test that a saving interval written as the time step takes its value."""
    translator = SystemDynamicsTranslator(fixtures_folder / 'teacup.mdl').check().prepare().run()
    assert translator.settings_.save_every == translator.settings_.dt


def test_run_default_solver_finds_the_exact_solution(translate):
    """Test that the adaptive default solver follows the exact teacup and not the Euler error of the file's step."""
    module = translate('teacup.mdl')
    trajectory = Simulation(module.create_system(), dt=module.DT).check().prepare().run(until=module.UNTIL).trajectory_
    exact = 70.0 + 110.0 * np.exp(-np.asarray(trajectory.times) / 10.0)
    assert np.max(np.abs(trajectory.values('teacup.teacup_temperature') - exact)) < 1e-3


def test_run_xmile_reads_like_vensim(fixtures_folder, translate):
    """Test that the same model from a Vensim and an XMILE file has the same states, parameters, and interactions."""
    vensim = SystemDynamicsTranslator(fixtures_folder / 'sir.mdl').check().prepare().run().source_
    xmile = SystemDynamicsTranslator(fixtures_folder / 'sir.xmile').check().prepare().run().source_
    assert vensim == xmile
    first, second = translate('sir.mdl').create_system(), translate('sir.xmile').create_system()
    assert first.elements[0].get_params().keys() == second.elements[0].get_params().keys()
    assert first.elements[0].get_params() == second.elements[0].get_params()
    assert [type(item) for item in first.interactions] == [type(item) for item in second.interactions]
    assert [item.label for item in first.interactions] == [item.label for item in second.interactions]


def test_run_module_has_one_shape(translate):
    """Test that a module defines its element class, its system, its observables, and its settings."""
    module = translate('sir.mdl')
    system = module.create_system()
    assert isinstance(system.elements[0], module.SirElement)
    assert all(isinstance(spec, ObservableSpec) for spec in module.observables(system))
    assert (module.START, module.UNTIL, module.DT, module.SAVE_EVERY) == (0.0, 40.0, 0.125, 0.125)


def test_run_auxiliaries_are_observables(fixtures_folder, translate):
    """Test that the trajectory records every auxiliary of the file under its snake-case name."""
    module = translate('sir.mdl')
    reference = _load_reference(fixtures_folder, 'sir.mdl')
    trajectory = _simulate(module)
    for name, column in (
        ('sir.prevalence', 'Prevalence'),
        ('sir.infection', 'Infection'),
        ('sir.recovery', 'Recovery'),
    ):
        values = _at_saved_times(trajectory, trajectory.values(name), module, reference)
        assert np.max(np.abs(values - reference[column].to_numpy())) < TOLERANCE


def test_run_sir_flows_are_named_transfers(translate):
    """Test that the infection is a transfer named after the flow, reported under the element, with people conserved."""
    module = translate('sir.mdl')
    system = module.create_system()
    infection = next(item for item in system.interactions if item.label == 'infection')
    assert isinstance(infection, TransferInteraction)
    assert (infection.state, infection.into, infection.flow) == ('susceptible', 'infected', 'infection')
    total = ObservableSpec('people', _total, element='sir')
    simulation = Simulation(
        system,
        solver=EulerSolver(dt=module.DT),
        dt=module.DT,
        observables=[total],
        invariants={'sir.people': 1e-6},
    )
    trajectory = simulation.check().prepare().run(until=module.UNTIL).trajectory_
    assert np.max(trajectory.values('sir.infection')) > 0


def test_run_invariant_catches_a_total_that_changes(translate):
    """Test that an invariant on a stock with births and deaths fails, so the conserved total of SIR means something."""
    module = translate('population.mdl')
    simulation = Simulation(
        module.create_system(),
        solver=EulerSolver(dt=module.DT),
        dt=module.DT,
        observables=[ObservableSpec('people', _growing_total, element='population')],
        invariants={'population.people': 1e-6},
    )
    with pytest.raises(ConservationError, match=r'population\.people'):
        simulation.check().prepare().run(until=module.UNTIL)


def test_run_loops_of_a_translated_model(translate):
    """Test that a population holds one reinforcing loop through births and one balancing loop through deaths."""
    analysis = LoopAnalysis().check().prepare(translate('population.mdl').create_system()).run()
    loops = dict(zip(analysis.loops_, analysis.polarities_, strict=True))
    assert loops == {
        ('population.population', 'population.births'): 'reinforcing',
        ('population.population', 'population.deaths'): 'balancing',
    }


def test_run_flows_with_the_environment_use_source_and_sink(translate):
    """Test that a flow of one stock moves from `Source()` when it enters and into `Sink()` when it leaves."""
    interactions = {item.label: item for item in translate('population.mdl').create_system().interactions}
    assert type(interactions['births'].state).__name__ == 'Source'
    assert type(interactions['deaths'].into).__name__ == 'Sink'


def test_run_goal_seeking_becomes_relaxation(fixtures_folder, translate):
    """Test that a stock changed by the gap to its target over a time relaxes, and that it runs like PySD."""
    module = translate('goal_seeking.mdl')
    system = module.create_system()
    assert [type(item) for item in system.interactions] == [RelaxationInteraction]
    relaxation = system.interactions[0]
    assert (relaxation.state, relaxation.goal.__name__, relaxation.time.__name__) == (
        'stock',
        '_goal__stock',
        '_time__stock',
    )
    reference = _load_reference(fixtures_folder, 'goal_seeking.mdl')
    trajectory = _simulate(module)
    values = _at_saved_times(trajectory, trajectory.values('goal_seeking.stock'), module, reference)
    assert np.max(np.abs(values - reference['Stock'].to_numpy())) < TOLERANCE
    observed = _at_saved_times(trajectory, trajectory.values('goal_seeking.stock_adjustment'), module, reference)
    assert np.max(np.abs(observed - (100.0 - values) / 4.0)) < TOLERANCE


def test_run_gap_times_a_rate_becomes_relaxation(tmp_path):
    """Test that the gap to a goal times a rate relaxes over the inverse of the rate."""
    text = 'Level = INTEG((Goal + 1 - Level) * Speed, 0) ~ ~ |\nGoal = 2 ~ ~ |\nSpeed = 0.5 ~ ~ |\n'
    source = _translate_text(tmp_path, text).run().source_
    assert 'RelaxationInteraction(' in source
    assert '1 / (s.speed)' in source


@pytest.mark.parametrize(
    'equation',
    ['(Level - Goal) / Speed', '-(Goal - Level) / Speed', '(Goal - Level) / Level', 'Level / Speed + Goal'],
)
def test_run_other_remaining_rates_become_rates(tmp_path, equation):
    """Test that a remaining rate of any other form becomes a rate interaction."""
    text = f'Level = INTEG({equation}, 1) ~ ~ |\nGoal = 2 ~ ~ |\nSpeed = 0.5 ~ ~ |\n'
    source = _translate_text(tmp_path, text).run().source_
    assert 'RateInteraction(' in source
    assert 'RelaxationInteraction(' not in source


def test_run_flow_of_several_stocks_stays_in_the_rates(tmp_path):
    """Test that a variable that enters two stocks stays in their remaining rates."""
    text = 'A = INTEG(Inflow, 1) ~ ~ |\nB = INTEG(Inflow, 1) ~ ~ |\nInflow = A + B ~ ~ |\n'
    source = _translate_text(tmp_path, text).run().source_
    assert 'TransferInteraction(' not in source
    assert source.count('RateInteraction(') == 2


def test_run_third_order_delay(fixtures_folder, translate):
    """Test that the output of `DELAY3` of a step is within the tolerance of PySD's run."""
    module = translate('third_order_delay.mdl')
    reference = _load_reference(fixtures_folder, 'third_order_delay.mdl')
    trajectory = _simulate(module)
    label = 'third_order_delay'
    for name, column in (('shipments', 'Shipments'), ('backlog', 'Backlog')):
        values = _at_saved_times(trajectory, trajectory.values(f'{label}.{name}'), module, reference)
        assert np.max(np.abs(values - reference[column].to_numpy())) < TOLERANCE
    assert reference['Shipments'].iloc[-1] > 14.0


@pytest.mark.parametrize(
    ('file_name', 'label', 'skipped'),
    [('signals.mdl', 'signals', {'Delayed Fixed'}), ('xmile_signals.xmile', 'xmile_signals', set())],
)
def test_run_built_in_functions_run_like_pysd(fixtures_folder, translate, file_name, label, skipped):
    """Test that every built-in function, with a start time other than zero, follows PySD's run."""
    module = translate(file_name)
    assert module.START != 0.0
    reference = _load_reference(fixtures_folder, file_name)
    trajectory = _simulate(module)
    for column in set(reference.columns) - skipped:
        name = f'{label}.{column.lower().replace(" ", "_")}'
        values = _at_saved_times(trajectory, trajectory.values(name), module, reference)
        assert np.max(np.abs(values - reference[column].to_numpy())) < TOLERANCE, column


def test_run_maps_min_of_max_to_clip(fixtures_folder):
    """Test that the source writes the built-in functions with the law utilities."""
    source = SystemDynamicsTranslator(fixtures_folder / 'signals.mdl').check().prepare().run().source_
    assert 'clip(s.level, 1.0, 3.0)' in source
    for name in ('lookup', 'where', 'step', 'ramp', 'pulse', 'pulse_train', 'delay1', 'smooth'):
        assert f'{name}(' in source
    assert 'distributed_delay(' in source
    assert 'delayed(s.level, 2)' in source
    assert 'minimum(' in source
    assert 'maximum(' in source


def test_run_random_functions_become_draws_that_rate_laws_reject(tmp_path, monkeypatch):
    """Test that random functions become draws, which the laws of a continuous model do not take yet."""
    text = (
        'Level = INTEG(Noise + Gauss + Unit + Plain, 1) ~ ~ |\nNoise = RANDOM UNIFORM(0, 1, 7) ~ ~ |\n'
        'Gauss = RANDOM NORMAL(-1, 1, 0, 0.5, 7) ~ ~ |\nUnit = RANDOM 0 1() ~ ~ |\nPlain = 1 + 2 ~ ~ |\n'
    )
    translator = _translate_text(tmp_path, text, 'noisy.mdl').run()
    assert "uniform('Noise_1', 0, 1)" in translator.source_
    assert "clip(normal('Gauss_1', 0, 0.5), -1, 1)" in translator.source_
    assert "uniform('Unit_1', 0, 1)" in translator.source_
    (tmp_path / 'translated_noisy.py').write_text(translator.source_)
    monkeypatch.syspath_prepend(tmp_path)
    module = importlib.import_module('translated_noisy')
    with pytest.raises(IncompatibleLawsError, match='random draw'):
        Simulation(module.create_system(), dt=0.25).check().prepare()


def test_run_regional_populations(fixtures_folder, translate):
    """Test that the trajectory holds one row per region for each stock, equal to PySD's run within the tolerance."""
    module = translate('regions.mdl')
    system = module.create_system()
    assert system.elements[0].get_params()['size'] == 3
    reference = _load_reference(fixtures_folder, 'regions.mdl')
    trajectory = _simulate(module)
    values = _at_saved_times(trajectory, trajectory.values('regions.population'), module, reference)
    for row, region in enumerate(('North', 'Middle', 'South')):
        assert np.max(np.abs(values[:, row] - reference[f'Population[{region}]'].to_numpy())) < TOLERANCE


def test_run_group_records_flows_and_auxiliaries(fixtures_folder, translate):
    """Test that flows have one row per member, and that auxiliaries record rows or one shared value."""
    module = translate('regions.mdl')
    reference = _load_reference(fixtures_folder, 'regions.mdl')
    trajectory = _simulate(module)
    births = _at_saved_times(trajectory, trajectory.values('regions.births'), module, reference)
    assert np.max(np.abs(births[:, 2] - reference['Births[South]'].to_numpy())) < TOLERANCE
    bonus = _at_saved_times(trajectory, trajectory.values('bonus'), module, reference)
    assert np.max(np.abs(bonus[:, 1] - reference['Bonus[Middle]'].to_numpy())) < TOLERANCE
    total = _at_saved_times(trajectory, trajectory.values('total_population'), module, reference)
    assert np.max(np.abs(total - reference['Total Population'].to_numpy())) < TOLERANCE


def test_run_sum_becomes_total(fixtures_folder):
    """Test that a sum over the range is a total, and a read of one region a reduction over that member."""
    source = SystemDynamicsTranslator(fixtures_folder / 'regions.mdl').check().prepare().run().source_
    assert 'return total(s.population)' in source
    assert 'total(s.population, where=s.member == 0)' in source


def test_run_unsupported_functions(fixtures_folder):
    """Test that one error names both variables and both functions."""
    translator = SystemDynamicsTranslator(fixtures_folder / 'unsupported.mdl').check().prepare()
    with pytest.raises(ValidationError) as error:
        translator.run()
    message = str(error.value)
    for word in ('Shipments', 'ALLOCATE BY PRIORITY', 'Sales Data', 'GET XLS DATA'):
        assert word in message


@pytest.mark.parametrize(
    ('text', 'words'),
    [
        ('Level = INTEG(Score, 1) ~ ~ |\nScore = GAME(Level) ~ ~ |\n', ['Score', 'GAME']),
        ('Level = INTEG(Trend, 1) ~ ~ |\nTrend = TREND(Level, 1, 0) ~ ~ |\n', ['Trend', 'TREND']),
        ('Level = INTEG(Chosen, 1) ~ ~ |\nChosen = FORECAST(Level, 1, 2) ~ ~ |\n', ['Chosen', 'FORECAST']),
        ('Level = INTEG(Late, 1) ~ ~ |\nLate = DELAY3I(Level + 1, 2, 0) ~ ~ |\n', ['Late', 'initial value']),
        ('Level = INTEG(Late, 1) ~ ~ |\nLate = SMOOTHI(Level + 1, 2, 0) ~ ~ |\n', ['Late', 'initial value']),
        ('Level = INTEG(Late, 1) ~ ~ |\nLate = DELAY FIXED(Level + 1, 2, 1) ~ ~ |\n', ['Late', 'DELAY FIXED']),
        ('Level = INTEG(Odd, 1) ~ ~ |\nOdd = QUANTUM(Level, 2) ~ ~ |\n', ['Odd', 'QUANTUM']),
        ('Level = INTEG(Lost, 1) ~ ~ |\nLost = Nowhere + 1 ~ ~ |\n', ['Lost', 'nowhere']),
        ('A = INTEG(B, 1) ~ ~ |\nB = C ~ ~ |\nC = B ~ ~ |\n', ['B', 'reads itself']),
        ('Level = INTEG(Late, Late) ~ ~ |\nLate = STEP(1, 2) ~ ~ |\n', ['Level', 'initial value']),
        (
            'R: a, b ~ ~ |\nP: c, d ~ ~ |\nM[R, P] = 1 ~ ~ |\nS[R] = INTEG(1, 0) ~ ~ |\n',
            ['M', 'more than one range'],
        ),
        ('R: a, b ~ ~ |\nP: c, d ~ ~ |\nM[P] = 1 ~ ~ |\nS[R] = INTEG(1, 0) ~ ~ |\n', ['M', 'over P']),
        ('R: a, b ~ ~ |\nS[R] = INTEG(1, 0) ~ ~ |\nU = INTEG(1, 0) ~ ~ |\n', ['U', 'unsubscripted stock']),
        (
            'R: a, b ~ ~ |\nP: c, d ~ ~ |\nS[R] = INTEG(1, 0) ~ ~ |\nU[P] = INTEG(1, 0) ~ ~ |\n',
            ['more than one range'],
        ),
        ('R: a, b ~ ~ |\nS[a] = INTEG(1, 0) ~~|\nS[b] = INTEG(2, 0) ~ ~ |\n', ['S', 'for every subscript']),
        ('Level = INTEG(Total, 1) ~ ~ |\nM = 1 ~ ~ |\nTotal = SUM(M) ~ ~ |\n', ['Total', 'SUM']),
        ('Lambda = INTEG(1, 0) ~ ~ |\n', ['Lambda', 'clashes']),
        ('Label = INTEG(1, 0) ~ ~ |\n', ['Label', 'clashes']),
        ('Member = INTEG(1, 0) ~ ~ |\n', ['Member', 'clashes']),
        ('Contact Rate = INTEG(1, 0) ~ ~ |\n"Contact-Rate" = 2 ~ ~ |\n', ['share the snake-case name']),
        (':MACRO: DOUBLE(x)\nDOUBLE = 2 * x ~ ~ |\n:END OF MACRO:\nLevel = INTEG(1, 0) ~ ~ |\n', ['double', 'macro']),
    ],
)
def test_run_reports_what_does_not_translate(tmp_path, text, words):
    """Test that every construct without a counterpart appears in one error with its variable."""
    translator = _translate_text(tmp_path, text)
    with pytest.raises(ValidationError) as error:
        translator.run()
    for word in words:
        assert word in str(error.value), str(error.value)


def test_run_reports_every_construct_at_once(tmp_path):
    """Test that one error lists every unsupported construct of a file."""
    text = 'Level = INTEG(A + B, 1) ~ ~ |\nA = GAME(Level) ~ ~ |\nB = QUANTUM(Level, 2) ~ ~ |\n'
    with pytest.raises(ValidationError) as error:
        _translate_text(tmp_path, text).run()
    assert 'A: GAME' in str(error.value)
    assert 'B: QUANTUM' in str(error.value)


def test_run_reports_a_file_name_that_is_no_identifier(tmp_path):
    """Test that a file whose name cannot name the element asks for a name, which then works."""
    translator = _translate_text(tmp_path, 'Level = INTEG(1, 0) ~ ~ |\n', '2 model.mdl')
    with pytest.raises(ValidationError, match='Give `name`'):
        translator.run()
    named = SystemDynamicsTranslator(tmp_path / '2 model.mdl', name='renamed').check().prepare().run()
    assert 'class RenamedElement(Element):' in named.source_


def test_run_reports_an_element_named_like_a_flow(tmp_path):
    """Test that a file named like an auxiliary asks for a name, since labels are unique among parts."""
    translator = _translate_text(tmp_path, 'Level = INTEG(Flow, 0) ~ ~ |\nFlow = Level + 1 ~ ~ |\n', 'flow.mdl')
    with pytest.raises(ValidationError, match='Give `name`'):
        translator.run()


def test_run_gives_generated_labels_that_no_variable_holds(tmp_path):
    """Test that the label of a generated interaction avoids a variable of that name."""
    text = 'Level = INTEG(2 * Level, 1) ~ ~ |\nLevel Rate = 3 ~ ~ |\n'
    source = _translate_text(tmp_path, text).run().source_
    assert "'level_rate_generated'" in source


def test_run_wraps_long_lines(tmp_path):
    """Test that the source keeps within the line length for a long expression."""
    long_sum = ' + '.join(f'Coefficient Number {index}' for index in range(20))
    constants = ''.join(f'Coefficient Number {index} = {index} ~ ~ |\n' for index in range(20))
    source = _translate_text(tmp_path, f'Level = INTEG({long_sum} * Level, 1) ~ ~ |\n{constants}').run().source_
    assert max(len(line) for line in source.splitlines()) <= 120


def test_run_leaves_out_delays_no_law_reads(tmp_path):
    """Test that the observables hold no delay that no law reads, which a run cannot record."""
    text = 'Level = INTEG(1, 0) ~ ~ |\nWatched = SMOOTH(Level, 2) ~ ~ |\nShown = Level * 2 ~ ~ |\n'
    source = _translate_text(tmp_path, text).run().source_
    assert "ObservableSpec('watched'" not in source
    assert "ObservableSpec('shown'" in source


def test_run_keeps_the_values_of_initial_expressions(tmp_path):
    """Test that the first values of stocks read constants, auxiliaries, and the time, at the start of a run."""
    text = (
        'A = INTEG(1, Size - 1) ~ ~ |\nB = INTEG(1, Half) ~ ~ |\nC = INTEG(1, A * 2) ~ ~ |\n'
        'D = INTEG(1, Time + 1) ~ ~ |\n'
        'E = INTEG(1, INITIAL(A + 1)) ~ ~ |\nF = INTEG(1, -Size) ~ ~ |\nSize = 10 ~ ~ |\nHalf = Size / 2 ~ ~ |\n'
    )
    source = _translate_text(tmp_path, text).run().source_
    for declaration in (
        'a: Annotated[float, StateSpec()] = 9.0',
        'b: Annotated[float, StateSpec()] = 5.0',
        'c: Annotated[float, StateSpec()] = 18.0',
        'd: Annotated[float, StateSpec()] = 1.0',
        'e: Annotated[float, StateSpec()] = 10.0',
        'f: Annotated[float, StateSpec()] = -10.0',
    ):
        assert declaration in source


@pytest.mark.parametrize('path', ['model.txt', 'missing.mdl'])
def test_check_rejects_a_path_without_a_model_file(tmp_path, path):
    """Test that a path of an unknown format or without a file raises an error."""
    (tmp_path / 'model.txt').write_text('text')
    with pytest.raises(ValidationError, match=r'model\.txt|missing\.mdl'):
        SystemDynamicsTranslator(tmp_path / path).check()


def test_check_rejects_a_name_that_is_no_identifier(fixtures_folder):
    """Test that a name that is no identifier raises an error."""
    with pytest.raises(ValidationError, match='not an identifier'):
        SystemDynamicsTranslator(fixtures_folder / 'teacup.mdl', name='not valid').check()


def test_prepare_reads_the_abstract_model(fixtures_folder):
    """Test that preparing keeps the abstract model PySD reads from the file."""
    translator = SystemDynamicsTranslator(fixtures_folder / 'teacup.xmile').check().prepare()
    assert [section.type for section in translator.model_.sections] == ['main']
