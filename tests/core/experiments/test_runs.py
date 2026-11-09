"""Test the `_runs` module of `skcomplex.core.experiments`: the runs shared by the experiments, and the surface."""

import inspect
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core import experiments
from skcomplex.core.analysis import bind_analysis
from skcomplex.core.base import BaseConfig
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.errors import ExperimentError, ValidationError
from skcomplex.core.experiments import calibrate, ensemble, orbit_diagram, sweep
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.systems.biology.ecology import LogisticGrowthSystem
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem


class LogisticMapElement(Element):
    """Logistic map element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.2,
        r: Annotated[object, ParameterSpec()] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r

    def _get_laws(self, s):
        """Return the laws of the logistic map element."""
        return [UpdateLaw(s.x, s.r * s.x * (1 - s.x))]


@dataclass(frozen=True)
class PeakResult:
    """Height and time of the peak of one series."""

    peak: float
    peak_time: float


def peak(trajectory, *, variable='I.x'):
    """Return the height and time of the peak of one series of a trajectory."""
    values = np.asarray(trajectory.values(variable), dtype=float)
    return PeakResult(float(values.max()), float(trajectory.times[values.argmax()]))


def test_surface_holds_functions_and_results():
    """Test that the public names are experiment functions and frozen results, and none is an estimator."""
    for name in experiments.__all__:
        item = getattr(experiments, name)
        if name.endswith('Result'):
            assert item.__dataclass_params__.frozen, name
        else:
            assert inspect.isfunction(item), name
        assert not (inspect.isclass(item) and issubclass(item, BaseConfig)), name
    for name in (
        'Experiment',
        'GridDesign',
        'BaseExperiment',
        'BasinStabilityExperiment',
        'CalibrationExperiment',
        'Calibration',
    ):
        assert not hasattr(experiments, name), name


def test_system_and_factory_give_the_same_run():
    """Test that a model parameter set on a copy of a system, passed to a factory, or set by path gives one run."""
    results = [
        sweep(LogisticGrowthSystem(), {'r': [0.7]}, until=10, dt=0.1, outputs=['population.x']),
        sweep(LogisticGrowthSystem, {'r': [0.7]}, until=10, dt=0.1, outputs=['population.x']),
        sweep(
            LogisticGrowthSystem().prepare().system_,
            {'growth.r': [0.7]},
            until=10,
            dt=0.1,
            outputs=['population.x'],
        ),
    ]
    for result in results[1:]:
        np.testing.assert_array_equal(results[0].series('population.x'), result.series('population.x'))


def test_a_sweep_leaves_the_system_unchanged():
    """Test that a sweep sets model parameters on copies, never on the system it is given."""
    system = LogisticGrowthSystem(r=0.3)
    sweep(system, {'r': [0.7, 0.9]}, until=1, outputs=['population.x'])
    assert system.r == 0.3


def test_paths_of_states_and_parameters_change_one_run():
    """Test that a dotted state path changes the initial value and a parameter path the value for the whole run."""
    system = CompositeSystem(elements=[LogisticMapElement('map')])
    result = sweep(system, [{'map.x': 0.5, 'map.r': 2.0}], until=3, outputs=['map.x'])
    assert result.series('map.x')[0].tolist() == [0.5, 0.5, 0.5, 0.5]


def test_analyses_of_one_function_are_numbered():
    """Test that two analyses of one function get numbered column prefixes."""
    analyses = [bind_analysis(peak, variable='I.x'), bind_analysis(peak, variable='S.x')]
    result = sweep(SIRSystem(), [{}], analyses=analyses, until=10)
    row = result.rows[0]
    assert row['peak-1.peak'] > 0
    assert row['peak-2.peak'] == result.series('S.x')[0].max()


def test_an_analysis_must_return_a_result():
    """Test that an analysis that returns no frozen result raises an error naming it."""
    with pytest.raises(ExperimentError, match='The analysis last returned float'):
        sweep(SIRSystem(), [{}], analyses=[last], until=1)


def last(trajectory):
    """Return no result of an analysis."""
    return 1.0


@pytest.mark.slow
def test_serial_and_process_runs_are_identical():
    """Test that 20 stochastic runs with seed 7 and a peak analysis give identical results serially and on processes."""
    results = [
        ensemble(
            SIRSystem(stochastic=True),
            replicates=20,
            analyses=[peak],
            until=50,
            outputs=['I.x'],
            random_state=7,
            executor=executor,
        )
        for executor in ('serial', 'process')
    ]
    np.testing.assert_array_equal(results[0].series('I.x'), results[1].series('I.x'))
    assert results[0].rows == results[1].rows


def test_thread_runs_equal_serial_runs():
    """Test that runs on threads give the same results as serial runs."""
    results = [
        ensemble(SIRSystem(stochastic=True), replicates=4, until=20, outputs=['I.x'], random_state=7, executor=executor)
        for executor in ('serial', 'thread')
    ]
    np.testing.assert_array_equal(results[0].series('I.x'), results[1].series('I.x'))


@pytest.mark.parametrize(
    'experiment',
    [
        lambda: sweep('sir', {'gamma': [0.1]}, until=1),
        lambda: calibrate('logistic_growth', {'population.x': ([0.0], [1.0])}, {}),
        lambda: orbit_diagram('logistic_map', 'r', [3.0], 'map.x', transient=1, keep=1),
    ],
)
def test_system_name_raises_a_named_error(experiment):
    """Test that an experiment of a system name raises an error that says to pass a system or a factory."""
    with pytest.raises(ExperimentError, match=r"not the system name '\w+'\. Pass an instance of the system class"):
        experiment()


def test_unknown_names_are_rejected():
    """Test that a name that is no model parameter, and a path that is no state or parameter, name what is available."""
    with pytest.raises(ValidationError, match=r'rate is not a model parameter of LogisticGrowthSystem\. Its model'):
        sweep(LogisticGrowthSystem(), {'rate': [1.0]}, until=1)
    with pytest.raises(ValidationError, match=r'growth\.rate is neither a state nor a parameter'):
        sweep(LogisticGrowthSystem(), {'growth.rate': [1.0]}, until=1)


def test_invalid_settings_are_rejected():
    """Test that zero replicates and an unknown executor raise errors naming them."""
    with pytest.raises(ValidationError, match='replicates = 0'):
        ensemble(SIRSystem(stochastic=True), replicates=0)
    with pytest.raises(ValidationError, match="executor = 'gpu'"):
        ensemble(SIRSystem(stochastic=True), replicates=1, executor='gpu')


def test_a_system_of_the_library_runs_as_an_instance_or_its_class():
    """Test that a catalog system runs as an instance or as its class, the factory of its instances."""
    for source in (LogisticMapSystem(), LogisticMapSystem):
        diagram = orbit_diagram(source, 'r', [2.9], 'map.x', transient=1000, keep=4, dt=1)
        assert diagram.periods.tolist() == [1]


def test_runs_of_one_structure_compile_once_per_call(monkeypatch):
    """Test that runs differing by dotted paths and seeds share one compile, and model parameters compile each."""
    calls = []
    compile_problem = experiments._runs._compile_problem

    def counted(system, **settings):
        calls.append(system.get_params(deep=False))
        return compile_problem(system, **settings)

    monkeypatch.setattr(experiments._runs, '_compile_problem', counted)
    rows = [{'I.x': value} for value in (1.0, 5.0, 10.0)]
    result = sweep(SIRSystem(), rows, replicates=2, until=5, outputs=['I.x'], random_state=0)
    assert len(calls) == 1
    np.testing.assert_allclose(result.series('I.x')[::2, 0], [1.0, 5.0, 10.0])
    sweep(SIRSystem(), rows, until=5, outputs=['I.x'])
    assert len(calls) == 2
    sweep(SIRSystem(), [{'beta': 0.2}, {'beta': 0.4}, {'beta': 0.2}], until=5, outputs=['I.x'])
    assert len(calls) == 4
