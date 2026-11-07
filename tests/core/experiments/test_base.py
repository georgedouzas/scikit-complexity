"""Test the `_base` module of `skcomplex.core.experiments`: the bases of designs and experiments, and the surface."""

import inspect

import numpy as np
import pytest
from scipy import stats

from skcomplex.core import experiments
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import ExperimentError, ValidationError
from skcomplex.core.experiments import BaseDesign, BaseExperiment, RandomDesign
from skcomplex.systems import create_system
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.mathematics.dynamical_systems import logistic_map
from skcomplex.systems.mathematics.stochastic_processes import weakening_ornstein_uhlenbeck


def test_surface_holds_classes_and_the_sensitivity_function():
    """Test that every public name of the package is a class or the sensitivity function, and the old names are gone."""
    assert all(inspect.isclass(getattr(experiments, name)) for name in experiments.__all__ if name != 'sensitivity')
    assert inspect.isfunction(experiments.sensitivity)
    for name in ('calibrate', 'SensitivityAnalysis', 'run_task', 'resolve_source', 'Task', 'RunResult', 'Source'):
        assert not hasattr(experiments, name), name


def test_bases_are_abstract():
    """Test that the bases of designs and experiments cannot be built without their hooks."""
    assert inspect.isabstract(BaseDesign)
    assert inspect.isabstract(BaseExperiment)


def test_sampled_design_reads_distributions():
    """Test that a sampled design maps its samples through the quantile function of a SciPy distribution."""
    rows = RandomDesign(n=200, random_state=0).prepare({'r': stats.norm(10, 1)}).run().rows_
    assert 9.5 < sum(row['r'] for row in rows) / len(rows) < 10.5


@pytest.mark.parametrize('bound', [(0.0, 1.0, 2.0), 'low', 3.0], ids=['triple', 'text', 'number'])
def test_sampled_design_rejects_invalid_bounds(bound):
    """Test that a bound that is neither a pair nor a distribution raises an error naming it when prepared."""
    with pytest.raises(ValidationError, match='The bounds of r must be'):
        RandomDesign(n=4).prepare({'r': bound})


def test_design_check_rejects_negative_size():
    """Test that a negative number of rows builds and fails when checked."""
    with pytest.raises(ValidationError, match='nonnegative integer'):
        RandomDesign(n=-1).prepare()


def test_replicate_runs_many_initial_conditions_at_once():
    """Test that member i of a replicated logistic map follows the single map from the i-th initial value."""
    starts = np.linspace(0.05, 0.95, 100)
    ensemble = (
        Simulation(BaseExperiment._replicate(logistic_map(r=3.7), 100, {'map.x': starts}), dt=1).prepare().run(until=50)
    )
    values = np.asarray(ensemble.trajectory_.values('map.x'))
    for index in (0, 37, 99):
        single = Simulation(logistic_map(x=float(starts[index]), r=3.7), dt=1).prepare().run(until=50)
        np.testing.assert_allclose(values[:, index], np.ravel(single.trajectory_.values('map.x')), atol=1e-12)


def test_replicate_runs_systems_with_interactions():
    """Test that member i of replicated predators and prey follows the single system from the i-th initial prey."""
    starts = np.asarray([20.0, 40.0, 60.0])
    ensemble = BaseExperiment._replicate(LotkaVolterraSystem().prepare().system_, 3, {'prey.n': starts})
    values = np.asarray(Simulation(ensemble, dt=0.1).prepare().run(until=5).trajectory_.values('predators.n'))
    for index, start in enumerate(starts):
        single = Simulation(LotkaVolterraSystem(prey=float(start)).prepare().system_, dt=0.1).prepare().run(until=5)
        np.testing.assert_allclose(values[:, index], np.ravel(single.trajectory_.values('predators.n')), rtol=1e-5)


def test_replicate_rejects_systems_with_groups():
    """Test that a system with a group does not replicate and the error says why."""
    with pytest.raises(ValidationError, match='Only systems of plain single elements and their interactions'):
        BaseExperiment._replicate(weakening_ornstein_uhlenbeck(4), 3)


def test_replicate_checks_paths_and_rows():
    """Test that an unknown path and values without one row per member raise named errors."""
    with pytest.raises(ValidationError, match=r'map\.z is not a state or parameter'):
        BaseExperiment._replicate(logistic_map(), 3, {'map.z': [1, 2, 3]})
    with pytest.raises(ValidationError, match='one row per member'):
        BaseExperiment._replicate(logistic_map(), 3, {'map.r': [3.5, 3.6]})


@pytest.mark.parametrize(
    'experiment',
    [
        lambda: experiments.Experiment('sir', until=1),
        lambda: experiments.CalibrationExperiment('logistic_growth', {'population.n': ([0.0], [1.0])}, {}),
        lambda: experiments.OrbitDiagramExperiment('logistic_map', 'map.r', [3.0], 'map.x', transient=1, keep=1),
    ],
)
def test_system_name_as_a_source_raises_a_named_error(experiment):
    """Test that an experiment of a system name raises an error that says to pass a system or a factory."""
    with pytest.raises(ExperimentError, match=r"not the system name '\w+'\. Pass the system's factory"):
        experiment().prepare()


def test_system_of_the_library_runs_from_its_factory_or_a_created_system():
    """Test that a catalog system runs as an experiment from its factory or from a system the catalog creates."""
    for source in (logistic_map, create_system('logistic_map')):
        diagram = experiments.OrbitDiagramExperiment(source, 'map.r', [2.9], 'map.x', transient=1000, keep=4, dt=1)
        assert diagram.prepare().run().periods_.tolist() == [1]
