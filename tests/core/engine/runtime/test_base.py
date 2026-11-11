"""Test the `_base` module of `skcomplex.core.engine.runtime`."""

import inspect
import time

from skcomplex.core.base import BaseConfig
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine import runtime as package
from skcomplex.core.engine.runtime import BaseClock, BaseObserver, simulate
from skcomplex.systems.biology.common.populations import PopulationElement

CATALOG_NAMES = {
    'SYSTEMS',
    'SystemEntry',
    'catalog',
    'create_simulation',
    'create_system',
    'describe',
    'register_system',
    'summarize',
}


class _TimesObserver(BaseObserver):
    """Observer that keeps the times of its frames and whether the run ended."""

    def _prepare(self, context):
        self.times_ = []
        self.ended_ = False

    def _run(self, inputs):
        if inputs is not None:
            self.times_.append(inputs.time)

    def _release(self):
        self.ended_ = True


def test_bases_derive_from_the_configurable_root():
    """Test that the observer and the clock bases derive from the root of every configurable class."""
    assert issubclass(BaseObserver, BaseConfig)
    assert issubclass(BaseClock, BaseConfig)


def test_verbs_take_one_input():
    """Test that an observer prepares on a problem and runs on a frame, and a clock prepares on nothing."""
    assert list(inspect.signature(BaseObserver.prepare).parameters) == ['self', 'problem']
    assert list(inspect.signature(BaseObserver.run).parameters) == ['self', 'frame', 'final']
    assert list(inspect.signature(BaseClock.prepare).parameters) == ['self', 'context']
    assert list(inspect.signature(BaseClock.run).parameters) == ['self', 'elapsed', 'final']


def test_observer_receives_every_frame_and_the_end_of_the_run():
    """Test that a simulation prepares an observer, runs it at every frame, and releases it at the end."""
    observer = _TimesObserver()
    simulate(CompositeSystem(elements=[PopulationElement('a')]), until=3, dt=1, observers=[observer])
    assert observer.times_ == [0.0, 1.0, 2.0, 3.0]
    assert observer.ended_


def test_observer_verbs_return_the_observer():
    """Test that preparing and running an observer return it."""
    observer = _TimesObserver()
    problem = simulate(CompositeSystem(elements=[PopulationElement('a')]), until=0).problem
    assert observer.prepare(problem) is observer
    assert observer.run(final=True) is observer
    assert observer.ended_


def test_clock_prepare_marks_the_start():
    """Test that preparing a clock marks the wall time of the start and no lag."""
    before = time.perf_counter()
    clock = BaseClock().prepare()
    assert before <= clock.started_ <= time.perf_counter()
    assert clock.max_lag_ == 0.0
    assert clock.run(1.0) is clock


def test_public_names_are_classes_or_simulate():
    """Every public name is a class, apart from `simulate`, the function that runs a system."""
    for name in set(package.__all__) - {'simulate'}:
        assert inspect.isclass(getattr(package, name)), name
    assert inspect.isfunction(package.simulate)


def test_no_catalog_in_the_engine():
    """No public name registers, lists, describes, creates, or runs systems by name."""
    assert not CATALOG_NAMES & set(package.__all__)
    assert not CATALOG_NAMES & set(dir(package))


def test_classes_derive_from_a_base_of_the_package():
    """Every observer derives from the observer base and every clock from the clock base."""
    for name in package.__all__:
        item = getattr(package, name)
        if name.endswith('Observer'):
            assert issubclass(item, package.BaseObserver), name
        if name.endswith('Clock'):
            assert issubclass(item, package.BaseClock), name
