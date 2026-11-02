"""Test the `_base` module of `skcomplex.engine.runtime`."""

import inspect
import time

from skcomplex.base import BaseConfig
from skcomplex.core.system import System
from skcomplex.engine.runtime import BaseClock, BaseObserver, Simulation
from skcomplex.models.biology.common.populations import PopulationElement


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
    Simulation(System(elements=[PopulationElement('a')]), dt=1, observers=[observer]).check().prepare().run(until=3)
    assert observer.times_ == [0.0, 1.0, 2.0, 3.0]
    assert observer.ended_


def test_observer_verbs_return_the_observer():
    """Test that preparing and running an observer return it."""
    observer = _TimesObserver()
    problem = Simulation(System(elements=[PopulationElement('a')])).check().prepare().problem_
    assert observer.check().prepare(problem) is observer
    assert observer.run(final=True) is observer
    assert observer.ended_


def test_clock_prepare_marks_the_start():
    """Test that preparing a clock marks the wall time of the start and no lag."""
    before = time.perf_counter()
    clock = BaseClock().check().prepare()
    assert before <= clock.started_ <= time.perf_counter()
    assert clock.max_lag_ == 0.0
    assert clock.run(1.0) is clock
