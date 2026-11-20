"""Test the `_base` module of `skcomplex.core.engine.runtime`."""

import inspect
import time

from skcomplex.core.base import BaseConfig
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine import runtime as package
from skcomplex.core.engine.runtime import BaseClock, BaseObserver
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.populations import PopulationElement


def _held(x: float = 1.0) -> System:
    """Return a population `a` whose rate is zero, so it holds its value at every frame."""
    hold = Interaction('hold', roles={'target': 'a'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t), 0)])
    return System(elements=[PopulationElement('a', x=x)], interactions=[hold])


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
    _held().prepare().simulate(until=3, dt=1, observers=[observer])
    assert observer.times_ == [0.0, 1.0, 2.0, 3.0]
    assert observer.ended_


def test_observer_verbs_return_the_observer():
    """Test that preparing and running an observer return it."""
    observer = _TimesObserver()
    problem = _held().prepare().simulate(until=0).problem
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


def test_public_names_are_classes():
    """Every public name is a class."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)), name


def test_classes_derive_from_a_base_of_the_package():
    """Every observer derives from the observer base and every clock from the clock base."""
    for name in package.__all__:
        item = getattr(package, name)
        if name.endswith('Observer'):
            assert issubclass(item, package.BaseObserver), name
        if name.endswith('Clock'):
            assert issubclass(item, package.BaseClock), name
