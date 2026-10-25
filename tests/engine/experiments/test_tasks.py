"""Test the `_tasks` module of `skcomplex.engine.experiments`."""

from skcomplex.engine.compile import compile_system
from skcomplex.engine.experiments._tasks import _Schedule
from skcomplex.engine.runtime import BaseObserver
from skcomplex.models.economics.resources import harvesting


def test_schedule_is_a_function_of_time_not_an_observer():
    """Test that the schedule keeps the initial value before its first change and the last due change after it."""
    problem = compile_system(harvesting())
    schedule = _Schedule([(50.0, 'harvest.effort', 0.025), (20.0, 'harvest.effort', 0.04)])
    function = schedule.functions(problem)['harvest.effort']
    initial = problem.parameters.read(problem.initial_parameters, 'harvest.effort')
    assert not isinstance(schedule, BaseObserver)
    assert [float(function(time)) for time in (0.0, 20.0, 49.0, 50.0)] == [initial, 0.04, 0.04, 0.025]
