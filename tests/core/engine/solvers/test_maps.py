"""Test the iteration of discrete maps."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import where
from skcomplex.core.engine.solvers import IterationSolver, StepSpec
from skcomplex.core.systems import System
from skcomplex.systems.economics.markets import CobwebInteraction, MarketElement


def test_iteration_applies_one_step_per_unit_time():
    """Test that the discrete stepper applies the map once per integer step."""
    system = System(
        elements=[MarketElement('m', price=1.0)],
        interactions=[CobwebInteraction('c', target='m', a=10, b=2, c=1, d=1)],
    )
    problem = system.prepare()._problem
    stepper = IterationSolver()
    stepper.set_params(random_state=np.random.default_rng(0)).prepare(problem)
    assert stepper.run(StepSpec(0.0, problem.initial_state, problem.initial_parameters, 2.0)).state_.tolist() == [2.5]


def _runs(activity):
    """Return the system of one counter whose update law has an activity that follows a given series."""

    def laws(s):
        """Return the count of steps, active by the series at every count."""
        n = s.target.n
        level = sum(where(n == step, value, 0) for step, value in enumerate(activity))
        return [UpdateLaw(n, n + 1, activity=level, label='burst')]

    rule = Interaction('count', roles={'target': 'counter'}, laws=laws)
    return System(elements=[Element('counter', values={'n': 0.0})], interactions=[rule])


def test_iteration_records_the_runs_of_activity_of_one_member():
    """Test that the activity `0, 2, 3, 0, 1, 0` of a member of no space gives two events of sizes 5 and 1."""
    trajectory = _runs([0, 2, 3, 0, 1, 0]).prepare().simulate(until=5, dt=1)
    assert [(event.label, event.time) for event in trajectory.events] == [('burst', 1.0), ('burst', 4.0)]
    assert [(event.size, event.duration, event.profile) for event in trajectory.events] == [(5, 2, (2, 3)), (1, 1, (1,))]


def test_iteration_leaves_out_an_unfinished_event():
    """Test that an event still active at the end of a run is not recorded."""
    assert not _runs([0, 2, 3, 0, 1, 0]).prepare().simulate(until=2, dt=1).events
