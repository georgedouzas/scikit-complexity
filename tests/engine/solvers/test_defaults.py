"""Test the `_defaults` module of `skcomplex.engine.solvers`."""

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RewritingSolver, ScipyODESolver
from skcomplex.engine.solvers._defaults import DEFAULT_SOLVERS
from skcomplex.models.biology.common.populations import ExponentialGrowthInteraction, PopulationElement
from skcomplex.types import FormKind


def test_default_solvers_solve_their_form():
    """Test that the default solver of every form solves that form."""
    for form, solver in DEFAULT_SOLVERS.items():
        assert form == solver._get_form() or form in solver._get_forms()


def test_simulation_without_a_solver_runs_the_default_solver():
    """Test that a simulation without a solver runs the default solver of the system's form."""
    system = System(
        elements=[PopulationElement('a', n=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    simulation = Simulation(system).check().prepare().run(until=0.1)
    assert type(simulation.stepper_) is DEFAULT_SOLVERS[FormKind.ODE] is ScipyODESolver


def test_simulation_runs_the_solver_instance_passed():
    """Test that a simulation runs the solver instance it is given."""
    system = System(
        elements=[PopulationElement('a', n=1)],
        interactions=[ExponentialGrowthInteraction('g', target='a', r=1.0)],
    )
    solver = ScipyODESolver(method='RK45')
    assert Simulation(system, solver=solver).check().prepare().run(until=0.1).stepper_ is solver


def test_rewriting_form_runs_the_rewriting_solver():
    """Test that a problem of the rewriting form runs on the rewriting solver when no solver is passed."""
    assert DEFAULT_SOLVERS[FormKind.REWRITING] is RewritingSolver
