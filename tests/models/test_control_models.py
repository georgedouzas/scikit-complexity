"""Test the control models: optimal controls of the library and the feedback controllers of engineering."""

import importlib
import pkgutil

import numpy as np
import pytest

import skcomplex.core
from skcomplex.core.analysis import controllability, linearize
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import Pontryagin
from skcomplex.core.system import System
from skcomplex.models import run
from skcomplex.models.biology.oncology import KirschnerPanettaTumour, TumourImmuneChemotherapy
from skcomplex.models.engineering.control import CartPendulum
from skcomplex.models.engineering.realizations import PID, Integrator, StateFeedback


def test_sir_vaccination_lowers_the_peak():
    """Test that the optimally vaccinated epidemic peaks lower than the epidemic without vaccination."""
    controlled = run('sir_vaccination').trajectory_
    uncontrolled = run('sir_vaccination', a=0.0).trajectory_
    assert np.all(uncontrolled['population.u'] == 0.0)
    assert controlled['population.I'].max() < 0.5 * uncontrolled['population.I'].max()
    assert controlled['population.u'][0] > controlled['population.u'][-1]


def test_pid_inverted_pendulum_returns_upright():
    """Test that the PID controller brings the pendulum from a tilt of 0.1 to within 0.01 of upright within 5 s."""
    trajectory = run('pid_inverted_pendulum').trajectory_
    assert trajectory['cart.theta'][0] == pytest.approx(0.1)
    late = trajectory.times >= 2.0
    assert np.all(np.abs(trajectory['cart.theta'][late]) < 0.01)
    assert np.all(np.abs(trajectory['tilt']) <= 0.1 + 1e-12)


def test_pendulum_without_control_falls():
    """Test that the pendulum without its controller falls away from upright."""
    trajectory = Simulation(System(components=[CartPendulum('cart', theta=0.1)]), dt=0.1).run(until=1.0).trajectory_
    assert trajectory['cart.theta'][-1] > 0.5


def test_lqr_oscillator_comes_to_rest():
    """Test that the regulated oscillator loses its energy, unlike the free oscillator, which keeps it."""
    trajectory = run('lqr_oscillator').trajectory_
    assert trajectory['energy'][0] == pytest.approx(0.5)
    assert trajectory['energy'][-1] < 1e-4
    free = run('lqr_oscillator', r=1e6).trajectory_
    assert free['energy'][-1] > 0.4


def test_lqr_oscillator_is_controllable():
    """Test that the oscillator with its force as control is controllable."""
    plant = run('lqr_oscillator', settings={'until': 0.1}).system
    plant = System(components=list(plant.components))
    model = linearize(plant)
    assert controllability(model.A, model.B) == 2


def test_optimal_harvesting_holds_the_singular_level():
    """Test that the optimal effort takes its bounds at both ends and holds the stock steady in between."""
    trajectory = run('optimal_harvesting').trajectory_
    effort, stock = trajectory['resource.effort'], trajectory['resource.x']
    assert effort[0] == pytest.approx(1.0, abs=1e-6)
    assert effort[-1] == pytest.approx(1.0, abs=1e-6)
    middle = (trajectory.times > 5) & (trajectory.times < 15)
    assert np.ptp(stock[middle]) < 1e-2
    np.testing.assert_allclose(trajectory['catch'], effort * stock)


def test_optimal_consumption_approaches_the_golden_rule():
    """Test that capital rises towards the modified golden rule `alpha k^(alpha - 1) = delta + rho`."""
    trajectory = run('optimal_consumption').trajectory_
    golden = (0.3 / (0.05 + 0.03)) ** (1 / 0.7)
    capital, consumption = trajectory['economy.k'], trajectory['economy.c']
    early = trajectory.times < 30.0
    assert np.all(np.diff(capital[early]) > 0)
    assert np.all(np.diff(consumption[early]) > 0)
    assert capital[-1] == pytest.approx(golden, rel=0.1)
    assert trajectory['saving_rate'][0] > trajectory['saving_rate'][-1]


def test_optimal_chemotherapy_shrinks_the_tumour():
    """Test that the optimal infusion ends with a smaller tumour than no treatment, starting at the largest dose."""
    treated = run('optimal_chemotherapy').trajectory_
    untreated = Simulation(System(components=[TumourImmuneChemotherapy('patient')]), dt=0.25).run(until=30.0)
    assert treated['patient.u'][0] == pytest.approx(1.0)
    assert treated['patient.A'][-1] < 0.5 * untreated.trajectory_['patient.A'][-1]


def test_optimal_immunotherapy_clears_the_tumour():
    """Test that the optimal adoptive therapy and IL-2 clear the tumour, which grows without them."""
    treated = run('optimal_immunotherapy').trajectory_
    assert treated['site.y'][-1] < 0.01
    assert treated['site.u1'][0] == pytest.approx(1.0)
    untreated = run('optimal_immunotherapy', w1=1e6, w2=1e6, settings={'until': 20.0}).trajectory_
    assert untreated['site.y'][-1] > 1.0


def test_optimal_control_models_record_their_costs():
    """Test that the running cost of the objective is recorded with the solution's cost."""
    simulation = run('sir_vaccination', settings={'until': 5.0, 'solver': Pontryagin(nodes=51)})
    assert np.all(simulation.trajectory_['population.cost'] > 0)
    assert simulation.stepper_.solution_.cost > 0


def test_controllers_live_only_in_the_engineering_realizations():
    """Test that no package of the core exports a controller, which the engineering realizations provide."""
    names = {'PID', 'StateFeedback', 'Integrator'}
    for module in pkgutil.walk_packages(skcomplex.core.__path__, 'skcomplex.core.'):
        if module.ispkg:
            assert not names & set(getattr(importlib.import_module(module.name), '__all__', [])), module.name
    assert issubclass(PID, object)
    assert StateFeedback('feedback', plant='plant').states == ()
    assert Integrator('memory').value == 0.0


def test_kirschner_panetta_is_not_registered_twice():
    """Test that the scaled tumour of the immunotherapy model is a new element, not the book's dimensional one."""
    assert KirschnerPanettaTumour('site').alpha1 == 2e7
