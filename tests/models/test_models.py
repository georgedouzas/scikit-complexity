"""Test the ready-made models against their known behavior."""

import numpy as np
import pytest

from skcomplex.core.base import Form
from skcomplex.core.compile import compile_system
from skcomplex.core.model import ConstantForce, ContinuousSpace, Gravity, LinearDrag, Particle, Spring, System
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.biology.ecology import LogisticGrowth, Mortality, Population, Predation
from skcomplex.models.biology.epidemiology import Compartment, Infection, Recovery
from skcomplex.models.economics.finance import Asset, GeometricBrownianMotion
from skcomplex.models.economics.macroeconomics import Economy, SolowGrowth
from skcomplex.models.economics.markets import Cobweb, Market


def test_particle_constant_acceleration():
    """Test that a constant force displaces a particle by `F * t**2 / (2 * m)`."""
    system = System(
        components=[Particle('p', m=2.0)],
        interactions=[ConstantForce('push', target='p', F=[4.0, 0.0, -2.0])],
        space=ContinuousSpace(n_dim=3),
    )
    position = Simulation(system, dt=0.5).run(until=2).trajectory_['p.x'][-1]
    assert np.allclose(position, [4.0, 0.0, -2.0])


def test_spring_conserves_energy():
    """Test that two particles joined by a spring keep their total energy."""
    system = System(
        components=[Particle('a', x=[0.0, 0.0], v=[0.0, 1.0]), Particle('b', x=[2.0, 0.0], m=2.0)],
        interactions=[Spring('spring', first='a', second='b', k=3.0, length=1.0)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = Simulation(system, solver=ScipyODE(rtol=1e-10, atol=1e-12), dt=0.1).run(until=5).trajectory_
    kinetic = 0.5 * (trajectory['a.v'] ** 2).sum(axis=1) + 0.5 * 2.0 * (trajectory['b.v'] ** 2).sum(axis=1)
    distance = np.linalg.norm(trajectory['a.x'] - trajectory['b.x'], axis=1)
    energy = kinetic + 0.5 * 3.0 * (distance - 1.0) ** 2
    assert np.allclose(energy, energy[0], rtol=1e-6)


def test_gravity_keeps_circular_orbit():
    """Test that a light body on a circular orbit keeps its radius."""
    system = System(
        components=[Particle('sun', m=1000.0), Particle('planet', x=[10.0, 0.0], v=[0.0, 10.0], m=1e-6)],
        interactions=[Gravity('gravity', first='sun', second='planet', G=1.0)],
        space=ContinuousSpace(n_dim=2),
    )
    trajectory = Simulation(system, solver=ScipyODE(rtol=1e-10, atol=1e-12), dt=0.5).run(until=6).trajectory_
    radius = np.linalg.norm(trajectory['planet.x'] - trajectory['sun.x'], axis=1)
    assert np.allclose(radius, 10.0, rtol=1e-4)


def test_linear_drag_slows_particle():
    """Test that drag decays the velocity exponentially."""
    system = System(
        components=[Particle('p', v=1.0)],
        interactions=[LinearDrag('air', target='p', gamma=0.5)],
        space=ContinuousSpace(n_dim=1),
    )
    velocity = Simulation(system, dt=0.5).run(until=2).trajectory_['p.v'][-1, 0]
    assert velocity == pytest.approx(np.exp(-1.0), rel=1e-4)


def test_logistic_growth_saturates():
    """Test that logistic growth approaches the carrying capacity."""
    system = System(components=[Population('p', n=5)], interactions=[LogisticGrowth('g', target='p', r=1.0, K=200)])
    assert Simulation(system, dt=1.0).run(until=30).trajectory_['p.n'][-1] == pytest.approx(200, rel=1e-3)


def test_lotka_volterra_oscillates():
    """Test that predators and prey cycle without dying out."""
    system = System(
        components=[Population('rabbits', n=40), Population('foxes', n=9)],
        interactions=[
            LogisticGrowth('growth', target='rabbits', r=1.1, K=1e6),
            Predation('hunt', predator='foxes', prey='rabbits', a=0.4, b=0.1),
            Mortality('death', target='foxes', d=0.4),
        ],
    )
    trajectory = Simulation(system, dt=0.1).run(until=50).trajectory_
    rabbits = trajectory['rabbits.n']
    assert rabbits.min() > 0
    assert rabbits.max() > 2 * rabbits.min()


def test_epidemic_stochastic_mode_is_jump_process():
    """Test that stochastic infection and recovery give integer counts of a jump process."""
    system = System(
        components=[Compartment('S', n=990), Compartment('I', n=10), Compartment('R')],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=0.0005, stochastic=True),
            Recovery('recovery', infected='I', recovered='R', gamma=0.1, stochastic=True),
        ],
    )
    assert compile_system(system).form is Form.JUMP
    states = Simulation(system, dt=1.0, random_state=3).run(until=50).trajectory_.states
    assert np.all(states >= 0)
    assert np.all(states == np.round(states))
    assert np.allclose(states.sum(axis=1), 1000)


def test_epidemic_deterministic_mode_conserves_population():
    """Test that the deterministic SIR model keeps the total population."""
    system = System(
        components=[Compartment('S', n=990), Compartment('I', n=10), Compartment('R')],
        interactions=[
            Infection('infection', susceptible='S', infected='I', beta=0.0005),
            Recovery('recovery', infected='I', recovered='R', gamma=0.1),
        ],
    )
    states = Simulation(system, dt=1.0).run(until=50).trajectory_.states
    assert np.allclose(states.sum(axis=1), 1000)


def test_solow_reaches_steady_state():
    """Test that Solow capital approaches `(s / delta) ** (1 / (1 - alpha))`."""
    s, alpha, delta = 0.3, 0.4, 0.1
    system = System(
        components=[Economy('country', k=1.0)],
        interactions=[SolowGrowth('solow', target='country', s=s, alpha=alpha, delta=delta)],
    )
    capital = Simulation(system, dt=10.0).run(until=300).trajectory_['country.k'][-1]
    assert capital == pytest.approx((s / delta) ** (1 / (1 - alpha)), rel=1e-3)


def test_cobweb_converges_to_equilibrium():
    """Test that a cobweb market with `d < b` converges to `(a - c) / (b + d)`."""
    a, b, c, d = 10.0, 2.0, 1.0, 1.0
    system = System(components=[Market('m', price=1.0)], interactions=[Cobweb('c', target='m', a=a, b=b, c=c, d=d)])
    price = Simulation(system).run(until=60).trajectory_['m.price'][-1]
    assert price == pytest.approx((a - c) / (b + d))


def test_asset_price_stays_positive_and_reproducible():
    """Test that geometric Brownian motion stays positive and is reproducible."""
    system = System(
        components=[Asset('stock', price=100.0)],
        interactions=[GeometricBrownianMotion('gbm', target='stock', mu=0.05, sigma=0.2)],
    )
    first = Simulation(system, dt=0.1, random_state=7).run(until=5).trajectory_['stock.price']
    second = Simulation(system, dt=0.1, random_state=7).run(until=5).trajectory_['stock.price']
    assert np.all(first > 0)
    assert np.array_equal(first, second)
