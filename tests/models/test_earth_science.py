"""Test the climate, oceanography, and seismology models against their known behavior."""

import warnings

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.earth.climate import (
    budyko_sellers,
    carbon_cycle,
    daisyworld,
    energy_balance,
    enso,
    geodynamo,
    hasselmann,
    lorenz_84,
    lorenz_96,
    saltzman_glaciation,
    soil_temperature,
    van_der_pol_ice_ages,
)
from skcomplex.models.earth.oceanography import stommel
from skcomplex.models.earth.seismology import burridge_knopoff


def _run(system, until, dt, **settings):
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, **settings).run(until=until).trajectory_


def _sign_changes(values):
    """Return the number of times a series changes sign."""
    return int((np.diff(np.sign(values)) != 0).sum())


def test_ice_albedo_bistability():
    """Test that the energy balance settles in a warm and in a frozen climate at the same solar constant."""
    warm = _run(energy_balance(T=300.0), until=1000.0, dt=10.0)['earth.T'][-1]
    cold = _run(energy_balance(T=220.0), until=1000.0, dt=10.0)['earth.T'][-1]
    assert abs(warm - 286.4) < 0.5
    assert abs(cold - 231.7) < 0.5


def test_budyko_sellers_has_ice_caps_and_a_snowball():
    """Test that present insolation keeps both a climate with polar ice caps and a snowball Earth."""
    capped = _run(budyko_sellers(18), until=40.0, dt=4.0)['bands.T'][-1]
    snowball = _run(budyko_sellers(18, T=-50.0), until=40.0, dt=4.0)['bands.T'][-1]
    assert capped[0] > 20.0 > -10.0 > capped[-1]
    assert np.all(np.diff(capped) < 0)
    assert np.all(snowball < -10.0)


def test_daisyworld_regulates_temperature():
    """Test that white and black daisies coexist and bring the temperature closer to their optimum than bare ground."""
    planet = _run(built := daisyworld(), until=100.0, dt=1.0, observables=describe('daisyworld').observables_of(built))
    bare = _run(
        built := daisyworld(white=0.0, black=0.0),
        until=1.0,
        dt=1.0,
        observables=describe('daisyworld').observables_of(built),
    )
    assert planet['planet.white'][-1] > 0.1
    assert planet['planet.black'][-1] > 0.1
    assert abs(planet['temperature'][-1] - 295.5) < abs(bare['temperature'][-1] - 295.5) - 3.0


def test_hasselmann_has_stationary_variance():
    """Test that the stochastic climate fluctuates with the variance `sigma**2 / (2 * lam)`."""
    solver = describe('hasselmann').defaults['solver']
    anomaly = _run(hasselmann(lam=1.0, sigma=0.5), until=400.0, dt=0.5, solver=solver)['ocean.T'][20:]
    assert abs(anomaly.var() - 0.125) < 0.04
    assert abs(anomaly.mean()) < 0.1


def test_enso_oscillates_with_period_longer_than_twice_the_delay():
    """Test that the delayed oscillator alternates between El Niño and La Niña with a period above twice the delay."""
    anomaly = _run(enso(), until=200.0, dt=0.5)['pacific.T'][100:]
    changes = _sign_changes(anomaly)
    assert changes >= 6
    assert 150.0 / (changes / 2) > 2 * 6.0
    assert anomaly.max() > 1.0
    assert anomaly.min() < -1.0


def test_saltzman_glaciation_has_a_100_kyr_cycle():
    """Test that the unforced glacial model oscillates with a period of about 100 kyr."""
    trajectory = _run(saltzman_glaciation(), until=50.0, dt=0.05)
    ice = trajectory['climate.x']
    rising = trajectory.times[1:][(ice[:-1] < 0) & (ice[1:] >= 0)]
    periods = np.diff(rising[rising > 10.0])
    assert len(periods) >= 2
    assert np.allclose(periods, 10.6, atol=0.5)


def test_carbon_cycle_is_excitable():
    """Test that a small loss of dissolved carbon decays while a larger one sets off a large excursion."""
    small = _run(carbon_cycle(w=2150.0), until=20.0, dt=0.1)
    large = _run(carbon_cycle(w=2050.0), until=20.0, dt=0.1)
    for trajectory in (small, large):
        assert np.allclose([trajectory['ocean.c'][-1], trajectory['ocean.w'][-1]], [83.58, 2197.37], atol=0.1)
    assert np.ptp(small['ocean.c']) < 20.0
    assert np.ptp(large['ocean.c']) > 100.0


def test_lorenz_84_relaxes_to_the_hadley_circulation():
    """Test that without eddies or land-sea forcing the jet relaxes to the forcing `F`."""
    jet = _run(lorenz_84(y=0.0, z=0.0, G=0.0), until=60.0, dt=1.0)['atmosphere.x'][-1]
    assert abs(jet - 8.0) < 1e-3


def test_lorenz_96_steady_state_is_unstable():
    """Test that the uniform state `F` of Lorenz-96 is kept exactly and that a small perturbation of it grows."""
    steady = _run(lorenz_96(8, perturbation=0.0), until=2.0, dt=0.5)['ring.x']
    assert np.allclose(steady, 8.0)
    perturbed = _run(lorenz_96(8), until=5.0, dt=0.5)['ring.x']
    assert np.abs(perturbed[-1] - 8.0).max() > 1.0


def test_geodynamo_is_symmetric_under_polarity_reversal():
    """Test that reversing both fields reverses the trajectory, and that the dipole reverses on its own."""
    north = _run(geodynamo(Q=0.1, D=0.2), until=50.0, dt=1.0)
    south = _run(geodynamo(Q=-0.1, D=-0.2), until=50.0, dt=1.0)
    assert np.allclose(north['dynamo.D'], -south['dynamo.D'], atol=1e-4)
    assert np.allclose(north['dynamo.V'], south['dynamo.V'], atol=1e-4)
    assert _sign_changes(_run(geodynamo(), until=300.0, dt=1.0)['dynamo.D']) > 2


def test_soil_temperature_reaches_its_fixed_point():
    """Test that the coupled soil maps settle on their nontrivial fixed point for a moderate logistic parameter."""
    trajectory = _run(soil_temperature(r=2.5, epsilon=0.2), until=200, dt=1)
    x = (2.5 + 0.2**2 / 0.8 - 1) / 2.5
    assert np.isclose(trajectory['soil.x'][-1], x)
    assert np.isclose(trajectory['soil.y'][-1], 0.2 * x / 0.8)


def test_stommel_is_bistable():
    """Test that the thermohaline boxes settle in a thermal or a haline circulation depending on their start."""
    thermal = _run(stommel(T=3.0, S=0.0), until=100.0, dt=1.0)
    haline = _run(stommel(T=0.0, S=4.0), until=100.0, dt=1.0)
    assert thermal['ocean.T'][-1] - thermal['ocean.S'][-1] > 0.3
    assert haline['ocean.T'][-1] - haline['ocean.S'][-1] < 0.0


def test_burridge_knopoff_sticks_and_slips():
    """Test that the blocks never slide backwards, slip in bursts, and follow the plate."""
    trajectory = _run(burridge_knopoff(velocity=0.05), until=40.0, dt=0.5)
    x, v = trajectory['fault.x'], trajectory['fault.v']
    assert np.all(np.diff(x, axis=0) > -1e-9)
    assert (v > 0.01).any()
    assert (np.abs(v) < 1e-6).mean() > 0.5
    assert x[-1].mean() > 0.05 * 40.0 - 2.0


def test_burridge_knopoff_default_solver_does_not_stall():
    """Test that the registered solver runs the fault through its slips without the stall of LSODA."""
    solver = describe('burridge_knopoff').defaults['solver']
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        trajectory = _run(burridge_knopoff(), until=400.0, dt=10.0, solver=solver)
    steps = np.diff(trajectory['fault.x'], axis=0)
    assert steps.min() > -1e-9
    assert (steps > 0.3).any()


def test_van_der_pol_ice_ages_cycle_every_hundred_thousand_years():
    """Test that the unforced ice-age oscillator settles on a relaxation cycle of about 100 thousand years."""
    trajectory = Simulation(van_der_pol_ice_ages(), dt=0.5).run(until=400.0).trajectory_
    climate = np.asarray(trajectory['climate.y'], dtype=float)
    times = np.asarray(trajectory.times)
    crossings = times[np.flatnonzero((climate[:-1] < 0) & (climate[1:] >= 0))]
    assert len(crossings) >= 3
    assert np.allclose(np.diff(crossings), 100.0, rtol=0.05)
