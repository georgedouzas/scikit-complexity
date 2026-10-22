"""Test structure factors, correlation lengths, domain growth, pattern wavelengths, and number fluctuations."""

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from skcomplex.core.analysis import (
    coarse_grain,
    correlation_length,
    dispersion_relation,
    domain_growth,
    domain_size,
    fields_of,
    number_fluctuations,
    pattern_wavelength,
    structure_factor,
)
from skcomplex.core.base import AnalysisError
from skcomplex.core.components import Field
from skcomplex.core.runtime import Simulation
from skcomplex.core.space import LatticeSpace
from skcomplex.core.system import System
from skcomplex.models import create, run
from skcomplex.models.chemistry.pattern_formation import Brusselator


def _exact_correlation_length(temperature: float) -> float:
    """Return the exact correlation length of the square Ising lattice above its critical temperature."""
    coupling = 1 / temperature
    return 1 / (-math.log(math.tanh(coupling)) - 2 * coupling)


def _reaction_diffusion_frames(rng: np.random.Generator, frames: int, side: int) -> np.ndarray:
    """Return exact stationary samples of `u_t = 4 lap u - u + noise` on a periodic lattice."""
    lattice = 2 * np.pi * np.fft.fftfreq(side)
    hat = 4 * np.sin(lattice[:, None] / 2) ** 2 + 4 * np.sin(lattice[None, :] / 2) ** 2
    amplitude = 1 / np.sqrt(4 * hat + 1)
    noise = rng.standard_normal((frames, side, side))
    return np.fft.ifft2(np.fft.fft2(noise) * amplitude).real


def _cahn_hilliard(rng: np.random.Generator, side: int, steps: int, every: int) -> tuple[list[float], list[np.ndarray]]:
    """Return frames of `u_t = lap(u^3 - u - lap u)` from a symmetric mixture by a stabilized spectral scheme."""
    dt = 0.5
    u = 0.01 * rng.standard_normal((side, side))
    wavenumbers = 2 * np.pi * np.fft.fftfreq(side)
    squared = wavenumbers[:, None] ** 2 + wavenumbers[None, :] ** 2
    times, frames = [], []
    for step in range(1, steps + 1):
        nonlinear = np.fft.fft2(u**3 - u)
        spectrum = (np.fft.fft2(u) * (1 + 2 * dt * squared) - dt * squared * nonlinear) / (
            1 + dt * (squared**2 + 2 * squared)
        )
        u = np.fft.ifft2(spectrum).real
        if step % every == 0:
            times.append(step * dt)
            frames.append(u.copy())
    return times, frames


def _glauber_quench(rng: np.random.Generator, side: int, sweeps: int, every: int) -> tuple[list[int], list[np.ndarray]]:
    """Return frames of a periodic Ising lattice quenched from random spins to temperature 1 by heat-bath sweeps."""
    spins = rng.choice([-1.0, 1.0], (side, side))
    rows, columns = np.indices((side, side))
    masks = [(rows + columns) % 2 == parity for parity in (0, 1)]
    times, frames = [], []
    for sweep in range(1, sweeps + 1):
        for mask in masks:
            field = sum(np.roll(spins, shift, axis) for shift in (1, -1) for axis in (0, 1))
            up = rng.random(spins.shape) < 1 / (1 + np.exp(-2 * field))
            spins = np.where(mask, np.where(up, 1.0, -1.0), spins)
        if sweep % every == 0:
            times.append(sweep)
            frames.append(spins.copy())
    return times, frames


def _vicsek_densities(rng: np.random.Generator, agents: int, side: float, noise: float, steps: int) -> np.ndarray:
    """Return coarse-grained densities of Vicsek agents of speed 0.5 aligning within distance 1 in a periodic box."""
    positions = rng.random((agents, 2)) * side
    headings = rng.uniform(-np.pi, np.pi, agents)
    densities = []
    for step in range(steps):
        pairs = cKDTree(positions, boxsize=side).query_pairs(1.0, output_type='ndarray')
        sines, cosines = np.sin(headings), np.cos(headings)
        total_sines, total_cosines = sines.copy(), cosines.copy()
        for first, second in ((0, 1), (1, 0)):
            np.add.at(total_sines, pairs[:, first], sines[pairs[:, second]])
            np.add.at(total_cosines, pairs[:, first], cosines[pairs[:, second]])
        headings = np.arctan2(total_sines, total_cosines) + noise * rng.uniform(-np.pi, np.pi, agents)
        positions = np.mod(positions + 0.5 * np.column_stack([np.cos(headings), np.sin(headings)]), side)
        if step >= steps // 3 and step % 50 == 0:
            densities.append(coarse_grain(positions, [side, side], [64, 64]))
    return np.asarray(densities)


def test_correlation_length_ising_lattice_above_its_critical_temperature():
    """Test that 128 by 128 Ising lattices at temperatures 2.5 and 3 give the exact length within 15 percent."""
    for temperature in (2.5, 3.0):
        system = create('ising', size=128, temperature=temperature, random_state=0)
        frames = fields_of(Simulation(system, dt=1, random_state=0).run(until=1500).trajectory_)[500::5]
        length = correlation_length(frames, frames=True)
        assert length == pytest.approx(_exact_correlation_length(temperature), rel=0.15)


def test_correlation_length_reaction_diffusion_field():
    """Test that the stationary noisy linear reaction-diffusion field gives sqrt(D / k) = 2 within 10 percent."""
    frames = _reaction_diffusion_frames(np.random.default_rng(0), 50, 128)
    assert correlation_length(frames, frames=True) == pytest.approx(2, rel=0.1)
    assert 1 < correlation_length(frames, frames=True, method='fit') < 4
    with pytest.raises(AnalysisError, match='Unknown method'):
        correlation_length(frames[0], method='guess')


def test_domain_growth_two_coarsening_laws():
    """Test that a Glauber Ising quench grows as t^(1/2) within 0.1 and a Cahn-Hilliard mixture as t^(1/3)."""
    times, frames = _glauber_quench(np.random.default_rng(0), 256, 400, 10)
    ising = domain_growth(times, frames, fit_range=(20, 400))
    assert ising.exponent == pytest.approx(0.5, abs=0.1)
    times, frames = _cahn_hilliard(np.random.default_rng(1), 128, 4000, 200)
    mixture = domain_growth(times, frames, fit_range=(200, 2000))
    assert mixture.exponent == pytest.approx(1 / 3, abs=0.05)
    with pytest.raises(AnalysisError, match='fewer than 2 frames'):
        domain_growth([0.0], frames[:1])


def test_pattern_wavelength_turing_patterns_on_two_domains():
    """Test that Swift-Hohenberg rolls have wavelength 2 pi and a Brusselator its critical Turing wavelength."""
    rolls = fields_of(run('swift_hohenberg').trajectory_)[-1]
    assert pattern_wavelength(rolls, spacing=16 * math.pi / 64).wavelength == pytest.approx(2 * math.pi, rel=0.05)
    brusselator = Simulation(create('brusselator', b=1.95), dt=10.0).run(until=400.0).trajectory_
    spots = fields_of(brusselator)[-1]
    critical = 2 * math.pi / math.sqrt(1 / math.sqrt(1.0 * 8.0))
    assert pattern_wavelength(spots).wavelength == pytest.approx(critical, rel=0.1)


def test_number_fluctuations_giant_in_ordered_flocks():
    """Test that an ideal gas has normal number fluctuations and ordered Vicsek agents giant ones."""
    rng = np.random.default_rng(2)
    gas = np.asarray([coarse_grain(rng.random((4000, 2)) * 32, [32, 32], [64, 64]) for _ in range(40)])
    assert number_fluctuations(gas).exponent == pytest.approx(0.5, abs=0.05)
    assert number_fluctuations(_vicsek_densities(rng, 2000, 22.6, 0.3, 2000)).exponent > 0.6


def test_structure_factor_and_domain_size_of_stripes():
    """Test that stripes of period 16 give their peak and domain size by both methods."""
    stripes = np.sign(np.cos(2 * np.pi * np.arange(128) / 16))[:, None] * np.ones((128, 128))
    factor = structure_factor(stripes)
    assert 2 * np.pi / factor.wavenumbers[factor.power.argmax()] == pytest.approx(16)
    assert domain_size(stripes[:, 0], method='zero') == 4
    assert domain_size(stripes) == pytest.approx(16, rel=0.5)
    with pytest.raises(AnalysisError, match='Unknown method'):
        domain_size(stripes, method='peak')


def _brusselator(b, *, discretization='spectral', boundary='periodic'):
    """Return a one-dimensional Brusselator field at its uniform state `(1, b)`."""
    values = {'u': 1.0, 'v': b, 'a': 1.0, 'b': b, 'Du': 1.0, 'Dv': 8.0}
    space = LatticeSpace((128,), spacing=0.5, boundary=boundary)
    return System(components=[Field('cells', Brusselator, space, values=values, discretization=discretization)])


def test_dispersion_relation_finds_the_turing_band():
    """Test that the Brusselator above its Turing threshold grows on the analytic band within one grid wavenumber."""
    relation = dispersion_relation(_brusselator(1.9))
    gap = np.sqrt(6.2**2 - 32)
    low, high = np.sqrt((6.2 - gap) / 16), np.sqrt((6.2 + gap) / 16)
    step = 2 * np.pi / (128 * 0.5)
    assert relation.uniform == pytest.approx({'u': 1.0, 'v': 1.9})
    assert relation.band.min() == pytest.approx(low, abs=step)
    assert relation.band.max() == pytest.approx(high, abs=step)
    assert relation.growth_rates[relation.magnitudes == 0] < 0


def test_dispersion_relation_of_finite_differences_and_below_threshold():
    """Test that finite differences agree with spectral modes at long waves, and that no mode grows below threshold."""
    spectral = dispersion_relation(_brusselator(1.9))
    stencil = dispersion_relation(_brusselator(1.9, discretization='finite-difference'))
    long = spectral.magnitudes < 0.3
    assert stencil.growth_rates[long] == pytest.approx(spectral.growth_rates[long], abs=1e-2)
    assert dispersion_relation(_brusselator(1.7)).band.size == 0


def test_dispersion_relation_rejects_non_periodic_fields():
    """Test that a field with walls has no Fourier modes and raises an error."""
    with pytest.raises(AnalysisError, match='periodic lattice'):
        dispersion_relation(_brusselator(1.9, boundary='fixed'))
