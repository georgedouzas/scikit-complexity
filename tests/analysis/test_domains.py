"""Test domain sizes, domain growth, pattern wavelengths, and interface densities."""

import math

import networkx as nx
import numpy as np
import pytest

from skcomplex.analysis import DomainAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import AnalysisError
from skcomplex.models import create, run


def _domains(kind: str, source: object, **settings: object) -> DomainAnalysis:
    """Return the domain analysis of a source for one kind, run."""
    return DomainAnalysis(kind, **settings).check().prepare(source).run()


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


def test_domain_growth_two_coarsening_laws():
    """Test that a Glauber Ising quench grows as t^(1/2) within 0.1 and a Cahn-Hilliard mixture as t^(1/3)."""
    times, frames = _glauber_quench(np.random.default_rng(0), 256, 400, 10)
    ising = _domains('growth', frames, times=times, fit_range=(20, 400))
    assert ising.exponent_ == pytest.approx(0.5, abs=0.1)
    assert ising.size_.shape == (len(frames),)
    times, frames = _cahn_hilliard(np.random.default_rng(1), 128, 4000, 200)
    mixture = _domains('growth', frames, times=times, fit_range=(200, 2000))
    assert mixture.exponent_ == pytest.approx(1 / 3, abs=0.05)
    with pytest.raises(AnalysisError, match='fewer than 2 frames'):
        _domains('growth', frames[:1], times=[0.0])


def test_domain_growth_of_an_ising_trajectory():
    """Test that a quenched Ising trajectory gives one domain size per frame that grows."""
    trajectory = (
        Simulation(create('ising', size=64, temperature=1.0), dt=1, random_state=0)
        .check()
        .prepare()
        .run(until=100)
        .trajectory_
    )
    growth = _domains('growth', trajectory, fit_range=(10, 100))
    assert growth.size_.shape == (len(trajectory.times),)
    assert growth.exponent_ > 0


@pytest.mark.slow
def test_pattern_wavelength_turing_patterns_on_two_domains():
    """Test that Swift-Hohenberg rolls have wavelength 2 pi and a Brusselator its critical Turing wavelength."""
    rolls = run('swift_hohenberg').trajectory_.lattice('cells.u')[-1]
    wavelength = _domains('wavelength', rolls, spacing=16 * math.pi / 64).wavelength_[0]
    assert wavelength == pytest.approx(2 * math.pi, rel=0.05)
    brusselator = Simulation(create('brusselator', b=1.95), dt=10.0).check().prepare().run(until=400.0).trajectory_
    spots = brusselator.lattice('cells.u')[-1]
    critical = 2 * math.pi / math.sqrt(1 / math.sqrt(1.0 * 8.0))
    assert _domains('wavelength', spots).wavelength_[0] == pytest.approx(critical, rel=0.1)


def test_pattern_wavelength_and_orientation_of_stripes():
    """Test that stripes of period 8 give wavelength 8 and full orientation order."""
    stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
    pattern = _domains('wavelength', stripes)
    assert pattern.wavelength_.tolist() == [8.0]
    assert pattern.orientation_[0] == pytest.approx(1.0)


def test_domain_size_of_stripes():
    """Test that stripes of period 16 give their domain size by both methods."""
    stripes = np.sign(np.cos(2 * np.pi * np.arange(128) / 16))[:, None] * np.ones((128, 128))
    assert _domains('size', stripes[:, 0], method='zero').size_[0] == 4
    assert _domains('size', stripes).size_[0] == pytest.approx(16, rel=0.5)
    with pytest.raises(AnalysisError, match='moment, zero'):
        DomainAnalysis(method='peak').check()


def test_interface_density_on_grids_and_networks():
    """Test interface densities of a uniform and a random lattice and along network edges."""
    assert _domains('interface-density', np.ones((8, 8))).density_[0] == 0
    random = np.random.default_rng(1).choice([-1, 1], (128, 128))
    assert _domains('interface-density', random).density_[0] == pytest.approx(0.5, abs=0.01)
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, {0: 0, 1: 0, 2: 1, 3: 1}, 'state')
    assert _domains('interface-density', graph).density_[0] == pytest.approx(1 / 3)


def test_domain_analysis_checks_its_settings():
    """Test the errors of an unknown kind, an ignored setting, and a graph for a field kind."""
    with pytest.raises(AnalysisError, match='Use one of size'):
        DomainAnalysis('texture').check()
    with pytest.raises(AnalysisError, match='does not read fit_range'):
        DomainAnalysis('size', fit_range=(1, 2)).check()
    with pytest.raises(AnalysisError, match='reads a field'):
        _domains('size', nx.path_graph(3))
