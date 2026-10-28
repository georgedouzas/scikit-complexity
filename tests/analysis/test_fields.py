"""Test structure factors, correlation lengths, and dispersion relations."""

import math

import numpy as np
import pytest

from skcomplex.analysis import (
    correlation_length,
    dispersion_relation,
    fields_of,
    structure_factor,
)
from skcomplex.core.elements import FieldElement
from skcomplex.core.errors import AnalysisError
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models import create
from skcomplex.models.chemistry.pattern_formation import BrusselatorElement


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
    assert 1 < correlation_length(frames, frames=True, method='prepare') < 4
    with pytest.raises(AnalysisError, match='Unknown method'):
        correlation_length(frames[0], method='guess')


def test_structure_factor_of_stripes():
    """Test that stripes of period 16 give their peak."""
    stripes = np.sign(np.cos(2 * np.pi * np.arange(128) / 16))[:, None] * np.ones((128, 128))
    factor = structure_factor(stripes)
    assert 2 * np.pi / factor.wavenumbers[factor.power.argmax()] == pytest.approx(16)


def _brusselator(b, *, discretization='spectral', boundary='periodic'):
    """Return a one-dimensional Brusselator field at its uniform state `(1, b)`."""
    values = {'u': 1.0, 'v': b, 'a': 1.0, 'b': b, 'Du': 1.0, 'Dv': 8.0}
    space = LatticeSpace((128,), spacing=0.5, boundary=boundary)
    cells = FieldElement('cells', element=BrusselatorElement, space=space, values=values, discretization=discretization)
    return System(elements=[cells])


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
