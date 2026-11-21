"""Test the `_noise` module of `skcomplex.core.components.laws`."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.analysis import spectrum
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.solvers import EulerMaruyamaSolver, HeunSolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System


@pytest.mark.parametrize('process', ['market', ('market', 'rates'), None])
def test_noise_takes_a_label_labels_or_no_process(process):
    """Test that a noise law takes a label, a tuple of labels, or no process, and gives its amplitude."""
    n = var('n')
    law = NoiseLaw(n, 0.1 * n, process=process)
    assert law._get_value({'n': 2.0}) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'process': 3}, 'Give a label'),
        ({'process': ()}, 'Give a label'),
        ({'conserved': True, 'process': 'market'}, 'random steps on every link'),
        ({'convention': 'ito-stratonovich'}, 'Unknown noise convention'),
    ],
)
def test_noise_rejects_invalid_processes_and_conventions(arguments, message):
    """Test that a noise law rejects an invalid process, a process of conserved noise, and an unknown convention."""
    with pytest.raises(ValidationError, match=message):
        NoiseLaw(var('n'), 1.0, **arguments)._check()


@pytest.mark.parametrize('hurst', [0.3, 0.7, var('H')])
def test_noise_takes_a_hurst_index(hurst):
    """Test that a noise law takes a Hurst index in (0, 1) or a parameter."""
    law = NoiseLaw(var('n'), 1.0, hurst=hurst)
    assert law.hurst is hurst


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'hurst': 1.0}, r'Give a number in \(0, 1\)'),
        ({'hurst': 0}, r'Give a number in \(0, 1\)'),
        ({'hurst': 0.7, 'process': 'shared'}, r'n .*Remove the process'),
        ({'hurst': 0.7, 'conserved': True}, r'n .*a conserved noise draws independent steps'),
    ],
)
def test_noise_rejects_invalid_hurst_indices(arguments, message):
    """Test that a noise law rejects a Hurst index out of range, with a process, or on a conserved noise."""
    with pytest.raises(ValidationError, match=message):
        NoiseLaw(var('n'), 1.0, **arguments)._check()


def _shake(s):
    """Return the conserved noise of amplitude `a` of a density."""
    return [NoiseLaw(s.target.rho, s.target.a, conserved=True)]


def _diffuse(s):
    """Return the diffusion of a density at the rate `D` with conserved noise of temperature `T`."""
    rho = s.target.rho
    return [RateLaw(rho.diff(s.t), s.target.D * laplacian(rho)), NoiseLaw(rho, (2 * s.target.T) ** 0.5, conserved=True)]


def _density(space, laws, values, **settings):
    """Return a prepared system of a density on a lattice under the laws of one interaction."""
    noise = Interaction('noise', roles={'target': 'f'}, laws=laws)
    return System(elements=[Element('f', values=values, space=space)], interactions=[noise], **settings).prepare()


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
@pytest.mark.parametrize('solver', [EulerMaruyamaSolver, HeunSolver])
@pytest.mark.parametrize('boundary', ['periodic', 'zero-flux'])
def test_conserved_noise_keeps_total(backend, solver, boundary):
    """Test that a field with only conserved noise keeps its total within 1e-9 relative over 1000 steps."""
    values = np.random.default_rng(0).random((8, 6))
    space = LatticeSpace((8, 6), boundary={'rho': boundary})
    system = _density(space, _shake, {'rho': values, 'a': 0.5}, solver=solver(dt=0.01), backend=backend)
    trajectory = system.simulate(until=10, random_state=0, dt=1.0)
    sites = trajectory.get_values('f.rho').reshape(len(trajectory.times), -1)
    np.testing.assert_allclose(sites.sum(axis=1), values.sum(), rtol=1e-9)
    assert np.abs(sites[-1] - values.ravel()).min() > 0


def test_conserved_noise_has_site_variance_of_links():
    """Test that every site of a periodic lattice gains the variance `2 d (a / h)^2` per unit time."""
    space = LatticeSpace((16, 16), spacing=0.5)
    system = _density(space, _shake, {'rho': 0.0, 'a': 0.5}, solver=EulerMaruyamaSolver(dt=0.05))
    sites = system.simulate(until=4, random_state=0, dt=4.0).get_values('f.rho')[-1]
    assert sites.var() == pytest.approx(4 * (0.5 / 0.5) ** 2 * 4, rel=0.15)


@pytest.mark.slow
def test_conserved_noise_gives_flat_structure_factor():
    """Test that diffusion with conserved noise has the flat structure factor `T / D` and a fixed mean."""
    values = {'rho': 0.0, 'D': 1.0, 'T': 0.5}
    system = _density(LatticeSpace((16, 16)), _diffuse, values, solver=EulerMaruyamaSolver(dt=0.01))
    trajectory = system.simulate(until=1000, random_state=0, dt=1.0)
    frames = trajectory.get_values('f.rho')[50:].reshape(-1, 16, 16)
    factor = spectrum(frames, kind='structure-factor', frames=True)
    np.testing.assert_allclose(factor.power, 0.5, rtol=0.1)
    np.testing.assert_allclose(np.abs(np.fft.fftn(frames, axes=(1, 2))[:, 0, 0]) ** 2, 0.0, atol=1e-18)
