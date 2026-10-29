"""Test the `_field` module of `skcomplex.core.elements`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import skfem

from skcomplex.analysis import SpectrumAnalysis
from skcomplex.core.elements import Element, FieldElement
from skcomplex.core.laws import NoiseLaw, RateLaw
from skcomplex.core.laws.utils import laplacian
from skcomplex.core.spaces import LatticeSpace, MeshSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import EulerMaruyamaSolver, HeunSolver
from skcomplex.errors import ValidationError
from skcomplex.specs import ParameterSpec, StateSpec, VisualSpec


class ConservedElement(Element):
    """Conserved element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec()] = 0.0,
        a: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.a = a

    def _get_laws(self, s):
        """Return the laws of the conserved element."""
        return [NoiseLaw(s.rho, s.a, conserved=True)]


class DiffusingElement(Element):
    """Diffusing element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[object, StateSpec()] = 0.0,
        D: Annotated[object, ParameterSpec()] = 1.0,
        T: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.D = D
        self.T = T

    def _get_laws(self, s):
        """Return the laws of the diffusing element."""
        return [RateLaw(s.rho, s.D * laplacian(s.rho)), NoiseLaw(s.rho, (2 * s.T) ** 0.5, conserved=True)]


@pytest.mark.parametrize(
    ('field', 'message'),
    [
        (
            FieldElement('f', element=ConservedElement, space=LatticeSpace(4), discretization='galerkin'),
            'Unknown discretization',
        ),
        (
            FieldElement('f', element=ConservedElement, space=LatticeSpace(4), boundary={'rho': 'open'}),
            "Unknown boundary 'open' of f.rho",
        ),
    ],
)
def test_field_checks_its_discretization_and_boundaries(field, message):
    """Test that an unknown discretization or boundary condition is rejected by the check."""
    with pytest.raises(ValidationError, match=message):
        field.check()


def test_field_takes_the_boundaries_of_its_space():
    """Test that a state without a boundary takes the edges of its space, and a fixed boundary its value."""
    field = FieldElement('f', element=ConservedElement, space=LatticeSpace(4, boundary='fixed', fill=2.0))
    assert field._get_boundary('rho') == ('fixed', 2.0)
    field = FieldElement('f', element=ConservedElement, space=LatticeSpace(4, boundary='reflecting'))
    assert field._get_boundary('rho') == ('zero-flux', 0.0)
    field = FieldElement('f', element=ConservedElement, space=LatticeSpace(4), boundary={'rho': ('fixed', 1)})
    assert field._get_boundary('rho') == ('fixed', 1.0)


def test_field_draws_an_image_on_a_lattice_and_a_mesh_on_a_mesh():
    """Test that the default visual of a field depends on its space."""
    grid = skfem.MeshTri.init_tensor(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    spaces = {'image': LatticeSpace((3, 3)), 'mesh': MeshSpace(grid.p, grid.t), 'sites': NetworkSpace(nx.path_graph(3))}
    for kind, space in spaces.items():
        field = FieldElement('f', element=ConservedElement, space=space.check().prepare()).check().prepare()
        assert field.default_visual_.kind == kind


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
@pytest.mark.parametrize('stepper', [EulerMaruyamaSolver, HeunSolver])
@pytest.mark.parametrize('boundary', ['periodic', 'zero-flux'])
def test_conserved_noise_keeps_total(backend, stepper, boundary):
    """Test that a field with only conserved noise keeps its total within 1e-9 relative over 1000 steps."""
    values = np.random.default_rng(0).random((8, 6))
    field = FieldElement(
        'f',
        element=ConservedElement,
        space=LatticeSpace((8, 6)),
        values={'rho': values},
        boundary={'rho': boundary},
    )
    trajectory = (
        Simulation(System(elements=[field]), backend=backend, solver=stepper(dt=0.01), random_state=0, dt=1.0)
        .check()
        .run(until=10)
        .trajectory_
    )
    sites = trajectory.values('f.rho').reshape(len(trajectory.times), -1)
    np.testing.assert_allclose(sites.sum(axis=1), values.sum(), rtol=1e-9)
    assert np.abs(sites[-1] - values.ravel()).min() > 0


def test_conserved_noise_has_site_variance_of_links():
    """Test that every site of a periodic lattice gains the variance `2 d (a / h)^2` per unit time."""
    field = FieldElement('f', element=ConservedElement, space=LatticeSpace((16, 16), spacing=0.5))
    trajectory = Simulation(
        System(elements=[field]),
        solver=EulerMaruyamaSolver(dt=0.05),
        random_state=0,
        dt=4.0,
    ).check()
    sites = trajectory.run(until=4).trajectory_.values('f.rho')[-1]
    assert sites.var() == pytest.approx(4 * (0.5 / 0.5) ** 2 * 4, rel=0.15)


@pytest.mark.slow
def test_conserved_noise_gives_flat_structure_factor():
    """Test that diffusion with conserved noise has the flat structure factor `T / D` and a fixed mean."""
    field = FieldElement('f', element=DiffusingElement, space=LatticeSpace((16, 16)))
    trajectory = (
        Simulation(System(elements=[field]), solver=EulerMaruyamaSolver(dt=0.01), random_state=0, dt=1.0)
        .check()
        .run(until=1000)
        .trajectory_
    )
    frames = trajectory.values('f.rho')[50:].reshape(-1, 16, 16)
    factor = SpectrumAnalysis('structure-factor', frames=True).check().prepare(frames).run()
    np.testing.assert_allclose(factor.power_, 0.5, rtol=0.1)
    np.testing.assert_allclose(np.abs(np.fft.fftn(frames, axes=(1, 2))[:, 0, 0]) ** 2, 0.0, atol=1e-18)
