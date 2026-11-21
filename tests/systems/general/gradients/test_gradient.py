"""Test the `_gradient` module of `skcomplex.systems.general.gradients`."""

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import Expression, var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import gradient, laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.general.gradients import GradientInteraction

KAPPA = 0.5


class OrderRateInteraction(Interaction):
    """Rate of an order parameter `phi` written by hand as a function of it."""

    def __init__(self, label: str, *, target: Role, rate: Callable[[Expression], Expression]) -> None:
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _prepare(self, context=None):
        """Set the rate of the order parameter."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            phi = context.target.phi
            self.laws_ = [RateLaw(phi.diff(context.t), self.rate(phi))]


def _allen_cahn(phi):
    """Return the Allen-Cahn rate."""
    return phi - phi**3 + KAPPA * laplacian(phi)


def _cahn_hilliard(phi):
    """Return the Cahn-Hilliard rate."""
    return laplacian(phi**3) - laplacian(phi) - KAPPA * laplacian(laplacian(phi))


class PairWellInteraction(GradientInteraction):
    """Harmonic well between two members, which moves both."""

    def __init__(
        self,
        label: str,
        *,
        first: Role,
        second: Role,
        k: float = 1.0,
    ) -> None:
        super().__init__(label, target=first, inertia=True)
        self.first = first
        self.second = second
        self.k = k

    def _get_potential(self, s):
        """Return `k |x_first - x_second|**2 / 2`."""
        gap = s.first.x - s.second.x
        return s.k * gap.dot_product(gap) / 2


def _harmonic(label, k, *, inertia=False):
    """Return a harmonic gradient on the state `x` of a target, a gradient flow or with inertia Newton's law."""
    return GradientInteraction(
        f'{label}_well',
        target=label,
        potential=lambda s: k * s.target.x.dot_product(s.target.x) / 2,
        inertia=inertia,
    )


def _density(s):
    """Return the Ginzburg-Landau density with a gradient term."""
    gradients = gradient(s.target.phi)
    return s.target.phi**4 / 4 - s.target.phi**2 / 2 + KAPPA * gradients.dot_product(gradients) / 2


def _field(until, interaction):
    """Return the order parameter of a periodic field with a fixed random start after a time."""
    rng = np.random.default_rng(0)
    space = LatticeSpace((16, 16), spacing=1.0, boundary='periodic')
    field = Element('mixture', values={'phi': 0.1 * rng.uniform(-1.0, 1.0, (16, 16))}, space=space)
    system = System(elements=[field], interactions=[interaction]).prepare()
    return system.simulate(until=until, dt=0.05).get_values('mixture.phi')


def _run(system, until, dt):
    """Return the trajectory of a system solved to a tight tolerance."""
    system.set_params(solver=ScipyODESolver(rtol=1e-10, atol=1e-12))
    return system.prepare().simulate(until=until, dt=dt)


def test_simulate_gives_the_gradient_flow_and_the_newtonian_motion():
    """Test that a harmonic gradient decays a flowing state and oscillates a state under Newton's law."""
    system = System(
        elements=[
            Element('slow', values={'x': [1.0]}),
            Element('fast', values={'x': [1.0], 'm': 2.0}),
        ],
        interactions=[_harmonic('slow', 2.0), _harmonic('fast', 4.0, inertia=True)],
    )
    trajectory = _run(system, 2 * np.pi, 0.1)
    times = trajectory.times
    np.testing.assert_allclose(trajectory.get_values('slow.x')[:, 0], np.exp(-2.0 * times), atol=1e-8)
    np.testing.assert_allclose(trajectory.get_values('fast.x')[:, 0], np.cos(np.sqrt(2.0) * times), atol=1e-7)


def test_simulate_moves_every_role_of_a_subclass():
    """Test that a gradient of two members keeps their centre of mass and oscillates their gap."""
    system = System(
        elements=[Element('a', values={'x': [-1.0], 'm': 1.0}), Element('b', values={'x': [1.0], 'm': 1.0})],
        interactions=[PairWellInteraction('well', first='a', second='b', k=0.5)],
    )
    trajectory = _run(system, np.pi, 0.1)
    a, b = trajectory.get_values('a.x')[:, 0], trajectory.get_values('b.x')[:, 0]
    np.testing.assert_allclose(a + b, 0.0, atol=1e-9)
    np.testing.assert_allclose(b - a, 2.0 * np.cos(trajectory.times), atol=1e-6)


def test_prepare_gives_the_functional_derivative_of_a_density():
    """Test that the Laplacian and the gradient forms of a density give one functional derivative."""
    phi = var('phi')
    target = SimpleNamespace(phi=phi)
    by_gradient = GradientInteraction('a', target='f', states='phi', potential=_density, variational=True)
    by_laplacian = GradientInteraction(
        'b',
        target='f',
        states='phi',
        potential=lambda s: s.target.phi**4 / 4
        - s.target.phi**2 / 2
        - KAPPA * s.target.phi * laplacian(s.target.phi) / 2,
        variational=True,
    )
    rates = [item.prepare(SimpleNamespace(target=target)).laws_[0].value for item in (by_gradient, by_laplacian)]
    expected = phi - phi**3 + KAPPA * laplacian(phi)
    assert all(bool((rate - expected).expand() == 0) for rate in rates)


def test_simulate_matches_the_allen_cahn_equation():
    """Test that the variational gradient of the Ginzburg-Landau density gives the Allen-Cahn trajectory."""
    allen_cahn = GradientInteraction('ordering', target='mixture', states='phi', potential=_density, variational=True)
    by_hand = OrderRateInteraction('ordering', target='mixture', rate=_allen_cahn)
    np.testing.assert_allclose(_field(2.0, allen_cahn), _field(2.0, by_hand), atol=1e-10)


def test_simulate_matches_the_cahn_hilliard_equation_and_conserves_the_total():
    """Test that the conserved variational gradient gives the Cahn-Hilliard trajectory and keeps the total."""
    cahn_hilliard = GradientInteraction(
        'ordering',
        target='mixture',
        states='phi',
        potential=_density,
        variational=True,
        conserved=True,
    )
    values = _field(1.0, cahn_hilliard)
    by_hand = OrderRateInteraction('ordering', target='mixture', rate=_cahn_hilliard)
    np.testing.assert_allclose(values, _field(1.0, by_hand), atol=1e-8)
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), atol=1e-9)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'states': ('x', '2y')}, 'Give names of states'),
        ({'potential': 'quadratic'}, 'function of the namespace'),
    ],
)
def test_prepare_rejects_invalid_arguments(arguments, message):
    """Test that `prepare` rejects a state that is not a name, or a potential that is not callable."""
    with pytest.raises(ValidationError, match=message):
        GradientInteraction('gradient', target='walkers', **arguments).prepare()


def test_simulate_divides_every_member_by_its_own_mass():
    """Test that a well between two members of masses 1 and 3 keeps the momentum `x_a' + 3 x_b'` at 0."""
    system = System(
        elements=[Element('a', values={'x': [-1.0], 'm': 1.0}), Element('b', values={'x': [1.0], 'm': 3.0})],
        interactions=[PairWellInteraction('well', first='a', second='b')],
    )
    trajectory = system.prepare().simulate(until=2.0, dt=0.1)
    momentum = trajectory.get_values("a.x'")[:, 0] + 3.0 * trajectory.get_values("b.x'")[:, 0]
    np.testing.assert_allclose(momentum, 0.0, atol=1e-9)


def test_simulate_divides_by_the_mass_of_newton_law():
    """Test that a harmonic well over the mass `4` oscillates at the frequency `1 / 2`."""
    well = _harmonic('body', 1.0, inertia=True)
    system = System(elements=[Element('body', values={'x': [1.0], 'm': 4.0})], interactions=[well])
    trajectory = _run(system, np.pi, np.pi)
    assert float(trajectory.get_values('body.x')[-1][0]) == pytest.approx(0.0, abs=1e-6)


def test_inertia_without_a_mass_moves_the_second_derivative_by_the_force():
    """Test that a harmonic well with inertia on an element with no mass reverses it after half a period `pi`."""
    well = _harmonic('body', 1.0, inertia=True)
    system = System(elements=[Element('body', values={'x': [1.0]})], interactions=[well])
    trajectory = _run(system, np.pi, np.pi)
    assert float(trajectory.get_values('body.x')[-1][0]) == pytest.approx(-1.0, abs=1e-6)
