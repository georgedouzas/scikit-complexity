"""Test the `_gradient` module of `skcomplex.core.components.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import GradientInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import gradient, laplacian
from skcomplex.core.components.spaces import ContinuousSpace, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec

KAPPA = 0.5


class BodyElement(Element):
    """Element with a vector state of a given order and a positive divisor."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec(vector=True, order=2)] = 0.0,
        x_d1: object = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.x_d1 = x_d1
        self.m = m


class OverdampedElement(Element):
    """Element with a vector state of order 1 and a positive friction."""

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[object, StateSpec(vector=True)] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.gamma = gamma


class OrderElement(Element):
    """Cell of a field with an order parameter."""

    def __init__(self, label: str, *, phi: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.phi = phi


class AllenCahnElement(OrderElement):
    """Cell of a field that follows the Allen-Cahn equation written by hand."""

    def _get_laws(self, s):
        """Return the Allen-Cahn rate."""
        return [RateLaw(s.phi, s.phi - s.phi**3 + KAPPA * laplacian(s.phi))]


class CahnHilliardElement(OrderElement):
    """Cell of a field that follows the Cahn-Hilliard equation written by hand."""

    def _get_laws(self, s):
        """Return the Cahn-Hilliard rate."""
        return [RateLaw(s.phi, laplacian(s.phi**3) - laplacian(s.phi) - KAPPA * laplacian(laplacian(s.phi)))]


class PairWellInteraction(GradientInteraction):
    """Harmonic well between two members, which moves both."""

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, target=first, divisor='m')
        self.first = first
        self.second = second
        self.k = k

    def _get_potential(self, s):
        """Return `k |x_first - x_second|**2 / 2`."""
        gap = s.first.x - s.second.x
        return s.k * gap.dot_product(gap) / 2


def _harmonic(label, k, divisor):
    """Return a harmonic gradient on the state `x` of a target."""
    return GradientInteraction(
        f'{label}_well',
        target=label,
        potential=lambda s: k * s.target.x.dot_product(s.target.x) / 2,
        divisor=divisor,
    )


def _density(s):
    """Return the Ginzburg-Landau density with a gradient term."""
    gradients = gradient(s.target.phi)
    return s.target.phi**4 / 4 - s.target.phi**2 / 2 + KAPPA * gradients.dot_product(gradients) / 2


def _field(element, until, *interactions):
    """Return the order parameter of a periodic field with a fixed random start after a time."""
    rng = np.random.default_rng(0)
    space = LatticeSpace((16, 16), spacing=1.0, boundary='periodic')
    values = {'phi': 0.1 * rng.uniform(-1.0, 1.0, (16, 16))}
    field = FieldElement('mixture', element=element, space=space, values=values)
    return simulate(
        CompositeSystem(elements=[field], interactions=list(interactions)),
        until=until,
        dt=0.05,
    ).get_values(
        'mixture.phi',
    )


def test_run_gives_the_gradient_flow_and_the_newtonian_motion():
    """Test that a harmonic gradient decays a state of order 1 and oscillates a state of order 2 with its period."""
    system = CompositeSystem(
        elements=[OverdampedElement('slow', x=[1.0], gamma=2.0), BodyElement('fast', x=[1.0], m=2.0)],
        interactions=[_harmonic('slow', 4.0, 'gamma'), _harmonic('fast', 4.0, 'm')],
        space=ContinuousSpace(n_dim=1),
    )
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    trajectory = simulate(system, until=2 * np.pi, solver=solver, dt=0.1)
    times = trajectory.times
    np.testing.assert_allclose(trajectory.get_values('slow.x')[:, 0], np.exp(-2.0 * times), atol=1e-8)
    np.testing.assert_allclose(trajectory.get_values('fast.x')[:, 0], np.cos(np.sqrt(2.0) * times), atol=1e-7)


def test_run_moves_every_role_of_a_subclass():
    """Test that a gradient of two members keeps their centre of mass and oscillates their gap."""
    system = CompositeSystem(
        elements=[BodyElement('a', x=[-1.0]), BodyElement('b', x=[1.0])],
        interactions=[PairWellInteraction('well', first='a', second='b', k=0.5)],
        space=ContinuousSpace(n_dim=1),
    )
    solver = ScipyODESolver(rtol=1e-10, atol=1e-12)
    trajectory = simulate(system, until=np.pi, solver=solver, dt=0.1)
    a, b = trajectory.get_values('a.x')[:, 0], trajectory.get_values('b.x')[:, 0]
    np.testing.assert_allclose(a + b, 0.0, atol=1e-9)
    np.testing.assert_allclose(b - a, 2.0 * np.cos(trajectory.times), atol=1e-6)


def test_run_gives_the_functional_derivative_of_a_density():
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
    rates = [item.prepare().run(SimpleNamespace(target=target)).laws_[0].value for item in (by_gradient, by_laplacian)]
    expected = phi - phi**3 + KAPPA * laplacian(phi)
    assert all(bool((rate - expected).expand() == 0) for rate in rates)


def test_run_matches_the_allen_cahn_equation():
    """Test that the variational gradient of the Ginzburg-Landau density gives the Allen-Cahn trajectory."""
    allen_cahn = GradientInteraction('ordering', target='mixture', states='phi', potential=_density, variational=True)
    np.testing.assert_allclose(_field(OrderElement, 2.0, allen_cahn), _field(AllenCahnElement, 2.0), atol=1e-10)


def test_run_matches_the_cahn_hilliard_equation_and_conserves_the_total():
    """Test that the conserved variational gradient gives the Cahn-Hilliard trajectory and keeps the total."""
    cahn_hilliard = GradientInteraction(
        'ordering',
        target='mixture',
        states='phi',
        potential=_density,
        variational=True,
        conserved=True,
    )
    values = _field(OrderElement, 1.0, cahn_hilliard)
    np.testing.assert_allclose(values, _field(CahnHilliardElement, 1.0), atol=1e-8)
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), atol=1e-9)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'states': ('x', '2y')}, 'Give names of states'),
        ({'divisor': 'm-1'}, 'name of a state or parameter'),
        ({'potential': 'quadratic'}, 'function of the namespace'),
    ],
)
def test_check_rejects_invalid_arguments(arguments, message):
    """Test that `check` rejects a state or a divisor that is not a name, or a potential that is not callable."""
    with pytest.raises(ValidationError, match=message):
        GradientInteraction('gradient', target='walkers', **arguments).prepare()


class CoordinatesElement(Element):
    """Element with two scalar coordinates of order 2 and their own masses."""

    def __init__(
        self,
        label: str,
        *,
        a: Annotated[object, StateSpec(order=2)] = 1.0,
        a_d1: object = 0.0,
        b: Annotated[object, StateSpec(order=2)] = 0.0,
        b_d1: object = 0.0,
        ma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        mb: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label)
        self.a = a
        self.a_d1 = a_d1
        self.b = b
        self.b_d1 = b_d1
        self.ma = ma
        self.mb = mb


def test_run_divides_every_state_by_its_own_divisor():
    """Test that a spring between two coordinates with their own masses keeps the momentum `ma a' + mb b'` at 0."""
    spring = GradientInteraction(
        'spring',
        target='body',
        states=('a', 'b'),
        potential=lambda s: (s.target.a - s.target.b) ** 2 / 2,
        divisor={'a': 'ma', 'b': 'mb'},
    )
    system = CompositeSystem(elements=[CoordinatesElement('body')], interactions=[spring])
    trajectory = simulate(system, until=2.0, dt=0.1)
    momentum = trajectory.get_values('body.a_d1') + 3.0 * trajectory.get_values('body.b_d1')
    np.testing.assert_allclose(momentum, 0.0, atol=1e-9)


def test_run_divides_by_a_divisor_expression():
    """Test that a harmonic well divided by `4 m` oscillates at the frequency `1 / sqrt(4 m)`."""
    well = _harmonic('body', 1.0, lambda member: 4 * member.m)
    system = CompositeSystem(
        elements=[BodyElement('body', x=[1.0], m=1.0)],
        interactions=[well],
        space=ContinuousSpace(n_dim=1),
    )
    trajectory = simulate(system, until=np.pi, solver=ScipyODESolver(rtol=1e-10, atol=1e-12), dt=np.pi)
    assert float(trajectory.get_values('body.x')[-1][0]) == pytest.approx(0.0, abs=1e-6)


def test_check_rejects_a_divisor_of_an_unknown_state():
    """Test that a divisor mapping that names a state the interaction does not move raises an error."""
    gradient = GradientInteraction('well', target='body', states='a', potential=lambda s: 0, divisor={'b': 'mb'})
    with pytest.raises(ValidationError, match='divisor'):
        gradient.prepare()
