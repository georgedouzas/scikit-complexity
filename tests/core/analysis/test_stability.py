"""Test the stability analysis: fixed points, Jacobians, stability, nullclines, divergence, and dispersion."""

from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR

from skcomplex.core.analysis import stability
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian, total
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import AnalysisError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.biology.ecology import LotkaVolterraSystem
from skcomplex.systems.biology.epidemiology import SIRSystem
from skcomplex.systems.chemistry.common.kinetics import CubicAutocatalysisInteraction
from skcomplex.systems.chemistry.pattern_formation import BrusselatorElement
from skcomplex.systems.mathematics.dynamical_systems import LogisticMapSystem, LorenzSystem
from skcomplex.systems.physics.mechanics import HarmonicOscillatorSystem


def _stability(source, **settings):
    """Return the stability of a source."""
    return stability(source, **settings)


def _rate(label, state, rate, target='cells'):
    """Return an interaction that adds a rate, a function of the namespace, to a state of its target."""
    return Interaction(
        label,
        roles={'target': target},
        laws=lambda s: [RateLaw(getattr(s.target, state).diff(s.t), rate(s))],
    )


def _brusselator(b, *, discretization='spectral', boundary='periodic'):
    """Return a one-dimensional Brusselator field at its uniform state `(1, b)`."""
    space = LatticeSpace((128,), spacing=0.5, boundary=boundary, discretization=discretization)
    cells = BrusselatorElement('cells', u=1.0, v=b, a=1.0, b=b, Du=1.0, Dv=8.0, space=space)
    reactions = [
        _rate('feed', 'u', lambda s: s.target.a),
        _rate('loss', 'u', lambda s: -s.target.b * s.target.u),
        _rate('gain', 'v', lambda s: s.target.b * s.target.u),
        CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='v', autocatalyst='u'),
        _rate('decay', 'u', lambda s: -s.target.u),
    ]
    diffusions = [
        _rate('u_diffusion', 'u', lambda s: s.target.Du * laplacian(s.target.u)),
        _rate('v_diffusion', 'v', lambda s: s.target.Dv * laplacian(s.target.v)),
    ]
    return System(elements=[cells], interactions=[*reactions, *diffusions])


def test_lotka_volterra_centre():
    """Test that the predator-prey interior point is `(d / b, r / a)` with purely imaginary eigenvalues, a centre."""
    system = LotkaVolterraSystem().prepare().system_
    analysis = _stability(system)
    d, b, r, a = (SR.var(name) for name in ('d_death', 'b_predation', 'r_growth', 'a_predation'))
    index = next(index for index, point in enumerate(analysis.fixed_points) if point['prey.x'] != 0)
    interior = analysis.fixed_points[index]
    assert bool(interior['prey.x'] == d / b)
    assert bool(interior['predators.x'] == r / a)
    assert analysis.classifications[index] == 'centre'
    assert abs(analysis.eigenvalues[index].real).max() < 1e-9
    assert _stability(system, points=[{'prey.x': 0, 'predators.x': 0}]).classifications == ['saddle']


def test_logistic_map_fixed_point_is_stable():
    """Test that the logistic map's fixed point `1 - 1 / r` is stable at r = 2.5 and 0 is unstable."""
    analysis = _stability(LogisticMapSystem(r=2.5))
    classes = {
        round(float(point['map.x'].subs({SR.var('r_map'): 2.5})), 9): classification
        for point, classification in zip(analysis.fixed_points, analysis.classifications, strict=True)
    }
    assert classes == {0.6: 'stable', 0.0: 'unstable'}


def test_parameters_substitute_before_solving():
    """Test that parameter values given by path give numeric fixed points."""
    points = _stability(LorenzSystem(), parameters={'lorenz.rho': 28.0, 'lorenz.beta': 8 / 3}).fixed_points
    assert sorted(round(float(point['lorenz.z']), 6) for point in points) == [0.0, 27.0, 27.0]


def test_volume_preservation():
    """Test that the frictionless harmonic oscillator has divergence 0 and Lorenz contracts volumes."""
    assert _stability(HarmonicOscillatorSystem().prepare().system_).divergence == 0
    sigma, beta = SR.var('sigma_lorenz'), SR.var('beta_lorenz')
    assert bool(_stability(LorenzSystem()).divergence == -sigma - beta - 1)


def test_jacobian_and_nullclines():
    """Test the Jacobian's shape and the nullclines of a two-state system."""
    analysis = _stability(LotkaVolterraSystem().prepare().system_)
    assert analysis.jacobian.dimensions() == (2, 2)
    assert set(analysis.nullclines) == {'prey.x', 'predators.x'}


def test_jump_process_has_no_vector_field():
    """Test that stochastic jump systems are rejected."""
    with pytest.raises(AnalysisError, match='jump process'):
        _stability(SIRSystem(stochastic=True).prepare().system_)


def _hawks_and_doves(V=2.0, C=3.0, size=2):
    """Return replicator dynamics of hawks and doves as the members of a group."""
    payoff = np.array([[(V - C) / 2, V], [0.0, V / 2]])

    def replication(s):
        """Return the replicator dynamics of the members."""
        x = s.target.x
        return [RateLaw(x.diff(s.t), x * (total(x, weights=payoff) - total(x * total(x, weights=payoff))))]

    return System(
        elements=[Element('population', values={'x': [1 / size] * size}, size=size)],
        interactions=[Interaction('replication', roles={'target': 'population'}, laws=replication)],
    )


def test_hawks_and_doves_as_group():
    """Test that hawks and doves, two members of a group, have the stable fixed point with `V / C` hawks."""
    analysis = _stability(_hawks_and_doves())
    index = next(index for index, point in enumerate(analysis.fixed_points) if point['population.x[0]'] not in {0, 1})
    interior = analysis.fixed_points[index]
    assert float(interior['population.x[0]']) == pytest.approx(2 / 3)
    assert float(interior['population.x[1]']) == pytest.approx(1 / 3)
    assert analysis.classifications[index] == 'stable node'


def test_group_above_limit():
    """Test that the Jacobian of a group of 1000 members that reads reductions states the limit of 64 members."""

    group = Element('group', values={'x': 0.0}, size=1000)
    system = System(
        elements=[group],
        interactions=[_rate('pull', 'x', lambda s: total(s.target.x) - s.target.x, 'group')],
    )
    with pytest.raises(AnalysisError, match=r'limit of 64 members.*smaller group'):
        _stability(system)


def test_points_are_the_given_points():
    """Test that given points replace the fixed points found by solving."""
    analysis = _stability(LotkaVolterraSystem().prepare().system_, points=[{'prey.x': 0, 'predators.x': 0}])
    assert analysis.fixed_points == [{'prey.x': 0, 'predators.x': 0}]
    assert len(analysis.eigenvalues) == 1


def test_dispersion_relation_finds_the_turing_band():
    """Test that the Brusselator above its Turing threshold grows on the analytic band within one grid wavenumber."""
    relation = _stability(_brusselator(1.9), kind='dispersion')
    gap = np.sqrt(6.2**2 - 32)
    low, high = np.sqrt((6.2 - gap) / 16), np.sqrt((6.2 + gap) / 16)
    step = 2 * np.pi / (128 * 0.5)
    assert relation.uniform == pytest.approx({'u': 1.0, 'v': 1.9})
    assert relation.band.min() == pytest.approx(low, abs=step)
    assert relation.band.max() == pytest.approx(high, abs=step)
    assert relation.growth_rates[relation.magnitudes == 0] < 0


def test_dispersion_relation_of_finite_differences_and_below_threshold():
    """Test that finite differences agree with spectral modes at long waves, and that no mode grows below threshold."""
    spectral = _stability(_brusselator(1.9), kind='dispersion')
    stencil = _stability(_brusselator(1.9, discretization='finite-difference'), kind='dispersion')
    long = spectral.magnitudes < 0.3
    assert stencil.growth_rates[long] == pytest.approx(spectral.growth_rates[long], abs=1e-2)
    assert _stability(_brusselator(1.7), kind='dispersion').band.size == 0


def test_dispersion_relation_rejects_an_interaction_on_another_component():
    """Test that an interaction binding a component other than the field has no place in the dispersion relation."""
    system = _brusselator(1.9)
    system.interactions.append(_rate('elsewhere', 'u', lambda s: s.target.Du * laplacian(s.target.u), 'other'))
    with pytest.raises(AnalysisError, match='other than the field'):
        _stability(system, kind='dispersion')


def test_dispersion_relation_rejects_non_periodic_fields_and_data():
    """Test that a field with walls has no Fourier modes, and that an array is not a system."""
    with pytest.raises(AnalysisError, match='periodic lattice'):
        _stability(_brusselator(1.9, boundary='fixed'), kind='dispersion')
    with pytest.raises(AnalysisError, match='from a system'):
        _stability(np.zeros(4), kind='dispersion')


def test_stability_checks_its_kind_and_settings():
    """Test that an unknown kind and a setting the kind ignores raise errors."""
    with pytest.raises(AnalysisError, match='fixed-points, dispersion'):
        stability(LotkaVolterraSystem().prepare().system_, kind='unknown')
    with pytest.raises(AnalysisError, match='does not read points'):
        stability(LotkaVolterraSystem().prepare().system_, kind='dispersion', points=[{'x': 0.0}])


class _Fisher(Interaction):
    """Interaction of Fisher's equation, `u' = r u (1 - u) + D laplacian(u)`."""

    def __init__(self, label, *, target: Role = 'plate', D=1.0, r=0.5):
        super().__init__(label)
        self.target = target
        self.D = D
        self.r = r

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            u, t = context.target.u, context.t
            self.laws_ = [RateLaw(u.diff(t), context.r * u * (1 - u) + context.D * laplacian(u))]


def test_dispersion_relation_of_a_system_of_a_plain_element():
    """Test that the modes of Fisher's equation around `u = 1` decay at `r + 2 - 2 cos(k)` on a unit lattice."""
    plate = Element('plate', values={'u': np.full(16, 0.9)}, space=LatticeSpace(16))
    relation = stability(System(elements=[plate], interactions=[_Fisher('fisher')]), kind='dispersion')
    assert relation.uniform == pytest.approx({'u': 1.0})
    assert relation.growth_rates == pytest.approx(-0.5 - (2 - 2 * np.cos(relation.magnitudes)))
