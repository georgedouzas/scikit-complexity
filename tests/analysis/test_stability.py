"""Test the stability analysis: fixed points, Jacobians, stability, nullclines, divergence, and dispersion."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR

from skcomplex.analysis import StabilityAnalysis
from skcomplex.core.elements import Element, FieldElement, GroupElement
from skcomplex.core.interactions import DiffusionInteraction, Interaction, ReactionInteraction
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import total
from skcomplex.core.spaces import LatticeSpace
from skcomplex.core.system import System
from skcomplex.errors import AnalysisError
from skcomplex.models.biology.ecology import lotka_volterra
from skcomplex.models.biology.epidemiology import sir
from skcomplex.models.chemistry.common.kinetics import CubicAutocatalysisInteraction
from skcomplex.models.chemistry.pattern_formation import BrusselatorElement
from skcomplex.models.mathematics.dynamical_systems import logistic_map, lorenz
from skcomplex.models.physics.mechanics import harmonic_oscillator
from skcomplex.specs import RoleSpec, StateSpec, VisualSpec


def _stability(source, **settings):
    """Return the stability analysis of a source, run."""
    return StabilityAnalysis(**settings).check().prepare(source).run()


def _brusselator(b, *, discretization='spectral', boundary='periodic'):
    """Return a one-dimensional Brusselator field at its uniform state `(1, b)`."""
    values = {'u': 1.0, 'v': b, 'a': 1.0, 'b': b, 'Du': 1.0, 'Dv': 8.0}
    space = LatticeSpace((128,), spacing=0.5, boundary=boundary)
    cells = FieldElement('cells', element=BrusselatorElement, space=space, values=values, discretization=discretization)
    reactions = [
        ReactionInteraction('feed', target='cells', products={'u': 1}, modifiers={'a': 1}),
        ReactionInteraction('conversion', target='cells', reactants={'u': 1}, products={'v': 1}, modifiers={'b': 1}),
        CubicAutocatalysisInteraction('autocatalysis', target='cells', substrate='v', autocatalyst='u'),
        ReactionInteraction('decay', target='cells', reactants={'u': 1}),
    ]
    diffusions = [
        DiffusionInteraction('u_diffusion', target='cells', state='u', coefficient=lambda s: s.target.Du),
        DiffusionInteraction('v_diffusion', target='cells', state='v', coefficient=lambda s: s.target.Dv),
    ]
    return System(elements=[cells], interactions=[*reactions, *diffusions])


def test_lotka_volterra_centre():
    """Test that the predator-prey interior point is `(d / b, r / a)` with purely imaginary eigenvalues, a centre."""
    system = lotka_volterra()
    analysis = _stability(system)
    d, b, r, a = (SR.var(name) for name in ('d_death', 'b_predation', 'r_growth', 'a_predation'))
    index = next(index for index, point in enumerate(analysis.fixed_points_) if point['prey.n'] != 0)
    interior = analysis.fixed_points_[index]
    assert bool(interior['prey.n'] == d / b)
    assert bool(interior['predators.n'] == r / a)
    assert analysis.classifications_[index] == 'centre'
    assert abs(analysis.eigenvalues_[index].real).max() < 1e-9
    assert _stability(system, points=[{'prey.n': 0, 'predators.n': 0}]).classifications_ == ['saddle']


def test_logistic_map_fixed_point_is_stable():
    """Test that the logistic map's fixed point `1 - 1 / r` is stable at r = 2.5 and 0 is unstable."""
    analysis = _stability(logistic_map(r=2.5))
    classes = {
        round(float(point['map.x'].subs({SR.var('r_map'): 2.5})), 9): classification
        for point, classification in zip(analysis.fixed_points_, analysis.classifications_, strict=True)
    }
    assert classes == {0.6: 'stable', 0.0: 'unstable'}


def test_parameters_substitute_before_solving():
    """Test that parameter values given by path give numeric fixed points."""
    points = _stability(lorenz(), parameters={'lorenz.rho': 28.0, 'lorenz.beta': 8 / 3}).fixed_points_
    assert sorted(round(float(point['lorenz.z']), 6) for point in points) == [0.0, 27.0, 27.0]


def test_volume_preservation():
    """Test that the frictionless harmonic oscillator has divergence 0 and Lorenz contracts volumes."""
    assert _stability(harmonic_oscillator()).divergence_ == 0
    sigma, beta = SR.var('sigma_lorenz'), SR.var('beta_lorenz')
    assert bool(_stability(lorenz()).divergence_ == -sigma - beta - 1)


def test_jacobian_and_nullclines():
    """Test the Jacobian's shape and the nullclines of a two-state system."""
    analysis = _stability(lotka_volterra())
    assert analysis.jacobian_.dimensions() == (2, 2)
    assert set(analysis.nullclines_) == {'prey.n', 'predators.n'}


def test_jump_process_has_no_vector_field():
    """Test that stochastic jump systems are rejected."""
    with pytest.raises(AnalysisError, match='jump process'):
        _stability(sir(stochastic=True))


def _hawks_and_doves(V=2.0, C=3.0, size=2):
    """Return replicator dynamics of hawks and doves as the members of a group."""
    payoff = np.array([[(V - C) / 2, V], [0.0, V / 2]])

    class StrategyElement(Element):
        """Strategy element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.5,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

    class ReplicationInteraction(Interaction):
        """Replication interaction."""

        def __init__(
            self,
            label: str,
            *,
            population: Annotated[object, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.population = population

        def _get_laws(self, s):
            """Return the laws of the replication interaction."""
            return [
                RateLaw(
                    s.population.x,
                    s.population.x
                    * (
                        total(s.population.x, weights=payoff)
                        - total(s.population.x * total(s.population.x, weights=payoff))
                    ),
                ),
            ]

    return System(
        elements=[GroupElement('population', element=StrategyElement, size=size, values={'x': [1 / size] * size})],
        interactions=[ReplicationInteraction('replication', population='population')],
    )


def test_hawks_and_doves_as_group():
    """Test that hawks and doves, two members of a group, have the stable fixed point with `V / C` hawks."""
    analysis = _stability(_hawks_and_doves())
    index = next(index for index, point in enumerate(analysis.fixed_points_) if point['population.x[0]'] not in {0, 1})
    interior = analysis.fixed_points_[index]
    assert float(interior['population.x[0]']) == pytest.approx(2 / 3)
    assert float(interior['population.x[1]']) == pytest.approx(1 / 3)
    assert analysis.classifications_[index] == 'stable node'


def test_group_above_limit():
    """Test that the Jacobian of a group of 1000 members that reads reductions states the limit of 64 members."""

    class MemberElement(Element):
        """Member element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the member element."""
            return [RateLaw(s.x, total(s.x) - s.x)]

    system = System(elements=[GroupElement('group', element=MemberElement, size=1000)])
    with pytest.raises(AnalysisError, match=r'limit of 64 members.*smaller group'):
        _stability(system)


def test_points_are_the_given_points():
    """Test that given points replace the fixed points found by solving."""
    analysis = _stability(lotka_volterra(), points=[{'prey.n': 0, 'predators.n': 0}])
    assert analysis.fixed_points_ == [{'prey.n': 0, 'predators.n': 0}]
    assert len(analysis.eigenvalues_) == 1


def test_dispersion_relation_finds_the_turing_band():
    """Test that the Brusselator above its Turing threshold grows on the analytic band within one grid wavenumber."""
    relation = _stability(_brusselator(1.9), kind='dispersion')
    gap = np.sqrt(6.2**2 - 32)
    low, high = np.sqrt((6.2 - gap) / 16), np.sqrt((6.2 + gap) / 16)
    step = 2 * np.pi / (128 * 0.5)
    assert relation.uniform_ == pytest.approx({'u': 1.0, 'v': 1.9})
    assert relation.band_.min() == pytest.approx(low, abs=step)
    assert relation.band_.max() == pytest.approx(high, abs=step)
    assert relation.growth_rates_[relation.magnitudes_ == 0] < 0


def test_dispersion_relation_of_finite_differences_and_below_threshold():
    """Test that finite differences agree with spectral modes at long waves, and that no mode grows below threshold."""
    spectral = _stability(_brusselator(1.9), kind='dispersion')
    stencil = _stability(_brusselator(1.9, discretization='finite-difference'), kind='dispersion')
    long = spectral.magnitudes_ < 0.3
    assert stencil.growth_rates_[long] == pytest.approx(spectral.growth_rates_[long], abs=1e-2)
    assert _stability(_brusselator(1.7), kind='dispersion').band_.size == 0


def test_dispersion_relation_rejects_an_interaction_on_another_part():
    """Test that an interaction binding a part other than the field has no place in the dispersion relation."""
    system = _brusselator(1.9)
    system.interactions.append(
        DiffusionInteraction('elsewhere', target='other', state='u', coefficient=lambda s: s.target.Du),
    )
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
        StabilityAnalysis('unknown').check()
    with pytest.raises(AnalysisError, match='does not read points'):
        StabilityAnalysis('dispersion', points=[{'x': 0.0}]).check()
