"""Test the `_reaction_diffusion` module of `skcomplex.systems.general.dynamics`."""

from collections.abc import Mapping

import numpy as np
import pytest

from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.general.dynamics import ReactionDiffusionSystem


class DecaySystem(ReactionDiffusionSystem):
    """A field that decays through a reaction it returns as an interaction, with its own boundary."""

    def _get_boundaries(self) -> Mapping[str, object]:
        return {'u': 'zero-flux'}

    def _get_equation_interactions(self) -> list[RateInteraction]:
        return [RateInteraction('decay', target='medium', state='u', rate=lambda s: -s.target.u)]


def front_position(values: np.ndarray, spacing: float) -> float:
    """Return the position where the front crosses one half."""
    return float(np.argmax(values < 0.5)) * spacing


def test_reaction_diffusion_front_moves_at_the_fisher_speed():
    """Test that the Fisher-Kolmogorov front moves at a speed within 5 percent of `2 sqrt(r D)`."""
    spacing, r, D = 0.25, 1.0, 1.0
    initial = np.zeros(800)
    initial[:20] = 1.0
    system = ReactionDiffusionSystem(
        states={'u': initial},
        equations={'u': lambda s: s.target.r * s.target.u * (1 - s.target.u)},
        diffusivities={'u': D},
        shape=800,
        spacing=spacing,
        boundary='reflecting',
        parameters={'r': r},
    )
    trajectory = simulate(system, until=80.0, dt=20.0)
    values = trajectory.get_values('medium.u')
    speed = (front_position(values[4], spacing) - front_position(values[2], spacing)) / 40.0
    assert speed == pytest.approx(2 * np.sqrt(r * D), rel=0.05)


def test_reaction_diffusion_field_without_diffusivity_changes_site_by_site():
    """Test that a field with kinetics and no diffusivity decays on every site on its own."""
    system = ReactionDiffusionSystem(
        states={'u': np.array([1.0, 0.0, 2.0, 0.0]), 'w': 0.0},
        equations={'u': lambda s: -s.target.u},
        diffusivities={'w': 1.0},
        shape=4,
    )
    values = simulate(system, until=1.0, dt=1.0).get_values('medium.u')
    np.testing.assert_allclose(values[-1], np.array([1.0, 0.0, 2.0, 0.0]) * np.exp(-1.0), rtol=1e-5)


def test_reaction_diffusion_takes_boundaries_per_field_and_its_own_kinetics():
    """Test that a subclass gives the boundary of a field and its kinetics as interactions."""
    system = DecaySystem(states={'u': 1.0}, diffusivities={'u': 1.0}, shape=8)
    field = system.prepare().system_.elements[0]
    assert field.boundary == {'u': 'zero-flux'}
    assert [interaction.label for interaction in system.system_.interactions] == ['decay', 'u_diffusion']
    values = simulate(system, until=1.0, dt=1.0).get_values('medium.u')
    np.testing.assert_allclose(values[-1], np.exp(-1.0), rtol=1e-5)


def test_reaction_diffusion_spectral_diffusion_damps_a_mode():
    """Test that spectral diffusion damps a sine mode of a periodic line at the rate `D k^2`."""
    size = 32
    k = 2 * np.pi / size
    initial = np.sin(k * np.arange(size))
    system = ReactionDiffusionSystem(
        states={'u': initial},
        diffusivities={'u': 2.0},
        shape=size,
        discretization='spectral',
    )
    values = simulate(system, until=10.0, dt=10.0).get_values('medium.u')
    np.testing.assert_allclose(values[-1], initial * np.exp(-2.0 * k**2 * 10.0), atol=1e-4)


def test_reaction_diffusion_noise_is_stochastic():
    """Test that a noise amplitude makes the system stochastic."""
    system = ReactionDiffusionSystem(
        states={'u': 0.0},
        equations={'u': lambda s: -s.target.u},
        diffusivities={'u': 1.0},
        noises={'u': 0.1},
        shape=8,
    )
    assert system.get_tags().randomness == 'stochastic'


def test_reaction_diffusion_rejects_the_diffusivity_of_an_unknown_field():
    """Test that a diffusivity of an undeclared field names it."""
    system = ReactionDiffusionSystem(states={'u': 0.0}, diffusivities={'v': 1.0}, shape=4)
    with pytest.raises(ValidationError, match="'v'"):
        system.prepare()
