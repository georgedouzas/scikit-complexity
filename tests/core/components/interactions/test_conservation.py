"""Test the `_conservation` module of `skcomplex.core.components.interactions`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import ConservationInteraction
from skcomplex.core.components.laws import FluxLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec

CELLS = 64


class CellElement(Element):
    """Cell of a field with one state."""

    def __init__(self, label: str, *, u: Annotated[object, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.u = u


class BurgersElement(CellElement):
    """Cell of a field that follows the Burgers equation written by hand."""

    def _get_laws(self, s):
        """Return the Burgers flux."""
        return [FluxLaw(s.u, s.u**2 / 2)]


class WireConservationInteraction(ConservationInteraction):
    """Conservation law that binds its field as `wire`."""

    def __init__(self, label: str, *, wire: Annotated[str, RoleSpec()], **arguments: object) -> None:
        super().__init__(label, target=wire, **arguments)
        self.wire = wire

    def _get_role(self, role: str) -> str:
        """Return `wire` for the target."""
        return 'wire' if role == 'target' else role


def _run(element, *interactions):
    """Return the state of a periodic finite-volume field that starts from a sine wave."""
    spacing = 2 * np.pi / CELLS
    values = {'u': 1.0 + 0.5 * np.sin((np.arange(CELLS) + 0.5) * spacing)}
    space = LatticeSpace((CELLS,), spacing=spacing, boundary='periodic')
    field = FieldElement('line', element=element, space=space, values=values, discretization='finite-volume')
    return simulate(CompositeSystem(elements=[field], interactions=list(interactions)), until=1.0, dt=0.1).values(
        'line.u',
    )


def test_run_matches_the_flux_law_and_conserves_the_total():
    """Test that the Burgers flux gives the trajectory of its flux law and keeps the total of the field."""
    burgers = ConservationInteraction('burgers', target='line', state='u', flux=lambda s: s.target.u**2 / 2)
    values = _run(CellElement, burgers)
    np.testing.assert_allclose(values, _run(BurgersElement), atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), rtol=1e-12)


def test_run_adds_the_source():
    """Test that a constant source raises the total of the field at the rate of the source times the cells."""
    burgers = ConservationInteraction(
        'burgers',
        target='line',
        state='u',
        flux=lambda s: s.target.u**2 / 2,
        source=lambda s: 0.2,
    )
    values = _run(CellElement, burgers)
    assert values[-1].sum() - values[0].sum() == pytest.approx(0.2 * CELLS * 1.0, rel=1e-9)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'state': '1u'}, 'no name'),
        ({'flux': 'quadratic'}, 'function of the namespace'),
        ({'limiter': 'superbee'}, 'Unknown limiter'),
        ({'riemann': 'roe'}, 'Unknown numerical flux'),
    ],
)
def test_check_rejects_invalid_arguments(arguments, message):
    """Test that `check` rejects a bad state, function, limiter, or numerical flux."""
    with pytest.raises(ValidationError, match=message):
        ConservationInteraction('conservation', target='line', **arguments).prepare()


class WaterElement(Element):
    """Cell of a channel with a depth, a discharge, and a gravity that may vary across the cells."""

    def __init__(
        self,
        label: str,
        *,
        h: Annotated[object, StateSpec()] = 1.0,
        q: Annotated[object, StateSpec()] = 0.0,
        g: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.h = h
        self.q = q
        self.g = g


class FluxWaterElement(WaterElement):
    """Cell of a channel whose shallow-water fluxes are written by hand."""

    def _get_laws(self, s):
        """Return the fluxes of the depth and the discharge."""
        return [FluxLaw(s.h, s.q), FluxLaw(s.q, s.q**2 / s.h + s.g * s.h**2 / 2)]


def _run_water(element, gravity, *interactions):
    """Return the depth of a periodic channel that starts from a bump of water."""
    centers = (np.arange(CELLS) + 0.5) / CELLS
    values = {'h': 1.0 + 0.2 * np.exp(-(((centers - 0.5) / 0.1) ** 2)), 'q': np.zeros(CELLS), 'g': gravity}
    space = LatticeSpace((CELLS,), spacing=1 / CELLS, boundary='periodic')
    field = FieldElement('channel', element=element, space=space, values=values, discretization='finite-volume')
    return simulate(CompositeSystem(elements=[field], interactions=list(interactions)), until=0.2, dt=0.05).values(
        'channel.h',
    )


def _water_interactions():
    """Return the conservation of the depth and of the discharge as two interactions."""
    return (
        ConservationInteraction('mass', target='channel', state='h', flux=lambda s: s.target.q),
        ConservationInteraction(
            'momentum',
            target='channel',
            state='q',
            flux=lambda s: s.target.q**2 / s.target.h + s.target.g * s.target.h**2 / 2,
        ),
    )


def test_run_couples_the_states_of_two_interactions_into_one_system():
    """Test that two interactions on the states of one field give the trajectory of the element's own fluxes."""
    values = _run_water(WaterElement, 1.0, *_water_interactions())
    np.testing.assert_allclose(values, _run_water(FluxWaterElement, 1.0), atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), rtol=1e-12)


def test_run_reads_parameters_that_vary_across_the_cells():
    """Test that a gravity varying across the cells changes the motion and keeps the total of the depth."""
    gravity = np.linspace(0.5, 1.5, CELLS)
    values = _run_water(WaterElement, gravity, *_water_interactions())
    uniform = _run_water(WaterElement, 1.0, *_water_interactions())
    assert not np.allclose(values[-1], uniform[-1])
    np.testing.assert_allclose(values.sum(axis=1), values[0].sum(), rtol=1e-12)


def test_run_without_numerical_flux_takes_the_divergence_on_a_spectral_field():
    """Test that the Burgers flux without a numerical flux keeps the total of a spectral field."""
    spacing = 2 * np.pi / CELLS
    values = {'u': 0.5 * np.sin(np.arange(CELLS) * spacing)}
    space = LatticeSpace((CELLS,), spacing=spacing, boundary='periodic')
    field = FieldElement('line', element=CellElement, space=space, values=values, discretization='spectral')
    burgers = ConservationInteraction(
        'burgers',
        target='line',
        state='u',
        flux=lambda s: s.target.u**2 / 2,
        riemann=None,
    )
    simulation = simulate(CompositeSystem(elements=[field], interactions=[burgers]), until=1.0, dt=0.1)
    trajectory = simulation.values('line.u')
    assert trajectory[-1].sum() == pytest.approx(trajectory[0].sum(), abs=1e-9)
    assert not np.allclose(trajectory[-1], trajectory[0])


def test_get_laws_reads_a_renamed_target():
    """Test that a conservation law that binds its field as `wire` gives the trajectory of the plain one."""
    plain = ConservationInteraction(
        'burgers',
        target='line',
        state='u',
        flux=lambda s: s.target.u**2 / 2,
        source=lambda s: 0.1,
    )
    renamed = WireConservationInteraction(
        'burgers',
        wire='line',
        state='u',
        flux=lambda s: s.wire.u**2 / 2,
        source=lambda s: 0.1,
    )
    np.testing.assert_allclose(_run(CellElement, renamed), _run(CellElement, plain))
