"""Test the `_michaelis_menten` module of `skcomplex.systems.biology.common.kinetics`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
from skcomplex.systems.biology.common.kinetics import MichaelisMentenInteraction


class CellElement(Element):
    """Cell with a substrate `S`, an enzyme `E`, and a total `T` of the substrate."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        S: Annotated[float, StateSpec()] = 1.0,
        E: Annotated[float, StateSpec()] = 1.0,
        T: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.S = S
        self.E = E
        self.T = T


def _run(interaction, until=1.0, dt=0.1, **values):
    """Return the times and the substrate of one cell under one reaction."""
    system = CompositeSystem(elements=[CellElement('cell', **values)], interactions=[interaction])
    trajectory = simulate(system, dt=dt, until=until)
    return np.asarray(trajectory.times), np.asarray(trajectory.values('cell.S'))


def test_michaelis_menten_saturates_at_the_maximal_rate():
    """Test that a substrate far above the Michaelis constant falls at the maximal rate."""
    uptake = MichaelisMentenInteraction('uptake', target='cell', reactants={'S': 1}, substrate='S', vmax=2.0, km=1.0)
    times, substrate = _run(uptake, S=1e6)
    np.testing.assert_allclose(substrate, 1e6 - 2.0 * times, rtol=1e-9)


def test_michaelis_menten_is_linear_below_the_michaelis_constant():
    """Test that a substrate far below the Michaelis constant decays at the rate `vmax / km`."""
    uptake = MichaelisMentenInteraction('uptake', target='cell', reactants={'S': 1}, substrate='S', vmax=2.0, km=1e4)
    times, substrate = _run(uptake, S=1e-3)
    np.testing.assert_allclose(substrate, 1e-3 * np.exp(-2e-4 * times), rtol=1e-6)


def test_michaelis_menten_scales_with_its_modifier():
    """Test that an enzyme modifier scales the propensity without changing."""
    uptake = MichaelisMentenInteraction(
        'uptake',
        target='cell',
        reactants={'S': 1},
        substrate='S',
        vmax=2.0,
        km=1.0,
        modifiers={'E': 1},
    )
    times, substrate = _run(uptake, S=1e6, E=3.0)
    np.testing.assert_allclose(substrate, 1e6 - 6.0 * times, rtol=1e-9)


def test_michaelis_menten_reads_a_substrate_function_and_the_target_parameters():
    """Test that a substrate function of the conserved total restores the state towards the total."""
    restoration = MichaelisMentenInteraction(
        'restoration',
        target='cell',
        products={'S': 1},
        substrate=lambda s: s.target.T - s.target.S,
        vmax=lambda s: s.target.T,
        km=0.1,
    )
    _, substrate = _run(restoration, until=20.0, S=0.0, T=2.0)
    assert 0.0 < substrate[1] < substrate[5] < 2.0
    assert substrate[-1] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'substrate': '1S'}, 'substrate'),
        ({'substrate': 3}, 'substrate'),
        ({'vmax': 'fast'}, 'vmax'),
        ({'km': None}, 'km'),
    ],
)
def test_michaelis_menten_rejects_invalid_arguments(arguments, message):
    """Test that the reaction rejects a substrate that is no name or function, and rates that are no numbers."""
    with pytest.raises(ValidationError, match=message):
        MichaelisMentenInteraction('uptake', target='cell', reactants={'S': 1}, **arguments).prepare()
