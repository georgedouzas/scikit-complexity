"""Test the `_fitzhugh_nagumo` module of `skcomplex.systems.biology.common.excitable`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.biology.common.excitable import FitzHughNagumoInteraction


class MembraneElement(Element):
    """Membrane with a potential `u` and a recovery `r`."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = -1.2,
        r: Annotated[float, StateSpec()] = -0.6,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.r = r


def _run(current: float) -> np.ndarray:
    """Return the potential of a membrane under a current."""
    kinetics = FitzHughNagumoInteraction('kinetics', target='membrane', fast='u', slow='r', I=current)
    system = CompositeSystem(elements=[MembraneElement('membrane')], interactions=[kinetics])
    trajectory = simulate(system, dt=0.1, until=200.0)
    return trajectory.get_values('membrane.u')[trajectory.times > 100.0]


def test_fitzhugh_nagumo_interaction_fires_under_current_and_rests_without():
    """Test that a current between the Hopf points makes the membrane oscillate and no current lets it rest."""
    firing, resting = _run(0.5), _run(0.0)
    assert firing.max() - firing.min() > 3.0
    assert resting.max() - resting.min() < 1e-3


def test_fitzhugh_nagumo_interaction_rejects_one_state_for_both_variables():
    """Test that the fast and the slow state must differ."""
    kinetics = FitzHughNagumoInteraction('kinetics', target='membrane', fast='u', slow='u')
    with pytest.raises(ValidationError, match='two different names'):
        kinetics.prepare()
