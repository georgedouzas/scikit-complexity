"""Test the `_dosing` module of `skcomplex.systems.biology.common.pharmacokinetics`."""

import numpy as np
import pytest

from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.common.kinetics import CompartmentElement
from skcomplex.systems.biology.common.pharmacokinetics import DosingInteraction


def _run(interaction, until=40.0, dt=0.5):
    """Return the times and the drug of one compartment under one dosing."""
    system = CompositeSystem(elements=[CompartmentElement('patient', x=0.0)], interactions=[interaction])
    trajectory = simulate(system, dt=dt, until=until)
    return np.asarray(trajectory.times), np.asarray(trajectory.values('patient.x'))


def test_dosing_without_period_is_a_constant_input():
    """Test that a dosing without a period raises the drug at the constant rate of the dose."""
    times, drug = _run(DosingInteraction('dosing', target='patient', state='x', dose=1.5))
    np.testing.assert_allclose(drug, 1.5 * times, rtol=1e-9, atol=1e-12)


def test_dosing_with_count_stops_after_the_last_infusion():
    """Test that a schedule of three infusions gives three doses and nothing after them."""
    dosing = DosingInteraction('dosing', target='patient', state='x', dose=2.0, period=10.0, width=1.0, count=3)
    times, drug = _run(dosing)
    assert drug[-1] == pytest.approx(6.0, rel=1e-6)
    assert np.allclose(drug[times >= 21.0], 6.0, rtol=1e-6)
    assert drug[np.isclose(times, 5.0)][0] == pytest.approx(2.0, rel=1e-6)


def test_dosing_without_count_never_ends():
    """Test that a schedule without a count keeps infusing from its start."""
    dosing = DosingInteraction('dosing', target='patient', state='x', dose=1.0, period=10.0, width=2.0, start=5.0)
    times, drug = _run(dosing)
    assert drug[np.isclose(times, 5.0)][0] == pytest.approx(0.0, abs=1e-9)
    assert drug[-1] == pytest.approx(8.0, rel=1e-6)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'dose': 'high'}, 'dose'),
        ({'period': 0.0}, 'period'),
        ({'period': 10.0, 'width': -1.0}, 'width'),
    ],
)
def test_dosing_rejects_invalid_arguments(arguments, message):
    """Test that the dosing rejects a dose that is no number or function, and a schedule that is not positive."""
    with pytest.raises(ValidationError, match=message):
        DosingInteraction('dosing', target='patient', state='x', **arguments).prepare()
