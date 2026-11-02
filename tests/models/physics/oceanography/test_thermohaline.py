"""Test the `_thermohaline` module of `skcomplex.models.physics.oceanography`."""

from skcomplex.models.physics.oceanography import stommel

from ..conftest import _run_default


def test_stommel_is_bistable():
    """Test that the thermohaline boxes settle in a thermal or a haline circulation depending on their start."""
    thermal = _run_default(stommel(T=3.0, S=0.0), until=100.0, dt=1.0)
    haline = _run_default(stommel(T=0.0, S=4.0), until=100.0, dt=1.0)
    assert thermal.values('ocean.T')[-1] - thermal.values('ocean.S')[-1] > 0.3
    assert haline.values('ocean.T')[-1] - haline.values('ocean.S')[-1] < 0.0
