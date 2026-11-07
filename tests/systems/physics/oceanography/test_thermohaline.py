"""Test the `_thermohaline` module of `skcomplex.systems.physics.oceanography`."""

from skcomplex.systems.physics.oceanography import StommelSystem

from ..conftest import _run_default


def test_stommel_is_bistable():
    """Test that the thermohaline boxes settle in a thermal or a haline circulation depending on their start."""
    thermal = _run_default(StommelSystem(T=3.0, S=0.0), until=100.0, dt=1.0)
    haline = _run_default(StommelSystem(T=0.0, S=4.0), until=100.0, dt=1.0)
    assert thermal.values('ocean.T')[-1] - thermal.values('ocean.S')[-1] > 0.3
    assert haline.values('ocean.T')[-1] - haline.values('ocean.S')[-1] < 0.0
