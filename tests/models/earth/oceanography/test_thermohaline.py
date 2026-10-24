"""Test the `_thermohaline` module of `skcomplex.models.earth.oceanography`."""

from skcomplex.models.earth.oceanography import stommel

from ..conftest import _run


def test_stommel_is_bistable():
    """Test that the thermohaline boxes settle in a thermal or a haline circulation depending on their start."""
    thermal = _run(stommel(T=3.0, S=0.0), until=100.0, dt=1.0)
    haline = _run(stommel(T=0.0, S=4.0), until=100.0, dt=1.0)
    assert thermal['ocean.T'][-1] - thermal['ocean.S'][-1] > 0.3
    assert haline['ocean.T'][-1] - haline['ocean.S'][-1] < 0.0
