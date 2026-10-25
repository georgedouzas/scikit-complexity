"""Test the `_landscapes` module of `skcomplex.models.computer_science.evolutionary_computation`."""

from skcomplex.models.computer_science.evolutionary_computation import bit_string_evolution

from ...conftest import _run


def test_bit_string_evolution_climbs_the_landscape():
    """Test that selection raises the mean fitness of the population while its size stays constant."""
    trajectory = _run(bit_string_evolution(), 100)
    fitness = [trajectory.agents('genomes', frame).values['fitness'].mean() for frame in (0, 100)]
    assert fitness[1] > fitness[0] + 0.05
    assert trajectory.values('genomes.count')[-1] == 100
