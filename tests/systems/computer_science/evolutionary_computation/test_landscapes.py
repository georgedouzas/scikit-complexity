"""Test the `_landscapes` module of `skcomplex.systems.computer_science.evolutionary_computation`."""

from skcomplex.systems.computer_science.evolutionary_computation import BitStringEvolutionSystem

from ..conftest import _run


def test_bit_string_evolution_climbs_the_landscape():
    """Test that selection raises the mean fitness of the population while its size stays constant."""
    trajectory = _run(BitStringEvolutionSystem(), 100)
    fitness = [trajectory.get_agents('genomes', frame).values['fitness'].mean() for frame in (0, 100)]
    assert fitness[1] > fitness[0] + 0.05
    assert trajectory.get_values('genomes.count')[-1] == 100
