"""Test the `_random_partition` module of `skcomplex.core.components.interactions`."""

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction, RandomPartition
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import count, mean, where
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])


def _stream(seed: int = 0) -> np.random.Generator:
    """Return the generator a simulation with a seed hands its solver."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])


def _majority(size: int = 3) -> Interaction:
    """Return discussions in random groups that adopt the opinion of their majority, ties to zero."""
    return Interaction(
        'talks',
        among='voters',
        along=RandomPartition(size),
        laws=lambda s: [UpdateLaw(s.member.opinion, where(mean(s.members.opinion, over=s.members) > 0.5, 1, 0))],
    )


def _run(opinions: object, interaction: Interaction, until: int, backend: object = 'auto', seed: int = 0) -> np.ndarray:
    """Return the opinions of the voters at every frame of a run."""
    voters = Element('voters', values={'opinion': opinions}, size=len(np.atleast_1d(opinions)))
    system = System(elements=[voters], interactions=[interaction], backend=backend).prepare()
    return system.simulate(until=until, random_state=seed).get_values('voters.opinion')


@BACKENDS
def test_random_partition_sets_every_group_to_its_majority(backend):
    """Test that after one step every group of three of the drawn order holds the majority opinion it had."""
    opinions = np.random.default_rng(1).integers(0, 2, 30).astype(float)
    values = _run(opinions, _majority(), 1, backend)
    groups = _stream().permutation(30).reshape(10, 3)
    expected = np.repeat((opinions[groups].sum(axis=1) >= 2).astype(float), 3)
    assert values[1][groups.ravel()].tolist() == expected.tolist()


@BACKENDS
def test_random_partition_draws_from_the_random_stream_of_the_run(backend):
    """Test that two runs with one seed draw the same groups, and runs with two seeds draw different ones."""
    opinions = np.random.default_rng(2).integers(0, 2, 60).astype(float)
    first, second, third = (_run(opinions, _majority(), 3, backend, seed) for seed in (0, 0, 1))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_random_partition_puts_the_remainder_in_a_smaller_group():
    """Test that a partition of seven members into threes gives every member the size of its group."""
    sizes = Interaction(
        'sizes',
        among='voters',
        along=RandomPartition(3),
        laws=lambda s: [UpdateLaw(s.member.opinion, count(over=s.members))],
    )
    assert sorted(_run(np.zeros(7), sizes, 1)[1].tolist()) == [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]


def test_random_partition_makes_the_system_stochastic():
    """Test that random groups tag the interaction as one that draws."""
    assert _majority().get_tags().randomness == 'stochastic'


@pytest.mark.parametrize('size', [1, 2.5, True])
def test_random_partition_refuses_groups_below_two_members(size):
    """Test that a size that is not a whole number of at least two raises an error."""
    with pytest.raises(ValidationError, match='at least 2 members'):
        RandomPartition(size)._check()


def test_random_partition_needs_the_group_among():
    """Test that random groups with no group to draw them from raise an error."""
    with pytest.raises(ValidationError, match='among'):
        Interaction('talks', along=RandomPartition(3)).prepare()
