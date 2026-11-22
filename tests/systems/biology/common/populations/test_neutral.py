"""Test the `_neutral` module of `skcomplex.systems.biology.common.populations`."""

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.populations import (
    NeutralBirthInteraction,
    NeutralDeathInteraction,
)


def test_neutral_births_and_deaths_change_counts_by_one():
    """Test that births and deaths keep integer counts that stay at zero or above."""
    system = System(
        elements=[Element('species', values={'x': 5.0}, size=10)],
        interactions=[
            NeutralBirthInteraction('birth', target='species', b0=0.5, b1=0.9),
            NeutralDeathInteraction('death', target='species', d0=0.1, d1=1.0),
        ],
    )
    counts = system.prepare().simulate(until=10.0, dt=1.0, random_state=0).get_values('species.x')
    assert np.array_equal(counts, np.round(counts))
    assert counts.min() >= 0


def test_neutral_deaths_alone_empty_the_species():
    """Test that deaths without births leave no individual after a long run."""
    system = System(
        elements=[Element('species', values={'x': 3.0})],
        interactions=[NeutralDeathInteraction('death', target='species', d0=1.0, d1=1.0)],
    )
    assert system.prepare().simulate(until=50.0, random_state=0).get_values('species.x')[-1] == 0
