"""Test the `_fisheries` module of `skcomplex.systems.economics.resources`."""

import numpy as np

from skcomplex.systems.economics.resources import (
    HilsaFisherySystem,
)

from ...conftest import _run_monitored


def test_hilsa_fishery_settles_and_collapses_with_the_catch():
    """Test that the stock settles at the spawning-ground limit, gains from a lower juvenile catch, and collapses."""
    base = _run_monitored(HilsaFisherySystem().prepare().system_, 0.5, 240).values('hilsa.spawners')
    assert np.ptp(base[-24:]) < 1e-3 * base[-1]
    protected = _run_monitored(HilsaFisherySystem(juvenile_harvest=0.2).prepare().system_, 0.5, 240).values(
        'hilsa.spawners',
    )
    assert protected[-1] > 2 * base[-1]
    overfished = _run_monitored(HilsaFisherySystem(adult_harvest=0.9).prepare().system_, 0.5, 240).values(
        'hilsa.spawners',
    )
    assert overfished[-1] < 0.1 * overfished[0]
