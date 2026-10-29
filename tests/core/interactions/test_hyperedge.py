"""Test the `_hyperedge` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

from skcomplex.core.interactions import HyperedgeInteraction
from skcomplex.specs import RoleSpec


class ConsensusInteraction(HyperedgeInteraction):
    """Interaction over the hyperedges of the voters."""

    def __init__(self, label: str, *, voters: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label)
        self.voters = voters


def test_hyperedge_interaction_has_no_laws_by_default():
    """Test that a hyperedge interaction stores its role and runs into no laws of its own."""
    consensus = ConsensusInteraction('consensus', voters='voters').check().prepare()
    assert consensus.voters == 'voters'
    assert list(ConsensusInteraction._get_specs()) == ['voters']
    assert consensus.run(SimpleNamespace()) == []
