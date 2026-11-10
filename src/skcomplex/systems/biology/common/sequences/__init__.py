"""Sequences shared by the biology packages.

They are sampled row by row, such as the sequences of RNA walks and the conformations of proteins.
"""

from ._sequence_sampling import RowScoreSpec, SequenceSamplingInteraction

__all__: list[str] = [
    'RowScoreSpec',
    'SequenceSamplingInteraction',
]
