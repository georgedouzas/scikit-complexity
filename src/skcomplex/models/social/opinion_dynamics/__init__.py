"""Opinion dynamics: voting and majority rules."""

from ._voters import MajorityVoter, Voter, majority_rule, voter_model

__all__: list[str] = [
    'MajorityVoter',
    'Voter',
    'majority_rule',
    'voter_model',
]
