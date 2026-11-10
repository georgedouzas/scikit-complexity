"""Learning realizations shared by the economics packages.

They are the scores of options that agents learn from realized payoffs.
"""

from ._payoff_learning import PayoffLearningInteraction

__all__: list[str] = [
    'PayoffLearningInteraction',
]
