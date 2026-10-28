"""Realizations shared by the social packages: imitation and the dissimilarity index."""

from ._dissimilarity_index import DissimilarityIndexAnalysis
from ._learning import imitate

__all__: list[str] = ['DissimilarityIndexAnalysis', 'imitate']
