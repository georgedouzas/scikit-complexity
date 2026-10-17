"""Define the spaces that vector states live in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sklearn.base import BaseEstimator


class Space(BaseEstimator):
    """Base class of spaces, which give vector states and parameters their dimension.

    Subclasses expose the number of dimensions as `n_dim`.
    """

    n_dim: int


class EuclideanSpace(Space):
    """Euclidean space with Cartesian coordinates.

    Args:
        n_dim:
            The number of dimensions.

    Examples:
        >>> from skcomplex.model import EuclideanSpace
        >>> EuclideanSpace(n_dim=2).n_dim
        2
    """

    def __init__(self, n_dim: int = 3) -> None:
        self.n_dim = n_dim
