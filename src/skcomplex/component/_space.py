"""Base classes for systems spaces."""

# Author: Georgios Douzas <gdouzas@icloud.com> License: MIT

from typing import Literal, Self

from sage.manifolds.all import EuclideanSpace as EuclideanManifold
from sklearn.utils import check_scalar

from ..base.system import BaseSpace

N_DIM = 3


class EuclideanSpace(BaseSpace):
    """Euclidean space class."""

    def __init__(
        self: Self,
        label: str,
        n_dim: int = 3,
        coordinate_system: Literal['cartesian', 'polar', 'spherical', 'cylindrical'] = 'cartesian',
    ) -> None:
        """Initialize the Euclidean space object."""
        self.n_dim = n_dim
        self.coordinate_system = coordinate_system
        super().__init__(label)

    def _init_param(self: Self, param_name: str) -> Self:
        """Check Euclidean space parameters."""
        if param_name == 'n_dim':
            self.n_dim_ = check_scalar(self.n_dim, 'n_dim', int, min_val=1, max_val=3)
        elif param_name == 'coordinate_system':
            self.manifold_ = EuclideanManifold(self.n_dim_)
            if self.coordinate_system == 'cartesian':
                self.coordinates_ = self.manifold_.cartesian_coordinates()
            elif self.n_dim_ == N_DIM - 1 and self.coordinate_system == 'polar':
                self.coordinates_ = self.manifold_.polar_coordinates()
            elif self.n_dim_ == N_DIM and self.coordinate_system == 'spherical':
                self.coordinates_ = self.manifold_.spherical_coordinates()
            elif self.n_dim_ == N_DIM and self.coordinate_system == 'cylindrical':
                self.coordinates_ = self.manifold_.cylindrical_coordinates()
            elif self.n_dim_ == N_DIM - 2 and self.coordinate_system != 'cartesian':
                error_msg = 'Parameter `coordinate_system` should be equal to `\'euclidean\'`.'
                raise ValueError(error_msg)
            elif self.n_dim_ == N_DIM - 1 and self.coordinate_system not in ('cartesian', 'polar'):
                error_msg = 'Parameter `coordinate_system` should be equal to either `\'euclidean\'` or `\'polar\'`.'
                raise ValueError(
                    error_msg,
                )
            elif self.n_dim_ == N_DIM and self.coordinate_system not in ('cartesian', 'spherical', 'cylindrical'):
                error_msg = (
                    'Parameter `coordinate_system` should be equal to either `\'euclidean\'` or'
                    ' `\'spherical\'` or `\'cylindrical\'`.'
                )
                raise ValueError(
                    error_msg,
                )
        return self
