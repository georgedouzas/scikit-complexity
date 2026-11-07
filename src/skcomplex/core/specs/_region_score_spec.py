"""Define the region score spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class RegionScoreSpec(BaseSpec):
    """Spec of an objective of a state of integer labels, whose sites of one label form a region.

    The objective is `sum_<i,j> w_ij c(l_i, l_j) [l_i != l_j] + sum_l a_l (V_l - A_l)^2 + b_l (S_l - B_l)^2`, the
    contact energy across the boundaries between regions and an elastic constraint on the volume and the surface of
    every region. The first sum runs over the pairs of neighbouring sites `i` and `j` with the weight `w_ij` of their
    contact, `l_i` is the label of site `i`, `V_l` the volume of region `l`, the sum of the volumes of its sites, and
    `S_l` its surface, the sum of the weights of the contacts of its sites with sites of other labels. A label copy
    changes the volumes and the surfaces of only the two regions it moves a site between, so a sampler keeps them as
    totals per label that every accepted move updates.

    Args:
        contact:
            The function `contact(first, second, **parameters)` of two arrays of labels, which broadcast, and the
            parameters of the system that it names, which returns the energy per unit of contact between regions of the
            labels, symmetric in the labels.

        volume:
            The function `volume(labels, **parameters)` of an array of labels, which returns the strength `a_l` and the
            target `A_l` of the volume constraint of every label, or `None` for no volume constraint.

        surface:
            The function `surface(labels, **parameters)` of an array of labels, which returns the strength `b_l` and
            the target `B_l` of the surface constraint of every label, or `None` for no surface constraint.

        hits:
            The number of copies proposed per site and sweep.

    Raises:
        ValidationError:
            The contact or a given constraint is not callable, or the number of hits is not positive.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.specs import RegionScoreSpec
        >>> contact = lambda first, second: np.ones(np.broadcast(first, second).shape)
        >>> area = lambda labels: (np.ones(labels.shape), np.full(labels.shape, 16.0))
        >>> RegionScoreSpec(contact, volume=area, hits=2).hits
        2
    """

    contact: Callable[..., object]
    volume: Callable[..., object] | None = None
    surface: Callable[..., object] | None = None
    hits: int = 1

    def _check(self) -> None:
        """Raise an error for a contact or a constraint that is not callable, or a number of hits below 1.

        Raises:
            ValidationError:
                The contact or a given constraint is not callable, or the number of hits is not positive.
        """
        for name in ('contact', 'volume', 'surface'):
            function = getattr(self, name)
            if not callable(function) and (name == 'contact' or function is not None):
                error_msg = f'The {name} of a region score must be callable, but it is {function!r}.'
                raise ValidationError(error_msg)
        if not isinstance(self.hits, int) or self.hits < 1:
            error_msg = f'A region score proposes at least one copy per site, but it has hits={self.hits!r}.'
            raise ValidationError(error_msg)
