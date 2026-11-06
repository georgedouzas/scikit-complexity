"""Cellular Potts sampling of a tissue: label copies that move cell boundaries under adhesion and area constraints."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import SamplingInteraction
from ....core.components.spaces import BaseNeighborhood
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RegionScoreSpec, RoleSpec
from ....core.types import Array, Indices

MATRIX_AXES = 2


class CellAdhesionInteraction(SamplingInteraction):
    """Cellular Potts sampling of a tissue, whose cells move their boundaries under adhesion and area constraints.

    The energy of the Graner-Glazier model is `sum_<i,j> w_ij J(t(s_i), t(s_j)) [s_i != s_j] + lambda sum_s (a_s -
    A)^2 + mu sum_s (p_s - P)^2`, where `s_i` is the cell of site `i`, `t(s)` the kind of cell `s`, `J` the adhesion
    energy per unit contact between kinds, `w_ij` the weight of the contact of two sites, `a_s` the area of a cell, and
    `p_s` its perimeter, the weighted number of its contacts with other cells. Every sweep proposes `hits` copies per
    site of the cell of a random neighbour into a random site, accepted by the Metropolis rule at the temperature. A
    heterotypic adhesion above the mean of the homotypic ones sorts the kinds, the differential adhesion of Steinberg.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of sites.

        kinds:
            The kind of every cell, by label.

        adhesion:
            The adhesion energy per unit contact between cells of every pair of kinds, a symmetric matrix.

        temperature:
            The temperature of the copies, the amplitude of the fluctuations of the cell boundaries.

        area_strength:
            The strength `lambda` of the area constraint.

        target_area:
            The target area `A` of every cell.

        perimeter_strength:
            The strength `mu` of the perimeter constraint.

        target_perimeter:
            The target perimeter `P` of every cell.

        state:
            The name of the state that holds the cell of a site.

        rule:
            The acceptance, `'metropolis'` or `'glauber'`.

        neighborhood:
            The neighbours of a site, or `None` for the default neighbourhood of the space.

        hits:
            The number of copies proposed per site and sweep.

    Raises:
        ValidationError:
            The adhesion is not a symmetric square matrix, or a kind is not one of its rows.

    Examples:
        >>> from skcomplex.systems.biology.physiology import CellAdhesionInteraction
        >>> adhesion = CellAdhesionInteraction('adhesion', target='tissue', kinds=(0, 1), adhesion=((1, 2), (2, 1)))
        >>> adhesion.objective.contact(np.array([0, 0]), np.array([0, 1])).tolist()
        [1.0, 2.0]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        kinds: Sequence[int],
        adhesion: npt.ArrayLike,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        area_strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        target_area: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 40.0,
        perimeter_strength: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        target_perimeter: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        state: str = 'cell',
        rule: str = 'metropolis',
        neighborhood: BaseNeighborhood | None = None,
        hits: int = 1,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            objective=RegionScoreSpec(self._get_contact, self._get_area, self._get_perimeter, hits),
            move='copy',
            rule=rule,
            neighborhood=neighborhood,
        )
        self.kinds = kinds
        self.adhesion = adhesion
        self.temperature = temperature
        self.area_strength = area_strength
        self.target_area = target_area
        self.perimeter_strength = perimeter_strength
        self.target_perimeter = target_perimeter
        self.hits = hits

    def _get_intensity(self, namespace: SimpleNamespace) -> object:
        """Return the intensity of the acceptance, one over the temperature."""
        return 1 / SR(self._get_parameter_symbol('temperature', namespace))

    def _get_contact(self, first: Indices, second: Indices) -> Array:
        """Return the adhesion energy per unit contact between the cells of two arrays of labels."""
        kinds = np.asarray(self.kinds, dtype=np.intp)
        return np.asarray(self.adhesion, dtype=float)[kinds[first], kinds[second]]

    def _get_area(self, labels: Indices, area_strength: float, target_area: float) -> tuple[float, float]:
        """Return the strength and the target of the area constraint, the same for every cell."""
        del labels
        return area_strength, target_area

    def _get_perimeter(
        self,
        labels: Indices,
        perimeter_strength: float,
        target_perimeter: float,
    ) -> tuple[float, float]:
        """Return the strength and the target of the perimeter constraint, the same for every cell."""
        del labels
        return perimeter_strength, target_perimeter

    def _check(self) -> None:
        """Raise an error for an adhesion that is not a symmetric square matrix, or a kind that is not one of its rows.

        Raises:
            ValidationError:
                The adhesion is not a symmetric square matrix, or a kind is not one of its rows.
        """
        super()._check()
        adhesion = np.asarray(self.adhesion, dtype=float)
        if (
            adhesion.ndim != MATRIX_AXES
            or adhesion.shape[0] != adhesion.shape[1]
            or not np.allclose(adhesion, adhesion.T)
        ):
            error_msg = f'The adhesion of {self.label!r} is a symmetric matrix over the kinds, but got {adhesion!r}.'
            raise ValidationError(error_msg)
        kinds = np.asarray(self.kinds)
        if kinds.ndim != 1 or (kinds.size and (kinds.min() < 0 or kinds.max() >= adhesion.shape[0])):
            error_msg = f'The kinds of {self.label!r} are rows of the adhesion, from 0 to {adhesion.shape[0] - 1}.'
            raise ValidationError(error_msg)
