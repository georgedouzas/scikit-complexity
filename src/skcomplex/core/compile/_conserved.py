"""Lower conserved noise to the Wiener processes of the links of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR

from ..base import Expression, ValidationError
from ..components import Field
from ..laws import Noise
from ..space import LatticeSpace
from ._kernels import NoiseProcess, Output, Reference, Source

type Indices = npt.NDArray[np.intp]


def link_tables(dimensions: tuple[int, ...], periodic: bool) -> list[tuple[Indices, Indices]]:
    """Return the links of every site along every dimension of a lattice, numbered `k n + i` for site `i`.

    The link `(i, k)` joins site `i` to its next neighbour along dimension `k`. Without periodic edges, the links that
    leave the lattice carry no process.

    Args:
        dimensions:
            The number of sites along every dimension.

        periodic:
            Whether opposite edges are joined.

    Returns:
        tables:
            For every dimension, the link to the next neighbour and the link from the previous neighbour of every site
            in member order, or -1 where the link carries no process.
    """
    size = int(np.prod(dimensions))
    sites = np.arange(size, dtype=np.intp).reshape(dimensions)
    tables = []
    for axis in range(len(dimensions)):
        following = axis * size + sites
        preceding = np.roll(following, 1, axis=axis)
        if not periodic:
            following = following.copy()
            np.moveaxis(following, axis, 0)[-1] = -1
            np.moveaxis(preceding, axis, 0)[0] = -1
        tables.append((following.ravel(), preceding.ravel()))
    return tables


def _check_place(owner: str, reference: Reference, place: object) -> LatticeSpace:
    """Return the lattice of the field a conserved noise drives, or raise an error when there is none."""
    space = getattr(place, 'space', None)
    is_lattice = (
        isinstance(place, Field)
        and isinstance(space, LatticeSpace)
        and space.geometry == 'square'
        and place.discretization == 'finite-difference'
        and reference.source is Source.MEMBER
    )
    if not is_lattice:
        error_msg = (
            f'The conserved noise law of {owner} drives {reference.entry.name}, which is not a state of a '
            'finite-difference field on a square lattice. Declare the state on a `Field` of a `LatticeSpace`.'
        )
        raise ValidationError(error_msg)
    return space  # type: ignore[return-value]


def _amplitude(owner: str, law: Noise, references: Mapping[Expression, Reference]) -> Expression:
    """Return the amplitude of a conserved noise, or raise an error when it reads a state or is not symbolic."""
    if not isinstance(law.value, Expression | int | float):
        error_msg = f'The conserved noise law of {owner} has the value {law.value!r}. Give a number or an expression.'
        raise ValidationError(error_msg)
    amplitude = SR(law.value)
    reads = [item for item in amplitude.variables() if item in references and references[item].is_state]
    if reads:
        error_msg = (
            f'The amplitude of the conserved noise law of {owner} reads the states {", ".join(map(str, reads))}. '
            'Write it as a number or an expression of parameters.'
        )
        raise ValidationError(error_msg)
    return amplitude


def conserved_outputs(
    owner: str,
    law: Noise,
    targets: list[Reference],
    places: Mapping[str, Field],
    references: Mapping[Expression, Reference],
) -> list[Output]:
    """Return the outputs of a conserved noise law, the divergence of a random flux on the links of a lattice.

    Every target component gets one process per link. Site `i` takes `a / h` times the increment of the link from its
    previous neighbour and `-a / h` times the increment of the link to its next neighbour along every dimension.

    Args:
        owner:
            The path of the component that returned the law.

        law:
            The conserved noise law.

        targets:
            The target components of the law.

        places:
            The fields by the name of each of their entries.

        references:
            A mapping from every symbol the law may use to the component it stands for.

    Returns:
        outputs:
            Two outputs per dimension and target component, with the link process of every site.

    Raises:
        ValidationError:
            A target is not a state of a finite-difference field on a square lattice, or the amplitude reads a state.
    """
    amplitude = _amplitude(owner, law, references)
    outputs = []
    for reference in targets:
        place = places.get(reference.entry.name)
        space = _check_place(owner, reference, place)
        kind, _ = place.boundary_of(reference.entry.name.rpartition('.')[2])  # type: ignore[union-attr]
        tables = link_tables(space.dimensions, kind == 'periodic')
        key = ('conserved', reference.entry.name, reference.column)
        dimension = len(tables) * space.size
        for spacing, (following, preceding) in zip(space.spacings, tables, strict=True):
            scaled = amplitude / spacing
            outputs.append(Output(reference, -scaled, process=NoiseProcess(key, dimension=dimension, links=following)))
            outputs.append(Output(reference, scaled, process=NoiseProcess(key, dimension=dimension, links=preceding)))
    return outputs
