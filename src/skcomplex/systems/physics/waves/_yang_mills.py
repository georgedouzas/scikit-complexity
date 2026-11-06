"""Lattice Yang-Mills theory: SU(2) and SU(3) link matrices sampled by the Wilson action, pure lattice QCD for SU(3)."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from functools import partial

import numpy as np
import numpy.typing as npt

from ....core.components.elements import GroupElement
from ....core.components.spaces import CellComplexSpace, DiscreteTime, LatticeSpace
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import MatrixGroupSpec, ObservableSpec
from ..._catalog import register_system
from ._color_link import COLOR, ColorLinkElement
from ._gauge_analysis import GaugeAnalysis
from ._isospin_link import ISOSPIN, IsospinLinkElement
from ._wilson_sampling import WilsonSamplingInteraction

LINKS = {2: (IsospinLinkElement, ISOSPIN), 3: (ColorLinkElement, COLOR)}


def _mean_plaquette(space: CellComplexSpace, group: MatrixGroupSpec, t: float, values: Mapping[str, object]) -> float:
    """Return the mean of `Re tr U_p / N` over the plaquettes."""
    del t
    matrices = group.matrices_of(np.reshape(np.asarray(values['links.U'], dtype=float), (-1, group.size)))
    plaquettes = group.holonomy_of(matrices, space.boundaries_, space.boundary_signs_)
    return float(np.trace(plaquettes, axis1=-2, axis2=-1).real.mean() / group.degree)


def _polyakov_loop(space: CellComplexSpace, group: MatrixGroupSpec, t: float, values: Mapping[str, object]) -> float:
    """Return the modulus of the mean of `tr P / N` over the loops that wind along the last axis."""
    del t
    links = np.reshape(np.asarray(values['links.U'], dtype=float), (-1, group.size))
    return float(GaugeAnalysis('polyakov', space, group=group).prepare(links).run().value_[0])


def _yang_mills_observables(system: System) -> list[ObservableSpec]:
    """Return the mean plaquette and the Polyakov loop of the links."""
    links = next(element for element in system.elements if element.label == 'links')
    space = links.space.prepare()
    group = next(group for element, group in LINKS.values() if links.element is element)
    return [
        ObservableSpec('mean_plaquette', function=partial(_mean_plaquette, space, group)),
        ObservableSpec('polyakov_loop', function=partial(_polyakov_loop, space, group)),
    ]


@register_system(
    field='physics/waves',
    tags=('gauge', 'lattice', 'monte carlo', 'yang-mills', 'qcd', 'cell complex'),
    references=('Satz 2022', 'Schwichtenberg 2020', 'Wilson 1974', 'Creutz 1980'),
    defaults={'until': 100, 'dt': 1},
    observables=_yang_mills_observables,
)
def lattice_yang_mills(
    size: int = 6,
    *,
    dimension: int = 4,
    extent: int | None = None,
    colors: int = 2,
    beta: float = 2.3,
    width: float = 0.5,
    hits: int = 4,
    links: npt.ArrayLike | None = None,
    seed: int = 0,
) -> System:
    """Return SU(2) or SU(3) matrices on the links of a periodic lattice, sampled by the Wilson action.

    The link matrices sample the Wilson action `-(beta / N) sum_p Re tr U_p` under discrete time, where `U_p` is the
    ordered product of the link matrices around the plaquette `p`, with the inverse of a link traversed against its
    orientation. Every sweep proposes `hits` Metropolis moves per link, each a product with a random matrix of the group
    near the identity, and projects the links back onto the group. SU(2) is the isospin connection of the weak force,
    and SU(3) the color connection of the strong force, so that SU(3) is pure lattice QCD. The last axis has `extent`
    sites, the inverse temperature of the lattice, along which the Polyakov loop winds. In two dimensions the plaquettes
    are independent, and the SU(2) mean plaquette at large size is `I_2(beta) / I_1(beta)`.

    Args:
        size:
            The number of sites along every axis but the last.

        dimension:
            The number of axes of the lattice.

        extent:
            The number of sites along the last axis, or `None` for `size`.

        colors:
            The degree `N` of the group SU(N), 2 or 3.

        beta:
            The inverse coupling of the Wilson action, `2 N / g^2`.

        width:
            The spread around the identity of the matrices that move a link.

        hits:
            The number of moves proposed per link and sweep.

        links:
            The initial link matrices, one `N` by `N` complex matrix per link, or `None` for matrices drawn from the
            Haar measure of the group.

        seed:
            The seed of the initial link matrices.

    Returns:
        system:
            The system.

    Raises:
        ValidationError:
            The number of colors is not 2 or 3, or the initial links do not hold one matrix of the group per link.

    Examples:
        >>> from skcomplex.systems.physics.waves import lattice_yang_mills
        >>> [element.label for element in lattice_yang_mills(2, dimension=2).elements]
        ['links']
        >>> lattice_yang_mills(2, dimension=2, colors=3).elements[0].element.__name__
        'ColorLinkElement'
    """
    if colors not in LINKS:
        error_msg = f'The lattice gauge group is SU(2) or SU(3), so colors is 2 or 3, but got {colors!r}.'
        raise ValidationError(error_msg)
    element, group = LINKS[colors]
    shape = (size,) * (dimension - 1) + (size if extent is None else extent,)
    space = CellComplexSpace(LatticeSpace(shape)).prepare()
    count = space.incidences_[0].shape[0]
    if links is None:
        matrices = group.uniform_of(np.random.default_rng(seed), count)
    else:
        matrices = np.asarray(links, dtype=complex)
    if matrices.shape != (count, colors, colors) or not np.allclose(group.projection_of(matrices), matrices):
        error_msg = (
            f'The initial links are {matrices.shape[0]} matrices of the shape {matrices.shape[1:]}, but the lattice '
            f'takes {count} matrices of SU({colors}), one per link.'
        )
        raise ValidationError(error_msg)
    group_links = GroupElement(
        'links',
        element=element,
        space=space,
        on='edges',
        values={'U': group.components_of(matrices)},
    )
    sampling = WilsonSamplingInteraction(
        'wilson',
        target='links',
        beta=beta,
        state='U',
        width=width,
        group=group,
        hits=hits,
    )
    return System(elements=[group_links], interactions=[sampling], time=DiscreteTime())
