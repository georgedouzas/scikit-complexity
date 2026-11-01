"""Quantum species: spins, qubits, bosonic modes, and particles on sites, with their Hilbert spaces and operators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp

from .....core.spaces import BaseSpace, LatticeSpace, NetworkSpace
from .....errors import ValidationError
from .....specs import HilbertSpec

"""Dimension of the smallest spin, a spin one half."""


"""Hilbert space of a spin one half, with the spin components `x`, `y`, `z` and the ladder operators."""


"""Hilbert space of a qubit, with the Pauli matrices `x`, `y`, `z` and the ladder operators `plus` and `minus`."""


def mode_space(quanta: int = 4) -> HilbertSpec:
    """Return the Hilbert space of a bosonic mode truncated at a number of quanta.

    Its operators are `annihilate`, `create`, and `number`, and its basis state `n` holds `n` quanta.

    Args:
        quanta:
            The largest number of quanta, so the dimension is `quanta + 1`.

    Returns:
        space:
            The Hilbert space.

    Raises:
        ValidationError:
            The number of quanta is not a positive integer.

    Examples:
        >>> from skcomplex.models.physics.common.quantum import mode_space
        >>> mode_space(4).dimension
        5
    """
    if isinstance(quanta, bool) or not isinstance(quanta, int | np.integer) or quanta < 1:
        error_msg = f'A mode has quanta = {quanta!r}, which is not a positive integer.'
        raise ValidationError(error_msg)
    size = int(quanta) + 1
    operators = {
        'annihilate': (qutip.destroy(size).full(), False),
        'create': (qutip.create(size).full(), False),
        'number': (qutip.num(size).full(), True),
    }
    return HilbertSpec(size, operators, display='number')


def _lattice_laplacian(lattice: LatticeSpace) -> npt.NDArray[np.float64]:
    """Return the Laplacian of a prepared lattice, spectral when it is periodic and of finite differences otherwise."""
    size = int(lattice.size_)
    laplacian = np.zeros((size, size))
    for axis, (length, spacing) in enumerate(zip(lattice.shape_, lattice.spacings_, strict=True)):
        if lattice.periodic_:
            waves = 2 * np.pi * np.fft.fftfreq(length, d=spacing)
            line = np.fft.ifft(-(waves[:, None] ** 2) * np.fft.fft(np.eye(length), axis=0), axis=0).real
        else:
            line = (np.eye(length, k=-1) - 2 * np.eye(length) + np.eye(length, k=1)) / spacing**2
        term: npt.NDArray[np.floating] = np.ones((1, 1))
        for index, item in enumerate(lattice.shape_):
            term = np.kron(term, line if index == axis else np.eye(item))
        laplacian += term
    return laplacian


def site_space(space: BaseSpace, potential: npt.ArrayLike | None = None, *, projectors: bool = False) -> HilbertSpec:
    """Return the Hilbert space of a particle on the sites of a lattice or a network, whose basis state `k` is site `k`.

    Its operators are the Laplacian, `laplacian`, the adjacency of the neighbouring sites, `adjacency`, with a
    potential the operator `potential` with the potential of every site on its diagonal, and with projectors the
    projector `site_<k>` on every site `k`. The Laplacian of a lattice is spectral when the lattice is periodic and
    of finite differences otherwise, with the spacing of the lattice, and the Laplacian of a network is its graph
    Laplacian.

    Args:
        space:
            The lattice or the network.

        potential:
            The potential of every site, or `None` for no potential.

        projectors:
            Whether the space holds the projector on every site.

    Returns:
        space:
            The Hilbert space, of the dimension of the number of sites.

    Raises:
        ValidationError:
            The space is neither a lattice nor a network.

    Examples:
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.models.physics.common.quantum import site_space
        >>> float(site_space(LatticeSpace(16, boundary='fixed')).operators['laplacian'][0].trace())
        -32.0
    """
    if not isinstance(space, LatticeSpace | NetworkSpace):
        error_msg = f'A particle on sites needs a LatticeSpace or a NetworkSpace, not {space!r}.'
        raise ValidationError(error_msg)
    prepared = space.check().prepare()
    size = int(prepared.size_)
    adjacency = np.zeros((size, size))
    for first, second in prepared._neighbor_pairs():
        adjacency[first, second] = adjacency[second, first] = 1.0
    laplacian = (
        _lattice_laplacian(prepared)
        if isinstance(prepared, LatticeSpace)
        else adjacency - np.diag(adjacency.sum(axis=1))
    )
    operators = {'laplacian': (laplacian, True), 'adjacency': (adjacency, True)}
    if potential is not None:
        operators['potential'] = (np.diag(np.broadcast_to(np.asarray(potential, dtype=float), (size,))), True)
    for site in range(size if projectors else 0):
        operators[f'site_{site}'] = (sp.csr_array(([1.0], ([site], [site])), shape=(size, size)), True)
    return HilbertSpec(size, operators, display=None)
