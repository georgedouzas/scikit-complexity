"""Quantum species: spins, qubits, bosonic modes, and particles on sites, with their Hilbert spaces and operators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp

from ....core.base import HilbertSpace, ValidationError, VisualSpec, state
from ....core.components import Element, QuantumView
from ....core.laws import LocalOperator, Operator, sparse_matrix
from ....core.space import LatticeSpace, NetworkSpace, Space, neighbour_pairs

DIGEST_SIZE = 8


def _spin_space(element: 'Spin') -> HilbertSpace:
    """Return the space of a spin of quantum number `j`, from `m = j` down to `m = -j`."""
    size = 2 * element.j + 1
    if not float(size).is_integer() or size < 2:  # noqa: PLR2004
        error_msg = f'Spin {element.label!r} has j = {element.j!r}, which is not a positive multiple of one half.'
        raise ValidationError(error_msg)
    operators = {
        'x': (qutip.jmat(element.j, 'x'), True),
        'y': (qutip.jmat(element.j, 'y'), True),
        'z': (qutip.jmat(element.j, 'z'), True),
        'plus': (qutip.jmat(element.j, '+'), False),
        'minus': (qutip.jmat(element.j, '-'), False),
    }
    return HilbertSpace(int(size), operators, basis={'up': 0, 'down': int(size) - 1})


QUBIT = HilbertSpace(
    2,
    {
        'x': (qutip.sigmax(), True),
        'y': (qutip.sigmay(), True),
        'z': (qutip.sigmaz(), True),
        'plus': (qutip.sigmap(), False),
        'minus': (qutip.sigmam(), False),
    },
    basis={'up': 0, 'excited': 0, 'down': 1, 'ground': 1},
)


def _mode_space(element: 'Mode') -> HilbertSpace:
    """Return the space of a bosonic mode truncated at a number of quanta."""
    quanta = element.quanta
    if isinstance(quanta, bool) or not isinstance(quanta, int | np.integer) or quanta < 1:
        error_msg = f'Mode {element.label!r} has quanta = {quanta!r}, which is not a positive integer.'
        raise ValidationError(error_msg)
    size = int(quanta) + 1
    operators = {
        'annihilate': (qutip.destroy(size), False),
        'create': (qutip.create(size), False),
        'number': (qutip.num(size), True),
    }
    return HilbertSpace(size, operators, display='number')


def _site_space(element: 'Site') -> HilbertSpace:
    """Return the space of a particle on the sites of a lattice or a network, with its Laplacian and adjacency."""
    space = element.space
    if isinstance(space, LatticeSpace):
        size = space.size
    elif isinstance(space, NetworkSpace):
        size = int(space.graph.number_of_nodes())
    else:
        error_msg = f'Site {element.label!r} has space {space!r}. Give it a LatticeSpace or a NetworkSpace.'
        raise ValidationError(error_msg)
    rows, columns = zip(*neighbour_pairs(space, size), strict=True) if size > 1 else ((), ())
    adjacency = sp.coo_array((np.ones(len(rows)), (rows, columns)), shape=(size, size))
    laplacian = -sparse_matrix(nx.laplacian_matrix(space.graph)) if isinstance(space, NetworkSpace) else None
    operators = {'laplacian': (laplacian, True), 'adjacency': (adjacency + adjacency.T, True)}
    return HilbertSpace(size, operators, display=None)


class Spin(Element):
    """Spin of quantum number `j`, with the spin components `x`, `y`, `z` and the ladder operators `plus` and `minus`.

    The basis is ordered from `m = j` to `m = -j`, so `'up'` is index 0 and `'down'` is the last index.

    Args:
        label:
            The label of the spin.

        j:
            The spin quantum number, a positive multiple of one half.

        initial:
            The state in the Hilbert space: `'up'`, `'down'`, a basis index, or a vector.

    Examples:
        >>> from skcomplex.core.components import QuantumView
        >>> from skcomplex.models.physics.realizations import Spin
        >>> QuantumView(Spin('s', j=1)).dimension
        3
    """

    visual: ClassVar[VisualSpec] = VisualSpec('expectations')
    j: float = 0.5
    initial: Any = state('up', domain=HilbertSpace.family(_spin_space))


class Qubit(Element):
    """Two-level system with the Pauli matrices `x`, `y`, `z` and the ladder operators `plus` and `minus`.

    The state `'up'`, also `'excited'`, is index 0 with `z = 1`, and `'down'`, also `'ground'`, is index 1.

    Args:
        label:
            The label of the qubit.

        initial:
            The state in the Hilbert space: `'up'`, `'down'`, `'excited'`, `'ground'`, a basis index, or a vector.

    Examples:
        >>> from skcomplex.core.components import QuantumView
        >>> from skcomplex.models.physics.realizations import Qubit
        >>> sorted(QuantumView(Qubit('q')).local_operators())
        ['minus', 'plus', 'x', 'y', 'z']
    """

    visual: ClassVar[VisualSpec] = VisualSpec('expectations')
    initial: Any = state('up', domain=QUBIT)


class Mode(Element):
    """Bosonic mode truncated at a number of quanta, with `annihilate`, `create`, and `number`.

    Args:
        label:
            The label of the mode.

        quanta:
            The largest number of quanta, so the dimension is `quanta + 1`.

        initial:
            The state in the Hilbert space: the number of quanta, or a vector.

    Examples:
        >>> from skcomplex.core.components import QuantumView
        >>> from skcomplex.models.physics.realizations import Mode
        >>> QuantumView(Mode('cavity', quanta=4)).dimension
        5
    """

    visual: ClassVar[VisualSpec] = VisualSpec('expectations')
    quanta: int = 4
    initial: Any = state(0, domain=HilbertSpace.family(_mode_space))


class Site(Element):
    """Single particle on the sites of a lattice or a network, whose basis state `k` is the particle at site `k`.

    Its namespace holds the Laplacian of the space, `laplacian`, spectral on a periodic lattice and of finite
    differences otherwise, the adjacency of neighbouring sites, `adjacency`, the site indices, `sites`, and the
    functions `projector(k)` and `diagonal(values)` that return the projector on a site and the operator with given
    values on the diagonal.

    Args:
        label:
            The label of the particle.

        space:
            The lattice or the network.

        initial:
            The state in the Hilbert space: the initial site, or a vector of amplitudes over the sites.

    Examples:
        >>> from skcomplex.core.components import QuantumView
        >>> from skcomplex.core.space import LatticeSpace
        >>> from skcomplex.models.physics.realizations import Site
        >>> QuantumView(Site('particle', space=LatticeSpace(16))).dimension
        16
    """

    visual: ClassVar[VisualSpec] = VisualSpec('expectations')
    space: Space | None = None
    initial: Any = state(0, domain=HilbertSpace.family(_site_space))

    def operator_namespace(self, key: str, view: QuantumView) -> dict[str, Any]:
        """Return the site indices and the functions `projector` and `diagonal` of the particle.

        Args:
            key:
                The key of the subsystem.

            view:
                The quantum view of the particle.

        Returns:
            names:
                A mapping from name to operator, function, or indices.
        """
        size = view.dimension

        def projector(site: int) -> Operator:
            values = np.zeros(size)
            values[view.basis_index(site)] = 1.0
            matrix = sp.csr_array(sp.diags_array(values.astype(complex)))
            return Operator({((key, LocalOperator(f'projector({site})', matrix, hermitian=True)),): 1})

        def diagonal(values: npt.ArrayLike) -> Operator:
            array = np.broadcast_to(np.asarray(values, dtype=float), (size,))
            digest = hashlib.blake2b(array.tobytes(), digest_size=DIGEST_SIZE).hexdigest()
            matrix = sp.csr_array(sp.diags_array(array.astype(complex)))
            return Operator({((key, LocalOperator(f'diagonal({digest})', matrix, hermitian=True)),): 1})

        return {'projector': projector, 'diagonal': diagonal, 'sites': tuple(range(size))}
