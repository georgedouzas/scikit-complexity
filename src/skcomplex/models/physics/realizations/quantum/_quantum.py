"""Quantum species: spins, qubits, bosonic modes, and particles on sites, with their Hilbert spaces and operators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp
from sage.all__sagemath_symbolics import CC

from .....core.elements import Element
from .....core.errors import ValidationError
from .....core.spaces import BaseSpace, LatticeSpace, NetworkSpace
from .....core.specs import HilbertSpec, StateSpec, VisualSpec


def spin_space(j: float = 0.5) -> HilbertSpec:
    """Return the Hilbert space of a spin of quantum number `j`, with the basis from `m = j` down to `m = -j`.

    Its operators are the spin components `x`, `y`, `z` and the ladder operators `plus` and `minus`.

    Args:
        j:
            The spin quantum number, a positive multiple of one half.

    Returns:
        space:
            The Hilbert space, of dimension `2 j + 1`.

    Raises:
        ValidationError:
            The quantum number is not a positive multiple of one half.

    Examples:
        >>> from skcomplex.models.physics.realizations.quantum import spin_space
        >>> spin_space(1).dimension
        3
    """
    size = 2 * j + 1
    if not float(size).is_integer() or size < 2:  # noqa: PLR2004
        error_msg = f'A spin has j = {j!r}, which is not a positive multiple of one half.'
        raise ValidationError(error_msg)
    operators = {
        'x': (qutip.jmat(j, 'x').full(), True),
        'y': (qutip.jmat(j, 'y').full(), True),
        'z': (qutip.jmat(j, 'z').full(), True),
        'plus': (qutip.jmat(j, '+').full(), False),
        'minus': (qutip.jmat(j, '-').full(), False),
    }
    return HilbertSpec(int(size), operators, basis={'up': 0, 'down': int(size) - 1})


SPIN_HALF = spin_space()
"""Hilbert space of a spin one half, with the spin components `x`, `y`, `z` and the ladder operators."""

QUBIT = HilbertSpec(
    2,
    {
        'x': (qutip.sigmax().full(), True),
        'y': (qutip.sigmay().full(), True),
        'z': (qutip.sigmaz().full(), True),
        'plus': (qutip.sigmap().full(), False),
        'minus': (qutip.sigmam().full(), False),
    },
    basis={'up': 0, 'excited': 0, 'down': 1, 'ground': 1},
)
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
        >>> from skcomplex.models.physics.realizations.quantum import mode_space
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


def site_space(space: BaseSpace, potential: npt.ArrayLike | None = None, *, projectors: bool = False) -> HilbertSpec:
    """Return the Hilbert space of a particle on the sites of a lattice or a network, whose basis state `k` is site `k`.

    Its operators are the graph Laplacian of the neighbouring sites, `laplacian`, their adjacency, `adjacency`, with a
    potential the operator `potential` with the potential of every site on its diagonal, and with projectors the
    projector `site_<k>` on every site `k`, as sparse matrices.

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
        >>> from skcomplex.models.physics.realizations.quantum import site_space
        >>> site_space(LatticeSpace(16)).operators['laplacian'][0].trace()
        np.float64(-30.0)
    """
    if not isinstance(space, LatticeSpace | NetworkSpace):
        error_msg = f'A particle on sites needs a LatticeSpace or a NetworkSpace, not {space!r}.'
        raise ValidationError(error_msg)
    prepared = space.check().prepare()
    size = int(prepared.size_)
    adjacency = np.zeros((size, size))
    for first, second in prepared._neighbor_pairs():
        adjacency[first, second] = adjacency[second, first] = 1.0
    laplacian = adjacency - np.diag(adjacency.sum(axis=1))
    operators = {'laplacian': (laplacian, True), 'adjacency': (adjacency, True)}
    if potential is not None:
        operators['potential'] = (np.diag(np.broadcast_to(np.asarray(potential, dtype=float), (size,))), True)
    for site in range(size if projectors else 0):
        operators[f'site_{site}'] = (sp.csr_array(([1.0], ([site], [site])), shape=(size, size)), True)
    return HilbertSpec(size, operators, display=None)


class SpinElement(Element):
    """Spin of the Hilbert space it takes, with the spin components `x`, `y`, `z` and the ladder operators.

    The basis is ordered from `m = j` to `m = -j`, so `'up'` is index 0 and `'down'` is the last index. Its laws read
    the operators of its state `psi`, such as `local_operator(namespace.psi, 'z')`.

    Args:
        label:
            The label of the spin.

        psi:
            The state in the Hilbert space: `'up'`, `'down'`, a basis index, or a vector.

        hilbert:
            The Hilbert space, `spin_space(j)`, by default that of a spin one half.

        visual:
            How the spin is drawn.

    Examples:
        >>> from skcomplex.models.physics.realizations.quantum import SpinElement, spin_space
        >>> SpinElement('s', hilbert=spin_space(1)).hilbert.dimension
        3
    """

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        *,
        hilbert: HilbertSpec = SPIN_HALF,
        visual: VisualSpec | None = VisualSpec('expectations'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert


class QubitElement(Element):
    """Two-level system with the Pauli matrices `x`, `y`, `z` and the ladder operators `plus` and `minus`.

    The state `'up'`, also `'excited'`, is index 0 with `z = 1`, and `'down'`, also `'ground'`, is index 1.

    Args:
        label:
            The label of the qubit.

        psi:
            The state in the Hilbert space: `'up'`, `'down'`, `'excited'`, `'ground'`, a basis index, or a vector.

        hilbert:
            The Hilbert space of the qubit.

        visual:
            How the qubit is drawn.

    Examples:
        >>> from skcomplex.models.physics.realizations.quantum import QubitElement
        >>> sorted(QubitElement('q').hilbert.operators)
        ['minus', 'plus', 'x', 'y', 'z']
    """

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        *,
        hilbert: HilbertSpec = QUBIT,
        visual: VisualSpec | None = VisualSpec('expectations'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert


class ModeElement(Element):
    """Bosonic mode of the Hilbert space it takes, with `annihilate`, `create`, and `number`.

    Args:
        label:
            The label of the mode.

        psi:
            The state in the Hilbert space: the number of quanta, or a vector.

        hilbert:
            The Hilbert space, `mode_space(quanta)`.

        visual:
            How the mode is drawn.

    Examples:
        >>> from skcomplex.models.physics.realizations.quantum import ModeElement, mode_space
        >>> ModeElement('cavity', hilbert=mode_space(4)).hilbert.dimension
        5
    """

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 0,
        *,
        hilbert: HilbertSpec | None = None,
        visual: VisualSpec | None = VisualSpec('expectations'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert


class SiteElement(Element):
    """Single particle on the sites of a lattice or a network, whose basis state `k` is the particle at site `k`.

    Its laws read the operators of `site_space`: `laplacian`, `adjacency`, and `potential`.

    Args:
        label:
            The label of the particle.

        psi:
            The state in the Hilbert space: the initial site, or a vector of amplitudes over the sites.

        hilbert:
            The Hilbert space, `site_space(space, potential)`.

        visual:
            How the particle is drawn.

    Examples:
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.models.physics.realizations.quantum import SiteElement, site_space
        >>> SiteElement('particle', hilbert=site_space(LatticeSpace(16))).hilbert.dimension
        16
    """

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 0,
        *,
        hilbert: HilbertSpec | None = None,
        visual: VisualSpec | None = VisualSpec('expectations'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert
