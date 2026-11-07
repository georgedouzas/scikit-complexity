"""Quantum spin chains: the quantum Heisenberg XXZ chain with a field and dephasing."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import qutip
from sage.all__sagemath_symbolics import CC, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import BaseInteraction, Interaction, NeighborInteraction
from ....core.components.laws import BaseLaw, DissipatorLaw, GeneratorLaw
from ....core.components.laws.utils import local_operator
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.compile import OperatorCompiler
from ....core.errors import ValidationError
from ....core.specs import HilbertSpec, ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system

MIN_SPIN_DIMENSION = 2


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
        >>> from skcomplex.systems.physics.statistical_mechanics._quantum_spins import spin_space
        >>> spin_space(1).dimension
        3
    """
    size = 2 * j + 1
    if not float(size).is_integer() or size < MIN_SPIN_DIMENSION:
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
        >>> from skcomplex.systems.physics.statistical_mechanics._quantum_spins import SpinElement, spin_space
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


class DephasingInteraction(Interaction):
    """Dephasing of every spin of a chain, a dissipator with the collapse operator `S^z_i` at the rate `gamma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the spin chain.

        gamma:
            The dephasing rate.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import DephasingInteraction
        >>> DephasingInteraction('dephasing', target='chain', gamma=0.1).gamma
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return one dissipator per spin.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The dissipators.
        """
        return [DissipatorLaw(local_operator(member.psi, 'z'), s.gamma) for member in s.target]


class XXZExchangeInteraction(NeighborInteraction):
    """Exchange `J (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j)` between neighbouring spins.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the spin chain.

        J:
            The exchange coupling, antiferromagnetic when positive.

        delta:
            The anisotropy `Δ`, 1 for the isotropic chain and 0 for the XX chain.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import XXZExchangeInteraction
        >>> XXZExchangeInteraction('exchange', target='chain', delta=0.5).delta
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        J: Annotated[float, ParameterSpec()] = 1.0,
        delta: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.J = J
        self.delta = delta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the exchange Hamiltonian of one bond.

        Args:
            s:
                The namespace of the bond, with `s.source` and `s.target`.

        Returns:
            laws:
                The Hamiltonian of the bond.
        """
        first, second = s.source.psi, s.target.psi
        exchange = (
            local_operator(first, 'x') * local_operator(second, 'x')
            + local_operator(first, 'y') * local_operator(second, 'y')
            + s.delta * local_operator(first, 'z') * local_operator(second, 'z')
        )
        return [GeneratorLaw(s.J * exchange)]


class ZeemanFieldInteraction(Interaction):
    """Uniform field `-h Σ S^z_i` on the spins of a chain.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the spin chain.

        h:
            The field.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ZeemanFieldInteraction
        >>> ZeemanFieldInteraction('field', target='chain', h=0.2).h
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        h: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.h = h

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Hamiltonian of the field.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The field Hamiltonian.
        """
        return [GeneratorLaw(-s.h * sum(local_operator(member.psi, 'z') for member in s.target))]


INITIAL_STATES = ('neel', 'flip', 'domain_wall', 'ground')


BOUNDARIES = {'periodic': 'periodic', 'open': 'fixed'}


def _initial_labels(n_sites: int, initial: str) -> list[str]:
    """Return the basis label of every spin of a product initial state."""
    if initial == 'neel':
        return ['up' if site % 2 == 0 else 'down' for site in range(n_sites)]
    if initial == 'flip':
        return ['down' if site == n_sites // 2 else 'up' for site in range(n_sites)]
    return ['up' if site < n_sites // 2 else 'down' for site in range(n_sites)]


def _sites(values: object) -> object:
    """Return the values of every site, without reducing them."""
    return values


def local_magnetization(member: SimpleNamespace) -> Expression:
    """Return the `z` component of one spin.

    Args:
        member:
            The namespace of one spin.

    Returns:
        operator:
            The local magnetisation.
    """
    return local_operator(member.psi, 'z')


def total_magnetization(s: SimpleNamespace) -> Expression:
    """Return the sum of the `z` components of the spins of the chain.

    Args:
        s:
            The namespace of the system.

    Returns:
        operator:
            The total magnetisation.
    """
    return sum(local_operator(member.psi, 'z') for member in s.chain)


@register_system(
    field='physics/statistical_mechanics',
    tags=('quantum', 'spins', 'lattice', 'magnetism', 'open system'),
    references=('Heisenberg 1928', 'Mihailović et al. 2023'),
    defaults={'until': 8.0, 'dt': 0.05, 'compiler': OperatorCompiler()},
    initial_state=lambda initial, **_: {'quantum_heisenberg.state': 'ground'} if initial == 'ground' else None,
    observables=lambda system: [
        ObservableSpec('sz', local_magnetization, group='chain', reduce=_sites),
        ObservableSpec('magnetization', total_magnetization),
    ],
)
def quantum_heisenberg(
    n_sites: int = 12,
    *,
    J: float = 1.0,
    delta: float = 1.0,
    h: float = 0.0,
    gamma: float = 0.0,
    boundary: str = 'periodic',
    initial: str = 'flip',
) -> System:
    """Return a chain of spins one half with the XXZ Hamiltonian, a field, and optional dephasing.

    The Hamiltonian is `J Σ (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j) - h Σ S^z_i` over neighbouring pairs, with
    `ħ = 1`. A dephasing rate `gamma > 0` adds the dissipator `S^z_i` on every site, so the state is a density matrix
    that loses its coherence. One flipped spin in a ferromagnetic background spreads as a magnon, whose probability at
    distance `d` is `J_d(J t)²` in the XX chain, a light cone in the space-time image of `S^z`. The total magnetisation
    is conserved.

    Args:
        n_sites:
            The number of spins.

        J:
            The exchange coupling, antiferromagnetic when positive.

        delta:
            The anisotropy `Δ`.

        h:
            The field.

        gamma:
            The dephasing rate.

        boundary:
            `'periodic'` for a ring, or `'open'` for a chain with two ends.

        initial:
            `'flip'` for one flipped spin at site `n_sites // 2`, `'neel'` for alternating spins, `'domain_wall'` for
            the left half up and the right half down, or `'ground'` for the ground state, which `run` gives the
            simulation as `initial_state={'quantum_heisenberg.state': 'ground'}`.

    Returns:
        system:
            The system, with the observables `sz` of every site, `magnetization`, and the energy.

    Raises:
        ValidationError:
            The boundary or the initial state is unknown.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import quantum_heisenberg
        >>> quantum_heisenberg(6).elements[0].prepare().size_
        6
    """
    if boundary not in BOUNDARIES or initial not in INITIAL_STATES:
        error_msg = (
            f'Unknown boundary {boundary!r} or initial state {initial!r}. Use a boundary of {", ".join(BOUNDARIES)} '
            f'and an initial state of {", ".join(INITIAL_STATES)}.'
        )
        raise ValidationError(error_msg)
    space = LatticeSpace(n_sites, boundary=BOUNDARIES[boundary])
    labels = _initial_labels(n_sites, initial)
    chain = LatticeElement('chain', element=SpinElement, space=space, values={'psi': labels})
    interactions: list[BaseInteraction] = [
        XXZExchangeInteraction('exchange', target='chain', J=J, delta=delta),
        ZeemanFieldInteraction('field', target='chain', h=h),
    ]
    if gamma > 0:
        interactions.append(DephasingInteraction('dephasing', target='chain', gamma=gamma))
    return System('quantum_heisenberg', elements=[chain], interactions=interactions)
