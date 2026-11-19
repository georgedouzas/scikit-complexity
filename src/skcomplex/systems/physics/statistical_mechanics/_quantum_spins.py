"""Quantum spin chains.

The module holds the quantum Heisenberg XXZ chain with a field and dephasing.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import qutip
from sage.all__sagemath_symbolics import CC, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws.utils import local_operator
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.quantum import DissipatorLaw, GeneratorLaw, HilbertSpec

MIN_SPIN_DIMENSION = 2
INITIAL_STATES = ('neel', 'flip', 'domain_wall', 'ground')
BOUNDARIES = {'periodic': 'periodic', 'open': 'fixed'}


def _get_spin_space(j: float) -> HilbertSpec:
    """Return the Hilbert space of a spin of quantum number `j`.

    The quantum number is a positive multiple of one half. The basis runs from `m = j` down to `m = -j`, and the
    operators are the spin components `x`, `y`, `z` and the ladder operators `plus` and `minus`. Another quantum number
    raises a `ValidationError`.
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


SPIN_HALF = _get_spin_space(0.5)


class SpinElement(Element):
    """Spins of the Hilbert space they take.

    Every spin has the spin components `x`, `y`, `z` and the ladder operators.

    The basis is ordered from `m = j` to `m = -j`, so `'up'` is index 0 and `'down'` is the last index. Laws read the
    operators of the state `psi` of a spin, such as `local_operator(s.target.psi, 'z')`. With `space`, there is one spin
    at every site of the space, as along a chain.

    Args:
        label:
            The label of the spin or of the spins.

        psi:
            The state in the Hilbert space: `'up'`, `'down'`, a basis index, or a vector, or one state per spin.

        hilbert:
            The Hilbert space of a spin, with the spin components `x`, `y`, `z` and the ladder operators `plus` and
            `minus` as its operators, by default that of a spin one half.

        space:
            The lattice of the spins, or `None` for one spin.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics._quantum_spins import SpinElement
        >>> SpinElement('s').hilbert.dimension
        2
    """

    visual = VisualSpec('expectations')

    def __init__(
        self,
        label: str,
        psi: Annotated[object, CC] = 'up',
        *,
        hilbert: HilbertSpec = SPIN_HALF,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.psi = psi
        self.hilbert = hilbert


class DephasingInteraction(Interaction):
    """Dephasing of every spin of a chain.

    The dissipator has the collapse operator `S^z_i` at the rate `gamma`.

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
        target: Role,
        gamma: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set one dissipator per spin.

        Args:
            context:
                The namespace of the spins of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [DissipatorLaw(local_operator(member.psi, 'z'), context.gamma) for member in context.target]


class XXZExchangeInteraction(Interaction):
    """Exchange between neighbouring spins.

    The exchange is `J (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j)` on every pair of neighbours of the space of the
    spins.

    Args:
        label:
            The label of the interaction.

        among:
            The path of the spins.

        J:
            The exchange coupling, antiferromagnetic when positive.

        delta:
            The anisotropy `Δ`, 1 for the isotropic chain and 0 for the XX chain.

        along:
            The pairs the exchange couples, `True` for the neighbours of the space of the spins.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import XXZExchangeInteraction
        >>> XXZExchangeInteraction('exchange', among='chain', delta=0.5).delta
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        among: Role,
        J: float = 1.0,
        delta: float = 1.0,
        along: bool | str = True,
    ) -> None:
        super().__init__(label, among=among, along=along)
        self.J = J
        self.delta = delta

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the exchange Hamiltonian of one pair of neighbours.

        Args:
            context:
                The namespace of the pair, with `source` and `target`, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            first, second = context.source.psi, context.target.psi
            exchange = (
                local_operator(first, 'x') * local_operator(second, 'x')
                + local_operator(first, 'y') * local_operator(second, 'y')
                + context.delta * local_operator(first, 'z') * local_operator(second, 'z')
            )
            self.laws_ = [GeneratorLaw(context.J * exchange)]


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

    def __init__(self, label: str, *, target: Role, h: float = 0.0) -> None:
        super().__init__(label)
        self.target = target
        self.h = h

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the Hamiltonian of the field.

        Args:
            context:
                The namespace of the spins of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [GeneratorLaw(-context.h * sum(local_operator(member.psi, 'z') for member in context.target))]


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


def _get_sites(values: object) -> object:
    """Return the values of every site, without reducing them."""
    return values


class QuantumHeisenbergSystem(BaseSystem):
    """A chain of spins one half with the XXZ Hamiltonian.

    The chain has a field and optional dephasing.

    The Hamiltonian is `J Σ (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j) - h Σ S^z_i` over neighbouring pairs, with `ħ =
    1`. A dephasing rate `gamma > 0` adds the dissipator `S^z_i` on every site, so the state is a density matrix that
    loses its coherence. One flipped spin in a ferromagnetic background spreads as a magnon, whose probability at
    distance `d` is `J_d(J t)²` in the XX chain, a light cone in the space-time image of `S^z`. The total magnetisation
    is conserved. The system has the observables `sz` of every site, `magnetization`, and the energy.

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
            the left half up and the right half down, or `'ground'` for the ground state, which the tuned run settings
            start from as `initial_state={'<label>.state': 'ground'}`.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The boundary or the initial state is unknown.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import QuantumHeisenbergSystem
        >>> len(QuantumHeisenbergSystem(6).prepare().system_.elements[0].psi)
        6
    """

    field = 'physics/statistical_mechanics'
    topics = ('quantum', 'spins', 'lattice', 'magnetism', 'open system')
    references = ('Heisenberg 1928', 'Mihailović et al. 2023')
    until = 8.0
    dt = 0.05
    observables = (
        Observable('sz', local_magnetization, group='chain', reduce=_get_sites),
        Observable('magnetization', total_magnetization),
    )

    def __init__(
        self,
        n_sites: int = 12,
        *,
        J: float = 1.0,
        delta: float = 1.0,
        h: float = 0.0,
        gamma: float = 0.0,
        boundary: str = 'periodic',
        initial: str = 'flip',
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_sites = n_sites
        self.J = J
        self.delta = delta
        self.h = h
        self.gamma = gamma
        self.boundary = boundary
        self.initial = initial

    @staticmethod
    def _get_initial_labels(n_sites: int, initial: str) -> list[str]:
        """Return the basis label of every spin of a product initial state."""
        if initial == 'neel':
            return ['up' if site % 2 == 0 else 'down' for site in range(n_sites)]
        if initial == 'flip':
            return ['down' if site == n_sites // 2 else 'up' for site in range(n_sites)]
        return ['up' if site < n_sites // 2 else 'down' for site in range(n_sites)]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the chain of spins with the exchange, the field, and the dephasing.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The boundary or the initial state is unknown.
        """
        super()._prepare(context)
        if self.boundary not in BOUNDARIES or self.initial not in INITIAL_STATES:
            error_msg = (
                f'Unknown boundary {self.boundary!r} or initial state {self.initial!r}. Use a boundary of '
                f'{", ".join(BOUNDARIES)} and an initial state of {", ".join(INITIAL_STATES)}.'
            )
            raise ValidationError(error_msg)
        space = LatticeSpace(self.n_sites, boundary=BOUNDARIES[self.boundary])
        labels = self._get_initial_labels(self.n_sites, self.initial)
        chain = SpinElement('chain', psi=labels, space=space)
        interactions: list[BaseInteraction] = [
            XXZExchangeInteraction('exchange', among='chain', J=self.J, delta=self.delta),
            ZeemanFieldInteraction('field', target='chain', h=self.h),
        ]
        if self.gamma > 0:
            interactions.append(DephasingInteraction('dephasing', target='chain', gamma=self.gamma))
        self.system_ = System(self.label, elements=[chain], interactions=interactions)
        if self.initial == 'ground':
            self.initial_state_ = {f'{self.label}.state': 'ground'}
