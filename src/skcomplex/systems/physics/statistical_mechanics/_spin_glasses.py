"""Lattice spin glasses.

The module covers Ising spins with quenched bond couplings, the Edwards-Anderson and Mattis systems.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import LatticeSpace, VonNeumannNeighborhood
from ....core.components.system import BaseSystem, System
from ....core.errors import ValidationError
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.spins import SpinSamplingInteraction

SPIN_IMAGE = VisualSpec('image', {'states': ('spin',), 'low': -1.0, 'high': 1.0})


class BondSpinElement(Element):
    """Ising spin on a square lattice with quenched couplings to its four neighbours.

    Its local energy is `-s sum J_k s_k`.

    The couplings are constant states of every site, so a bond between two sites is stored at both of them with the same
    value. A `BondSamplingInteraction` flips the spins by Metropolis Monte Carlo at the temperature `T`, with the energy
    `-sum_<ij> J_ij s_i s_j - h sum_i s_i`.

    Args:
        label:
            The label of the spin or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        spin:
            The initial spin, 1 or -1.

        J_up:
            The coupling to the upper neighbour.

        J_left:
            The coupling to the left neighbour.

        J_right:
            The coupling to the right neighbour.

        J_down:
            The coupling to the lower neighbour.

        h:
            The external field.

        temperature:
            The temperature.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import BondSpinElement
        >>> BondSpinElement('spin', J_left=-1.0).J_left
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J_up: Annotated[float, StateSpec()] = 1.0,
        J_left: Annotated[float, StateSpec()] = 1.0,
        J_right: Annotated[float, StateSpec()] = 1.0,
        J_down: Annotated[float, StateSpec()] = 1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.spin = spin
        self.J_up = J_up
        self.J_left = J_left
        self.J_right = J_right
        self.J_down = J_down
        self.h = h
        self.temperature = temperature


class MattisSpinElement(BondSpinElement):
    """Spin of the Mattis system.

    The bond spin also holds its quenched pattern `xi`, with couplings `J xi_i xi_j`.

    Args:
        label:
            The label of the spin or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        spin:
            The initial spin, 1 or -1.

        J_up:
            The coupling to the upper neighbour.

        J_left:
            The coupling to the left neighbour.

        J_right:
            The coupling to the right neighbour.

        J_down:
            The coupling to the lower neighbour.

        h:
            The external field.

        temperature:
            The temperature.

        pattern:
            The quenched pattern `xi`, 1 or -1.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import MattisSpinElement
        >>> MattisSpinElement('spin', pattern=-1.0).pattern
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        spin: float = 1.0,
        J_up: float = 1.0,
        J_left: float = 1.0,
        J_right: float = 1.0,
        J_down: float = 1.0,
        h: float = 0.0,
        temperature: float = 1.0,
        pattern: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
    ) -> None:
        super().__init__(
            label,
            visual=visual,
            spin=spin,
            J_up=J_up,
            J_left=J_left,
            J_right=J_right,
            J_down=J_down,
            h=h,
            temperature=temperature,
        )
        self.pattern = pattern


NEAREST = VonNeumannNeighborhood()


class BondSamplingInteraction(SpinSamplingInteraction):
    """Metropolis sweeps of Ising spins with quenched couplings to their neighbours.

    The local energy is `-s (sum J_k s_k + h)`.

    The couplings `J_up`, `J_left`, `J_right`, and `J_down` are constant states of every spin, in the order of the
    offsets of the von Neumann neighbourhood, and the field `h` and the temperature are its parameters.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the lattice group of spins.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import BondSamplingInteraction
        >>> BondSamplingInteraction('sampling', target='spins').neighborhood
        VonNeumannNeighborhood()
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, state='spin', energy=self._energy, neighborhood=NEAREST)

    def _energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-s (sum_k J_k s_k + h)` of a spin."""
        s = namespace.target
        couplings = (s.J_up, s.J_left, s.J_right, s.J_down)
        neighbours = neighbors(s.spin, NEAREST, reduce='each')
        return -s.spin * (sum(coupling * other for coupling, other in zip(couplings, neighbours, strict=True)) + s.h)


def _square(size: int | Sequence[int]) -> tuple[int, int]:
    """Return the shape of a square lattice of a side, or a given shape of two sides."""
    if isinstance(size, Sequence):
        rows, columns = (int(item) for item in size)
        return rows, columns
    return int(size), int(size)


def _bond_states(
    right: npt.NDArray[np.floating],
    down: npt.NDArray[np.floating],
) -> dict[str, npt.NDArray[np.floating]]:
    """Return the couplings of every site to its up, left, right, and down neighbours.

    The couplings come from the bonds right and down.
    """
    return {
        'J_up': np.roll(down, 1, axis=0),
        'J_left': np.roll(right, 1, axis=1),
        'J_right': right,
        'J_down': down,
    }


def _overlap(member: SimpleNamespace) -> Expression:
    """Return the overlap of a spin with its quenched pattern."""
    return member.spin * member.pattern


class EdwardsAndersonSystem(BaseSystem):
    """The Edwards-Anderson spin glass on a periodic square lattice.

    The bonds are quenched and random.

    Every bond takes the coupling `+J` or `-J` with equal probability, or a Gaussian coupling of mean 0 and standard
    deviation `J`, drawn once. The magnetization stays near zero at every temperature, while at low temperature the
    spins freeze in random directions and the energy per spin approaches the ground-state energy, about `-1.40 J` for
    `+J` and `-J` bonds. One unit of time is one sweep of Metropolis updates. The system has the spins `spins.spin`, the
    couplings `spins.J_right` and `spins.J_down` of the bonds to the right and down, and the observable `magnetization`.

    Args:
        size:
            The side of the square lattice, or its two sides.

        temperature:
            The temperature, which can change during a run.

        J:
            The scale of the couplings.

        h:
            The external field.

        disorder:
            `'bimodal'` for couplings `+J` and `-J`, or `'gaussian'` for Gaussian couplings.

        random_state:
            The seed or generator of the couplings and of the random initial spins.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The disorder is neither `'bimodal'` nor `'gaussian'`.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import EdwardsAndersonSystem
        >>> sorted(set(EdwardsAndersonSystem(8).prepare().system_.elements[0].values['J_right'].ravel().tolist()))
        [-1.0, 1.0]
    """

    field = 'physics/statistical_mechanics'
    topics = ('spin glass', 'disorder', 'lattice', 'Monte Carlo', 'spins')
    references = ('Edwards and Anderson 1975', 'Bertin 2021', 'Binder and Young 1986')
    until = 500
    dt = 5
    observables = (ObservableSpec('magnetization', group='spins', variable='spin'),)

    def __init__(
        self,
        size: int | Sequence[int] = 48,
        *,
        temperature: float = 0.8,
        J: float = 1.0,
        h: float = 0.0,
        disorder: str = 'bimodal',
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.temperature = temperature
        self.J = J
        self.h = h
        self.disorder = disorder
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the Edwards-Anderson spin glass on a periodic square lattice.

        The bonds are quenched and random.
        """
        shape = _square(self.size)
        rng = np.random.default_rng(self.random_state)
        if self.disorder == 'bimodal':
            right, down = self.J * rng.choice([-1.0, 1.0], size=(2, *shape))
        elif self.disorder == 'gaussian':
            right, down = self.J * rng.standard_normal((2, *shape))
        else:
            error_msg = f"The disorder is 'bimodal' or 'gaussian', not {self.disorder!r}."
            raise ValidationError(error_msg)
        spins = rng.choice([-1.0, 1.0], size=shape)
        group = LatticeElement(
            'spins',
            element=BondSpinElement,
            space=LatticeSpace(shape),
            values={'spin': spins, 'h': self.h, 'temperature': self.temperature, **_bond_states(right, down)},
            visual=SPIN_IMAGE,
        )
        return System(self.label, elements=[group], interactions=[BondSamplingInteraction('sampling', target='spins')])


class MattisSpinGlassSystem(BaseSystem):
    """The Mattis spin glass on a periodic square lattice.

    The couplings are `J_ij = J xi_i xi_j` of a pattern `xi`.

    The gauge change `sigma_i = xi_i s_i` maps the system at `h = 0` to the Ising system, so the overlap `m_xi = sum_i
    xi_i s_i / N` orders below the Ising critical temperature `2 J / ln(1 + sqrt 2)` while the magnetization averages to
    zero over the disorder. One unit of time is one sweep of Metropolis updates. The system has the spins `spins.spin`,
    the pattern `spins.pattern`, and the observables `magnetization` and `overlap`.

    Args:
        size:
            The side of the square lattice, or its two sides.

        temperature:
            The temperature, which can change during a run.

        J:
            The coupling.

        h:
            The external field.

        random_state:
            The seed or generator of the pattern and of the random initial spins.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.physics.statistical_mechanics import MattisSpinGlassSystem
        >>> values = MattisSpinGlassSystem(8).prepare().system_.elements[0].values
        >>> pattern = values['pattern'].reshape(8, 8)
        >>> bool(np.array_equal(values['J_right'].reshape(8, 8), pattern * np.roll(pattern, -1, axis=1)))
        True
    """

    field = 'physics/statistical_mechanics'
    topics = ('spin glass', 'disorder', 'lattice', 'Monte Carlo', 'spins')
    references = ('Mattis 1976', 'Bertin 2021')
    until = 300
    dt = 2
    observables = (
        ObservableSpec('magnetization', group='spins', variable='spin'),
        ObservableSpec('overlap', _overlap, group='spins'),
    )

    def __init__(
        self,
        size: int | Sequence[int] = 48,
        *,
        temperature: float = 1.5,
        J: float = 1.0,
        h: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.temperature = temperature
        self.J = J
        self.h = h
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        shape = _square(self.size)
        rng = np.random.default_rng(self.random_state)
        pattern = rng.choice([-1.0, 1.0], size=shape)
        right = self.J * pattern * np.roll(pattern, -1, axis=1)
        down = self.J * pattern * np.roll(pattern, -1, axis=0)
        spins = rng.choice([-1.0, 1.0], size=shape)
        group = LatticeElement(
            'spins',
            element=MattisSpinElement,
            space=LatticeSpace(shape),
            values={
                'spin': spins,
                'pattern': pattern,
                'h': self.h,
                'temperature': self.temperature,
                **_bond_states(right, down),
            },
            visual=VisualSpec('image', {'states': ()}),
        )
        return System(self.label, elements=[group], interactions=[BondSamplingInteraction('sampling', target='spins')])
