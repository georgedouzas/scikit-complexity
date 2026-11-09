"""Lenia: a continuous automaton of ring-kernel convolutions, and its expansion.

The expanded universe adds kernels and channels.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, exp, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import both, clip, convolve, displacement, where
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ParameterError
from ....core.specs import ObservableSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import AnyArray, Array


class LeniaSiteElement(Element):
    """Site of a Lenia world, with a value in `[0, 1]` for each of its channels.

    A world has up to three channels. The updates of the world change the channels, and a world of one channel leaves
    `B` and `C` at 0.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial value of the first channel.

        B:
            The initial value of the second channel.

        C:
            The initial value of the third channel.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LeniaSiteElement
        >>> LeniaSiteElement('site', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        B: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.B = B
        self.C = C


type RingKernel = tuple[str, float, float, Sequence[float], float, float]


class LeniaUpdateInteraction(Interaction):
    """Update of a channel of a Lenia world by the growth maps of convolutions.

    The convolutions are of channels with ring kernels.

    Every kernel `(source, radius, h, beta, mu, sigma)` reads the state of its source channel, and its concentric
    rings peak at `beta`. At the distance `q` in units of its radius, with `B` rings, it is `beta_i K_C(B q - i)` on
    the ring `i = floor(B q)`, with the bump `K_C(x) = exp(4 - 1 / (x (1 - x)))`, normalised to a sum of 1 over the
    sites. Its growth map is `G(u) = 2 exp(-(u - mu)^2 / (2 sigma^2)) - 1`, and one step adds the growths weighted by
    `h` over the sum of the weights and clips the channel, `A <- [A + sum_k (h_k / h) G_k(K_k * A_k) / T]` in
    `[0, 1]`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the world.

        channel:
            The name of the state of the channel that the update changes.

        kernels:
            The kernels `(source, radius, h, beta, mu, sigma)`, with the name of the state of the source channel and
            the radius in sites.

        T:
            The number of steps per unit time.

        n_dim:
            The number of dimensions of the world.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LeniaUpdateInteraction
        >>> LeniaUpdateInteraction('update', target='world').T
        10.0
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        channel: str = 'A',
        kernels: Sequence[RingKernel] = (('A', 13.0, 1.0, (1.0,), 0.15, 0.015),),
        T: float = 10.0,
        n_dim: int = 2,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.channel = channel
        self.kernels = kernels
        self.T = T
        self.n_dim = n_dim

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the channel by the kernels.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        distance = sqrt(sum(displacement(axis) ** 2 for axis in range(self.n_dim)))
        total = sum(kernel[2] for kernel in self.kernels)
        growth: Expression = 0
        for source, radius, weight, peaks, mu, sigma in self.kernels:
            kernel = self._shell(distance / radius, peaks) / self._norm(radius, peaks)
            potential = convolve(getattr(s.target, source), kernel, math.ceil(radius))
            growth += weight / total * (2 * exp(-((potential - mu) ** 2) / (2 * sigma**2)) - 1)
        value = getattr(s.target, self.channel)
        return [UpdateLaw(value, clip(value + growth / self.T, 0, 1))]

    @staticmethod
    def _shell(q: Expression, peaks: Sequence[float]) -> Expression:
        """Return the rings of a kernel at the distance `q` in units of its radius.

        They are zero outside the unit ball.
        """
        rings = len(peaks)
        shell: Expression = 0
        for index, peak in enumerate(peaks):
            x = rings * q - index
            inside = both(x > 0, x < 1)
            shell += where(inside, peak * exp(4 - 1 / where(inside, x * (1 - x), 1)), 0)
        return shell

    def _norm(self, radius: float, peaks: Sequence[float]) -> float:
        """Return the sum of the rings of a kernel over the sites within its radius."""
        steps = np.arange(-math.ceil(radius), math.ceil(radius) + 1)
        grid = np.meshgrid(*[steps] * self.n_dim, indexing='ij')
        q: Array = np.sqrt(sum(axis**2 for axis in grid)) / radius
        rings = len(peaks)
        values = np.zeros_like(q)
        for index, peak in enumerate(peaks):
            x = rings * q - index
            inside = (x > 0) & (x < 1)
            values += np.where(inside, peak * np.exp(4 - 1 / np.where(inside, x * (1 - x), 1)), 0)
        return float(values.sum())


type Kernel = tuple[int, int, float, float, Sequence[float], float, float]

ORBIUM = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0.05, 0.02, 0, 0, 0, 0, 0, 0, 0.24, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.14, 0.21, 0.21, 0.08, 0.07, 0.1, 0.1, 0.07, 0, 0.25, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.09, 0.27, 0.36, 0.38, 0.34, 0.19, 0.18, 0.17, 0.18, 0.18, 0.11, 0.35, 0, 0, 0],
        [0, 0, 0, 0, 0.01, 0.07, 0.35, 0.45, 0.47, 0.38, 0.13, 0.08, 0.05, 0.06, 0.12, 0.22, 0.71, 0, 0, 0],
        [0, 0, 0, 0.04, 0.13, 0.18, 0.34, 0.4, 0.38, 0.28, 0.13, 0, 0, 0, 0, 0.05, 0.4, 0.33, 0, 0],
        [0, 0, 0.02, 0.14, 0.17, 0.13, 0.11, 0.25, 0.27, 0.26, 0.21, 0, 0, 0, 0, 0, 0.02, 0.86, 0, 0],
        [0.24, 0, 0.1, 0.17, 0.08, 0, 0, 0.2, 0.31, 0.37, 0.37, 0.24, 0, 0, 0, 0, 0, 0.45, 0.16, 0],
        [0, 0.12, 0.18, 0.11, 0, 0, 0, 0.27, 0.43, 0.51, 0.54, 0.51, 0, 0, 0, 0, 0, 0, 0.52, 0],
        [0, 0.59, 0.22, 0.03, 0, 0, 0, 0.19, 0.53, 0.64, 0.68, 0.68, 0.42, 0, 0, 0, 0, 0, 0.43, 0],
        [0, 0.55, 0.24, 0, 0, 0, 0, 0.03, 0.62, 0.76, 0.84, 0.85, 0.81, 0.11, 0, 0, 0, 0, 0.28, 0.1],
        [0, 0, 0.6, 0, 0, 0, 0, 0, 0.67, 0.87, 0.96, 0.99, 0.97, 0.56, 0, 0, 0, 0, 0.22, 0.14],
        [0, 0, 0.84, 0, 0, 0, 0, 0, 0.51, 0.97, 1, 1, 1, 0.91, 0.28, 0, 0, 0.02, 0.22, 0.13],
        [0, 0, 0.55, 0.07, 0, 0, 0, 0, 0.33, 1, 1, 0.98, 1, 0.97, 0.56, 0.16, 0.05, 0.12, 0.24, 0.09],
        [0, 0, 0.03, 0.45, 0, 0, 0, 0, 0.18, 0.85, 1, 0.89, 0.86, 0.87, 0.65, 0.33, 0.2, 0.22, 0.23, 0.04],
        [0, 0, 0, 0.42, 0.11, 0, 0, 0, 0.11, 0.62, 0.88, 0.82, 0.78, 0.74, 0.61, 0.41, 0.29, 0.27, 0.16, 0],
        [0, 0, 0, 0.08, 0.33, 0.07, 0, 0, 0.12, 0.42, 0.67, 0.72, 0.68, 0.63, 0.53, 0.4, 0.31, 0.22, 0.06, 0],
        [0, 0, 0, 0, 0.16, 0.26, 0.15, 0.12, 0.17, 0.34, 0.5, 0.56, 0.55, 0.5, 0.43, 0.34, 0.24, 0.12, 0, 0],
        [0, 0, 0, 0, 0, 0.15, 0.23, 0.24, 0.26, 0.32, 0.39, 0.42, 0.41, 0.36, 0.31, 0.23, 0.13, 0.02, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.07, 0.17, 0.23, 0.25, 0.27, 0.28, 0.26, 0.23, 0.16, 0.09, 0.02, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0.1, 0.13, 0.13, 0.12, 0.08, 0.04, 0, 0, 0, 0, 0],
    ],
)
"""The cells of Orbium unicaudatus, the gliding soliton of Lenia, from the pattern library of Chan, rounded."""

MIN_AXES = 2


CHANNELS = ('A', 'B', 'C')


def _mass(values: AnyArray) -> float:
    """Return the total mass of a channel, the sum of the values of its sites."""
    return float(np.sum(np.asarray(values, dtype=float)))


class LeniaSystem(BaseSystem):
    """Lenia, a continuous automaton whose ring-kernel convolutions grow solitons.

    The solitons are gliding and lifelike. The world is a torus of sites with values in `[0, 1]`, and one step of
    `1 / T` adds the growth map of the convolution of the world with a ring kernel and clips the result,
    `A <- [A + G(K * A) / T]` in `[0, 1]`, with `G(u) = 2 exp(-(u - mu)^2 / (2 sigma^2)) - 1` and the kernel of radius
    `R`, whose concentric rings peak at `beta`. The expanded universe of Chan (2020) has several channels and several
    kernels `(source, destination, r, h, beta, mu, sigma)`, each of radius `r R` with its own growth map, which reads
    its source channel and grows its destination with the weight `h` over the sum of the weights into that destination,
    `A_j <- [A_j + sum_k (h_k / h_j) G_k(K_k * A_i) / T]`, on worlds of three or more dimensions. By default the world
    holds Orbium unicaudatus, which glides across the torus at a steady speed while its mass stays near 71. Every step
    of the system is one update, so a unit of Lenia's time takes `T` steps. The system has the values `world.A`,
    `world.B`, and `world.C` of the three channels of the sites, the last two at 0 for a world of one channel. Its
    standard observables `mass_A`, `mass_B`, and `mass_C` are the totals of the values of every channel.

    Args:
        shape:
            The number of sites along every axis of the torus, two or more axes.

        initial:
            The initial values, an array of the shape shared by every channel, or one per channel, or `None` for
            Orbium in the middle of a two-dimensional world.

        R:
            The radius of the kernels in sites.

        T:
            The number of steps per unit time, the inverse of the increment of the update.

        mu:
            The centre of the growth map of the single kernel.

        sigma:
            The width of the growth map of the single kernel.

        beta:
            The peaks of the concentric rings of the single kernel.

        channels:
            The number of channels of the world, from one to three.

        kernels:
            The kernels `(source, destination, r, h, beta, mu, sigma)` of the expanded universe, with the channels as
            indices, or `None` for the single kernel of `mu`, `sigma`, and `beta` on the first channel.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ParameterError:
            The world has fewer than two axes or not one to three channels, Orbium does not fit a default world, the
            initial values do not have the shape, or a kernel names a channel the world does not have.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LeniaSystem
        >>> LeniaSystem().prepare().system_.elements[0].values['A'].shape
        (64, 64)
    """

    field = 'computer_science/automata'
    topics = ('lattice', 'continuous automaton', 'convolution', 'soliton', 'artificial life', 'field')
    references = ('Chan 2019', 'Chan 2020')
    until = 200
    dt = 1
    observables = tuple(
        ObservableSpec(f'mass_{channel}', group='world', variable=channel, reduce=_mass) for channel in CHANNELS
    )

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        initial: npt.ArrayLike | None = None,
        R: float = 13.0,
        T: float = 10.0,
        mu: float = 0.15,
        sigma: float = 0.015,
        beta: Sequence[float] = (1.0,),
        channels: int = 1,
        kernels: Sequence[Kernel] | None = None,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.initial = initial
        self.R = R
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.channels = channels
        self.kernels = kernels

    @staticmethod
    def _get_initial_values(shape: tuple[int, ...], channels: int, initial: npt.ArrayLike | None) -> AnyArray:
        """Return the initial values of every channel, one array of the shape per channel."""
        if initial is None:
            if len(shape) != MIN_AXES or min(shape) < ORBIUM.shape[0]:
                error_msg = f'Orbium needs a two-dimensional world of at least {ORBIUM.shape[0]} sites a side.'
                raise ParameterError(error_msg)
            values = np.zeros(shape)
            top, left = (shape[0] - ORBIUM.shape[0]) // 2, (shape[1] - ORBIUM.shape[1]) // 2
            values[top : top + ORBIUM.shape[0], left : left + ORBIUM.shape[1]] = ORBIUM
        else:
            values = np.asarray(initial, dtype=float)
        if values.shape == shape:
            return np.broadcast_to(values, (channels, *shape)).copy()
        if values.shape != (channels, *shape):
            error_msg = f'The initial values have the shape {values.shape}, but the world has {channels} of {shape}.'
            raise ParameterError(error_msg)
        return values

    def _build_system(self) -> CompositeSystem:
        """Return Lenia, a continuous automaton of ring-kernel convolutions.

        Its convolutions grow solitons. The solitons are gliding and lifelike.
        """
        shape = tuple(self.shape)
        if len(shape) < MIN_AXES or not 1 <= self.channels <= len(CHANNELS):
            error_msg = (
                f'Lenia needs two or more axes and one to three channels, but got {shape!r} and {self.channels!r}.'
            )
            raise ParameterError(error_msg)
        values = self._get_initial_values(shape, self.channels, self.initial)
        rings = (
            ((0, 0, 1.0, 1.0, tuple(self.beta), self.mu, self.sigma),) if self.kernels is None else tuple(self.kernels)
        )
        if any(not 0 <= index < self.channels for kernel in rings for index in kernel[:2]):
            error_msg = f'Every kernel needs a source and a destination among the {self.channels} channels.'
            raise ParameterError(error_msg)
        world = FieldElement(
            'world',
            element=LeniaSiteElement,
            space=LatticeSpace(shape),
            values={CHANNELS[index]: values[index] for index in range(self.channels)},
        )
        updates = [
            LeniaUpdateInteraction(
                f'update_{CHANNELS[index]}',
                target='world',
                channel=CHANNELS[index],
                kernels=[
                    (CHANNELS[source], radius * self.R, weight, tuple(peaks), centre, width)
                    for source, destination, radius, weight, peaks, centre, width in rings
                    if destination == index
                ],
                T=self.T,
                n_dim=len(shape),
            )
            for index in range(self.channels)
            if any(kernel[1] == index for kernel in rings)
        ]
        return CompositeSystem(self.label, elements=[world], interactions=updates, time=DiscreteTime())
