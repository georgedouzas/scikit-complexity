"""Share the helpers of the compilation tests."""

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import I

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class PhaseElement(Element):
    """Complex amplitude `z` with the real frequency `omega`."""

    def __init__(
        self,
        label: str,
        *,
        z: complex = 1.0,
        omega: float = 2.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.z = z
        self.omega = omega


class QuadraticElement(Element):
    """Complex point `z` with the complex constant `c` of the quadratic map."""

    def __init__(
        self,
        label: str,
        *,
        z: complex = 0.0,
        c: complex = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.z = z
        self.c = c


class PairNoiseInteraction(Interaction):
    """Noise of the two values `x` and `y` of an element, each scaled by the other, on one local process."""

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'pair',
        process: str = 'common',
        conventions: tuple[str, str] = ('ito', 'ito'),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.process = process
        self.conventions = conventions

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise of both values on the shared process."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            first, second = self.conventions
            pair = context.target
            self.laws_: list[BaseLaw] = [
                NoiseLaw(pair.x, pair.sigma * pair.y, convention=first, process=self.process),
                NoiseLaw(pair.y, pair.sigma * pair.x, convention=second, process=self.process),
            ]


def pair_system(size: int | None = None, **settings: object) -> System:
    """Return a pair, or a group of pairs, whose values `x` and `y` take noise on one local process."""
    values: dict[str, object] = {'x': 1.0, 'y': 1.0, 'sigma': 0.5}
    if size is not None:
        values = {name: np.full(size, value) for name, value in values.items()}
    pair = Element('pair', values=values, size=size)
    return System(elements=[pair], interactions=[PairNoiseInteraction('noise', **settings)])


def phase_system(z: complex = 1.0 + 0j, omega: object = 2.0, **settings: object) -> System:
    """Return a complex amplitude `osc.z` that turns as `dz/dt = i omega z`."""
    turn = Interaction(
        'turn',
        roles={'target': 'osc'},
        laws=lambda s: [RateLaw(s.target.z, I * s.target.omega * s.target.z)],
    )
    return System(elements=[PhaseElement('osc', z=z, omega=omega)], interactions=[turn], **settings)


def quadratic_system(element: Element, **settings: object) -> System:
    """Return the complex quadratic map `z -> z^2 + c` of an element with the values `z` and `c`."""
    square = Interaction(
        'square',
        roles={'target': element.label},
        laws=lambda s: [UpdateLaw(s.target.z, s.target.z**2 + s.target.c)],
    )
    return System(elements=[element], interactions=[square], **settings)
