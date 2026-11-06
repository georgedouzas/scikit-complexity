"""Share the helpers of the compilation tests."""

from types import SimpleNamespace
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import BaseLaw, NoiseLaw
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


class PairElement(Element):
    """Element whose two states take unit noise on one local process."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec()] = 0.5,
        process: str = 'common',
        conventions: tuple[str, str] = ('ito', 'ito'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.sigma = sigma
        self.process = process
        self.conventions = conventions

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the noise of both states on the shared process."""
        first, second = self.conventions
        return [
            NoiseLaw(s.x, s.sigma * s.y, convention=first, process=self.process),
            NoiseLaw(s.y, s.sigma * s.x, convention=second, process=self.process),
        ]
