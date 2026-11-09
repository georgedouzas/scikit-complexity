"""Define FitzHugh-Nagumo kinetics.

They are a fast variable with a cubic nullcline and a slow linear recovery.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import BaseLaw, RateLaw
from .....core.components.spaces import BaseSpace
from .....core.errors import ValidationError
from .....core.specs import ParameterSpec, RoleSpec


class FitzHughNagumoInteraction(Interaction):
    """FitzHugh-Nagumo kinetics of a fast variable `v` and a slow recovery `w`.

    The fast variable and the recovery are states of a target.

    The fast variable follows `v' = v - v³/3 - w + I` and the recovery `w' = epsilon (v + a - b w)`. On a field the
    kinetics act on every cell, and a diffusion of the fast variable makes the medium excitable.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the element or field whose states follow the kinetics.

        fast:
            The name of the fast state, such as a membrane potential.

        slow:
            The name of the slow recovery state.

        a:
            The offset of the recovery.

        b:
            The damping of the recovery.

        epsilon:
            The time scale of the recovery.

        I:
            The applied current.

    Raises:
        ValidationError:
            A state name is not a name, or the fast and the slow state are the same.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.systems.biology.common.excitable import FitzHughNagumoInteraction
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class CellElement(Element):
        ...     def __init__(
        ...         self, label, visual=None, *, v: Annotated[float, StateSpec()] = -1.2,
        ...         w: Annotated[float, StateSpec()] = -0.6,
        ...     ):
        ...         super().__init__(label, visual=visual)
        ...         self.v = v
        ...         self.w = w
        >>> kinetics = FitzHughNagumoInteraction('kinetics', target='cell', I=0.5)
        >>> system = System(elements=[CellElement('cell')], interactions=[kinetics])
        >>> trajectory = simulate(system, until=100.0, dt=0.1)
        >>> bool(trajectory.values('cell.v').max() > 1.5)
        True
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        fast: str = 'v',
        slow: str = 'w',
        a: Annotated[float, ParameterSpec()] = 0.7,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.08,
        I: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.fast = fast
        self.slow = slow
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.I = I

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for state names that are not names or that coincide.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                A state name is not a name, or the fast and the slow state are the same.
        """
        super()._prepare(context)
        names = (self.fast, self.slow)
        if not all(isinstance(name, str) and name.isidentifier() for name in names) or self.fast == self.slow:
            error_msg = f'{self.label!r} reads the states {names!r}. Give two different names of states.'
            raise ValidationError(error_msg)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the cubic fast kinetics and the slow linear recovery.

        Args:
            s:
                The namespace of the symbols the laws can use, with the namespace of the target.

        Returns:
            laws:
                The rate laws of the fast and the slow state.
        """
        v, w = getattr(s.target, self.fast), getattr(s.target, self.slow)
        return [RateLaw(v, v - v**3 / 3 - w + s.I), RateLaw(w, s.epsilon * (v + s.a - s.b * w))]
