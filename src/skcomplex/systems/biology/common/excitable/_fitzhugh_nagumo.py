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
from .....core.types import Role


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
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.biology.common.excitable import FitzHughNagumoInteraction
        >>> kinetics = FitzHughNagumoInteraction('kinetics', target='cell', I=0.5)
        >>> cell = Element('cell', values={'v': -1.2, 'w': -0.6})
        >>> system = System(elements=[cell], interactions=[kinetics]).prepare()
        >>> trajectory = system.simulate(100.0, dt=0.1)
        >>> bool(trajectory.get_values('cell.v').max() > 1.5)
        True
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        fast: str = 'v',
        slow: str = 'w',
        a: float = 0.7,
        b: Annotated[float, RealSet.closed_open(0, oo)] = 0.8,
        epsilon: Annotated[float, RealSet.open(0, oo)] = 0.08,
        I: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.fast = fast
        self.slow = slow
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.I = I

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the cubic fast kinetics and the slow linear recovery.

        Args:
            context:
                The namespace of the target.

        Raises:
            ValidationError:
                A state name is not a name, or the fast and the slow state are the same.
        """
        super()._prepare(context)
        names = (self.fast, self.slow)
        if not all(isinstance(name, str) and name.isidentifier() for name in names) or self.fast == self.slow:
            error_msg = f'{self.label!r} reads the states {names!r}. Give two different names of states.'
            raise ValidationError(error_msg)
        if isinstance(context, SimpleNamespace):
            v, w, t = getattr(context.target, self.fast), getattr(context.target, self.slow), context.t
            self.laws_: list[BaseLaw] = [
                RateLaw(v.diff(t), v - v**3 / 3 - w + context.I),
                RateLaw(w.diff(t), context.epsilon * (v + context.a - context.b * w)),
            ]
