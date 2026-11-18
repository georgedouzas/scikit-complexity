"""Define advection.

Advection carries a state of a field by a drift, the conservation law whose flux is the drift times the state.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from sage.all__sagemath_symbolics import SR
from sage.modules.free_module_element import FreeModuleElement

from .....core.components.interactions import Interaction
from .....core.components.laws import RateLaw
from .....core.components.laws.utils import divergence
from .....core.components.spaces import BaseSpace
from .....core.errors import ValidationError
from .....core.specs import Role


class AdvectionInteraction(Interaction):
    """Advection `du/dt + divergence(a u) = source(ns)` of a state `u` of a field by a drift.

    Physics names it advection, the transport of a state by a flow or by the streaming of a phase-space density.
    The drift field is `a(ns)`. The flux of the conservation law is the drift times the state, written in conservative
    form, so the total of the state on a periodic field stays constant and, on a spectral field, the dealiasing of
    products applies to it. The drift is one expression for a field of one dimension, or one per dimension, of the
    states and parameters of the field, and may read other states than the one it carries, such as the flow of a fluid
    derived from a streamfunction, or the drift of a probability density in a Fokker-Planck equation.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the field whose state is carried.

        state:
            The name of the carried state of the field.

        drift:
            A function of the namespace that returns the drift, or one drift per dimension.

        source:
            A function of the namespace that returns the source, or `None` for none.

    Raises:
        ValidationError:
            The state is not a name, or a function is not callable.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.fields import AdvectionInteraction
        >>> space = LatticeSpace((50,), spacing=0.1, boundary='periodic', discretization='finite-volume')
        >>> values = {'u': np.exp(-((np.arange(50) - 25) / 5.0) ** 2)}
        >>> tracer = Element('tracer', values=values, space=space)
        >>> drift = AdvectionInteraction('drift', target='tracer', state='u', drift=lambda s: 1.0)
        >>> trajectory = System(elements=[tracer], interactions=[drift]).prepare().simulate(until=1.0, dt=0.01)
        >>> total = trajectory.get_values('tracer.u')[-1].sum()
        >>> round(float(total), 6) == round(float(values['u'].sum()), 6)
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'u',
        drift: Callable[[SimpleNamespace], object] | None = None,
        source: Callable[[SimpleNamespace], object] | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.drift = drift
        self.source = source

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rate of the divergence of the flux `a u` and the rate of the source.

        Args:
            context:
                The space the interaction inherits from its system, or the namespace of the field, the time `t`, and
                the parameters.

        Raises:
            ValidationError:
                The state is not a name, the drift is not callable, or the source is neither callable nor `None`.
        """
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = f'{type(self).__name__} {self.label!r} carries the state {self.state!r}, which is no name.'
            raise ValidationError(error_msg)
        if not callable(self.drift):
            error_msg = f'The drift of {self.label!r} is {self.drift!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)
        if self.source is not None and not callable(self.source):
            error_msg = f'The source of {self.label!r} is {self.source!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            u = getattr(context.target, self.state)
            drift = self.drift(context)
            components = drift if isinstance(drift, FreeModuleElement | list | tuple) else [drift]
            self.laws_ = [RateLaw(u, -divergence([SR(component) * u for component in components]))]
            if self.source is not None:
                self.laws_.append(RateLaw(u, self.source(context)))
