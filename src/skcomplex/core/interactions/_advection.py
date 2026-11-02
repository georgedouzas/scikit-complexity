"""Define advection, the conservation law whose flux is a drift field times the state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR
from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RoleSpec
from ._conservation import ConservationInteraction


class AdvectionInteraction(ConservationInteraction):
    """Advection `du/dt + divergence(a u) = source(ns)` of a state `u` of a field by a drift field `a(ns)`.

    The flux of the conservation law is the drift times the state, written in conservative form, so the total of the
    state on a periodic field stays constant and, on a spectral field, the dealiasing of products applies to it. The
    drift is one expression for a field of one dimension, or one per dimension, of the states and parameters of the
    field, and may read other states than the one it carries, such as the flow of a fluid derived from a
    streamfunction, or the drift of a probability density in a Fokker-Planck equation.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field whose state is carried.

        state:
            The name of the carried state of the field.

        drift:
            A function of the namespace that returns the drift, or one drift per dimension.

        source:
            A function of the namespace that returns the source, or `None` for none.

        limiter:
            The slope limiter of the reconstruction of a finite-volume field, `'minmod'`, `'van-leer'`, or `None`.

        riemann:
            The numerical flux of a finite-volume field, `'rusanov'` or `'godunov'`, or `None` for the divergence of
            the flux on a field of another discretization.

        sources:
            The paths of other elements the drift and the source read, as the namespaces `sources`. A field over the
            position lattice of a target on a phase space is read broadcast along the class axes.

    Raises:
        ValidationError:
            The state is not a name, a function is not callable, or the limiter or the numerical flux is unknown.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.elements import Element, FieldElement
        >>> from skcomplex.core.interactions import AdvectionInteraction
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.specs import StateSpec
        >>> class CellElement(Element):
        ...     def __init__(self, label, u: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.u = u
        >>> space = LatticeSpace((50,), spacing=0.1, boundary='periodic')
        >>> values = {'u': np.exp(-((np.arange(50) - 25) / 5.0) ** 2)}
        >>> volumes = 'finite-volume'
        >>> field = FieldElement('tracer', element=CellElement, space=space, values=values, discretization=volumes)
        >>> drift = AdvectionInteraction('drift', target='tracer', state='u', drift=lambda s: 1.0)
        >>> system = System(elements=[field], interactions=[drift])
        >>> trajectory = Simulation(system, dt=0.01).check().prepare().run(until=1.0).trajectory_
        >>> total = trajectory.values('tracer.u')[-1].sum()
        >>> round(float(total), 6) == round(float(values['u'].sum()), 6)
        True
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'u',
        drift: Callable[[SimpleNamespace], object] | None = None,
        source: Callable[[SimpleNamespace], object] | None = None,
        limiter: str | None = 'minmod',
        riemann: str | None = 'rusanov',
        sources: Annotated[Sequence[str], RoleSpec(variadic=True)] = (),
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            source=source,
            limiter=limiter,
            riemann=riemann,
            sources=sources,
        )
        self.drift = drift

    def _check(self) -> None:
        """Raise an error for a state that is not a name, a drift that is not callable, or an unknown scheme.

        Raises:
            ValidationError:
                The type declares states, the state is not a name, a function is not callable, or the limiter or the
                numerical flux is unknown.
        """
        super()._check()
        if not callable(self.drift):
            error_msg = f'The drift of {self.label!r} is {self.drift!r}. Give a function of the namespace.'
            raise ValidationError(error_msg)

    def _get_flux(self, namespace: SimpleNamespace) -> object:
        """Return the flux `a u`, one component per component of the drift.

        Args:
            namespace:
                The namespace of the symbols the flux can use, with the namespace of the field.

        Returns:
            flux:
                The flux, one expression or one per dimension.
        """
        state = getattr(self._get_member(namespace, 'target'), self.state)
        drift = self.drift(namespace) if callable(self.drift) else None
        if isinstance(drift, FreeModuleElement | list | tuple):
            return [SR(component) * state for component in drift]
        return SR(drift) * state
