"""Define the interaction that changes a field by a flux and a source."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

from sage.modules.free_module_element import FreeModuleElement

from ...errors import ValidationError
from ...specs import RoleSpec, Tags
from ...types import ElementKind
from ..base import BasePart
from ..laws import BaseLaw, FluxLaw, RateLaw
from ..laws.utils import divergence
from ..spaces import BaseSpace
from ._base import RATE_DOMAINS
from ._interaction import Interaction

LIMITERS = ('minmod', 'van-leer', None)
RIEMANN_SOLVERS = ('rusanov', 'godunov', None)


class ConservationInteraction(Interaction):
    """An interaction that changes a field by the divergence of a flux and a source.

    It gives the conservation law `du/dt + divergence(flux) = source` of a state `u` of a field. The flux is one
    expression for a field of one dimension, or one per dimension. On a finite-volume field the interaction lowers to a
    flux law with a slope limiter and a numerical flux, and on other fields to the rate `-divergence(flux)`, which the
    field discretizes by its own operators. Without a source the total of the state on a periodic field stays constant.
    Several conservation laws on one field form one system, whose numerical flux reads the wave speeds of all of them. A
    flux or a source function takes a namespace with the field as `target`, the time `t`, and the parameters a subclass
    declares.

    Unlike `RateInteraction`, which gives the rate of a state directly, it writes the rate as the divergence of a flux,
    so the scheme keeps the total and captures shocks. Choose it when an amount moves through space without being
    created or destroyed, apart from a source.

    It appears in many fields. Cars on a road follow the traffic flow model of Lighthill and Whitham, a gas follows the
    Euler equations, and water in a channel follows the shallow water equations.

    Args:
        label:
            The name of the interaction, unique among the parts of its system, such as `'traffic'`.

        target:
            The path of the field whose state is conserved.

        state:
            The name of the conserved state of the field.

        flux:
            A function of the namespace that returns the flux, or one flux per dimension.

        source:
            A function of the namespace that returns the source, or `None` for none.

        limiter:
            The slope limiter of the reconstruction, `'minmod'`, `'van-leer'`, or `None` for first order.

        riemann:
            The numerical flux of a finite-volume field, `'rusanov'` or `'godunov'`, the exact solver for one state, or
            `None` for the divergence of the flux on a field of another discretization.

        sources:
            The paths of other elements the flux and the source read, as the namespaces `sources`. A field over the
            position lattice of a target on a phase space is read broadcast along the class axes.

    Raises:
        ValidationError:
            The state is not a name, a function is not callable, or the limiter or the numerical flux is unknown.

    Examples:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> from skcomplex.core.components.elements import Element, FieldElement
        >>> from skcomplex.core.components.interactions import ConservationInteraction
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.engine.runtime import simulate
        >>> class CellElement(Element):
        ...     def __init__(self, label, u: Annotated[float, StateSpec()] = 0.0):
        ...         super().__init__(label)
        ...         self.u = u
        >>> space = LatticeSpace((50,), spacing=0.1, boundary='periodic')
        >>> values = {'u': 1.0 + np.sin(2 * np.pi * np.arange(50) / 50)}
        >>> volumes = 'finite-volume'
        >>> field = FieldElement('wave', element=CellElement, space=space, values=values, discretization=volumes)
        >>> burgers = ConservationInteraction('burgers', target='wave', state='u', flux=lambda s: s.target.u**2 / 2)
        >>> system = CompositeSystem(elements=[field], interactions=[burgers])
        >>> trajectory = simulate(system, until=0.5, dt=0.01)
        >>> round(float(trajectory.get_values('wave.u')[-1].sum()), 6)
        50.0
    """

    law_classes: ClassVar[tuple[type[BaseLaw], ...]] = (FluxLaw, RateLaw)

    def _get_tags(self) -> Tags:
        """Return the tags of a flux with a Riemann solver or of a rate.

        A source adds a rate.
        """
        transport = (FluxLaw,) if self._get_argument('riemann') is not None else (RateLaw,)
        source = () if self._get_argument('source') is None else (RateLaw,)
        return self._get_variant_tags((*transport, *source))

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec(elements=(ElementKind.ANY,), states=('state',), domains=RATE_DOMAINS)],
        state: str = 'u',
        flux: Callable[[SimpleNamespace], object] | None = None,
        source: Callable[[SimpleNamespace], object] | None = None,
        limiter: str | None = 'minmod',
        riemann: str | None = 'rusanov',
        sources: Annotated[Sequence[str], RoleSpec(variadic=True, elements=(ElementKind.ANY,))] = (),
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.flux = flux
        self.source = source
        self.limiter = limiter
        self.riemann = riemann
        self.sources = sources

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an invalid state, function, or scheme.

        It raises it for a state that is not a name, a function that is not callable, or an unknown scheme.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The type declares states, the state is not a name, a function is not callable, or the limiter or the
                numerical flux is unknown.
        """
        super()._prepare(context)
        if not isinstance(self.state, str) or not self.state.isidentifier():
            error_msg = f'{type(self).__name__} {self.label!r} conserves the state {self.state!r}, which is no name.'
            raise ValidationError(error_msg)
        for name in ('flux', 'source'):
            value = getattr(self, name)
            if value is not None and not callable(value):
                error_msg = f'The {name} of {self.label!r} is {value!r}. Give a function of the namespace.'
                raise ValidationError(error_msg)
        if self.limiter not in LIMITERS:
            error_msg = f'Unknown limiter {self.limiter!r} of {self.label!r}. Use minmod, van-leer, or None.'
            raise ValidationError(error_msg)
        if self.riemann not in RIEMANN_SOLVERS:
            error_msg = f'Unknown numerical flux {self.riemann!r} of {self.label!r}. Use rusanov, godunov, or None.'
            raise ValidationError(error_msg)

    def _get_flux(self, namespace: SimpleNamespace) -> object:
        """Return the flux, or `None` for none.

        This is the hook a subclass overrides to derive the flux from its arguments.
        """
        return None if self.flux is None else self.flux(namespace)

    def _get_transport_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the laws of the divergence of the flux.

        This is the hook a subclass overrides to lower its own flux.
        """
        flux = self._get_flux(namespace)
        if flux is None:
            return []
        state = getattr(self._get_member(namespace, 'target'), self.state)
        if self.riemann is None:
            components = list(flux) if isinstance(flux, FreeModuleElement | list | tuple) else [flux]
            return [RateLaw(state, -divergence(components))]
        return [FluxLaw(state, flux, limiter=self.limiter, riemann=self.riemann)]

    def _get_laws(self, namespace: SimpleNamespace) -> list[BasePart]:
        """Return the laws of the divergence of the flux and the rate of the source.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with the namespace of the field.

        Returns:
            laws:
                The flux law, and the rate law of the source when there is one.
        """
        laws = self._get_transport_laws(namespace)
        if self.source is not None:
            laws.append(RateLaw(getattr(self._get_member(namespace, 'target'), self.state), self.source(namespace)))
        return laws
