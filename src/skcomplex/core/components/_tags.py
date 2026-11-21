"""Define the capability tags of components, systems, solvers, and backends."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from ..base import BaseSpec
from ..enums import DomainTag, LawTag, RandomnessTag, SpaceTag, StructureTag, TimeTag
from ..errors import ValidationError

STOCHASTIC_LAWS = frozenset({LawTag.NOISE, LawTag.JUMP, LawTag.SAMPLING})
DIFFERENTIABLE_LAWS = frozenset({LawTag.RATE, LawTag.UPDATE, LawTag.EQUATION})


@dataclass(frozen=True)
class Tags(BaseSpec):
    """A set of tags that say what a component, a solver, or a backend handles.

    They say the times a component runs in, the domains of its values, its spaces, the kinds of its laws, whether it
    draws random numbers, and whether its structure changes, such as a continuous time and rate and noise laws for a
    stochastic differential equation. They come from the arguments and annotations of a component alone, so they are
    known before `prepare`, and a system compares them with those of a solver to pick one that runs it. An empty set
    means any value, and a law kind or randomness of `None` means the component does not show it. They match an
    epidemic, a market, or a magnet with a solver that runs it.

    Read more in the user guide, under Solvers.

    Args:
        time:
            The times the component runs in, among `'continuous'`, `'discrete'`, and `'static'`.

        domains:
            The domains of the states, among `'real'`, `'integer'`, `'boolean'`, `'categorical'`, `'complex'`,
            `'vector'`, `'matrix group'`, and `'power series'`.

        space:
            The spaces, among `'none'`, `'continuous'`, `'lattice'`, `'network'`, `'hypergraph'`, `'cell complex'`,
            `'phase space'`, and `'curved spacetime'`.

        laws:
            The kinds of the laws, among `'rate'`, `'noise'`, `'jump'`, `'update'`, `'equation'`, `'objective'`, and
            `'sampling'`, or `None` when the arguments do not show them.

        randomness:
            `'deterministic'`, `'stochastic'`, or `None` when the arguments do not show it.

        structure:
            How the structure changes, among `'variable membership'` and `'changing topology'`, empty for a fixed one.

        reparameterized:
            Whether every random draw of a stochastic component is a function of a uniform or normal number of the step,
            so its runs have gradients at fixed random numbers.

    See Also:
        BaseComponent: Gives its tags through `get_tags`.

    Examples:
        >>> from skcomplex.core.components import Tags
        >>> tags = Tags(time=frozenset({'continuous'}), laws=frozenset({'rate', 'noise'}), randomness='stochastic')
        >>> Tags(time=frozenset({'continuous'}), laws=frozenset({'rate'})).get_missing(tags)
        {'laws': ['noise']}
    """

    time: frozenset[str] = frozenset()
    domains: frozenset[str] = frozenset()
    space: frozenset[str] = frozenset()
    laws: frozenset[str] | None = frozenset()
    randomness: str | None = 'deterministic'
    structure: frozenset[str] = frozenset()
    reparameterized: bool = False

    def _check(self) -> None:
        """Raise an error for a tag that is none of the values of its field.

        Raises:
            ValidationError:
                A tag is unknown, named in the message with the values of its field.
        """
        fields = {
            'time': (self.time, TimeTag),
            'domains': (self.domains, DomainTag),
            'space': (self.space, SpaceTag),
            'laws': (self.laws or frozenset(), LawTag),
            'randomness': (frozenset() if self.randomness is None else {self.randomness}, RandomnessTag),
            'structure': (self.structure, StructureTag),
        }
        for name, (values, known) in fields.items():
            unknown = sorted(set(values) - set(known))
            if unknown:
                accepted = ', '.join(repr(str(item)) for item in known)
                error_msg = f'The {name} tag {unknown[0]!r} is unknown. Use one of {accepted}.'
                raise ValidationError(error_msg)

    @property
    def differentiable(self) -> bool:
        """Whether runs of the component can be differentiated in its parameters.

        It holds for rate, update, and equation laws, which the JAX backend lowers to pure functions, when they are
        deterministic or their draws are reparameterized, so the JAX run draws them from a key.
        """
        randomness = self.randomness == 'deterministic' or (self.randomness == 'stochastic' and self.reparameterized)
        return randomness and self.laws is not None and bool(self.laws) and self.laws <= DIFFERENTIABLE_LAWS

    def get_missing(self, tags: Tags) -> dict[str, list[str]]:
        """Return the tags of a system that these supported tags do not support, by field.

        A field of the supported tags that is empty supports any value. The system needs one of its times to be
        supported, and every one of its spaces, domains, and law kinds. Law kinds that are `None` are not checked.

        Args:
            tags:
                The tags of a system.

        Returns:
            missing:
                The unsupported tags of every field that has some, sorted.
        """
        found: dict[str, list[str]] = {}
        if self.time and tags.time and not self.time & tags.time:
            found['time'] = sorted(tags.time)
        needed = {'space': (self.space, tags.space), 'domains': (self.domains, tags.domains)}
        needed['laws'] = (self.laws or frozenset(), tags.laws or frozenset())
        for name, (supported, values) in needed.items():
            if supported and values - supported:
                found[name] = sorted(values - supported)
        return found
