"""Define the capability tags of parts, systems, solvers, and backends."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec

TIMES = ('continuous', 'discrete', 'static')
DOMAINS = ('real', 'integer', 'boolean', 'categorical', 'complex', 'vector', 'matrix group', 'power series')
SPACES = ('none', 'continuous', 'lattice', 'network', 'hypergraph', 'cell complex', 'phase space', 'curved spacetime')
LAWS = ('rate', 'noise', 'jump', 'update', 'equation', 'sampling')
RANDOMNESS = ('deterministic', 'stochastic')
STRUCTURES = ('variable membership', 'changing topology')
STOCHASTIC_LAWS = frozenset({'noise', 'jump', 'sampling'})
DIFFERENTIABLE_LAWS = frozenset({'rate', 'update', 'equation'})


@dataclass(frozen=True)
class Tags(BaseSpec):
    """Capability tags.

    They say what a part is, what a system derives from its parts, or what a solver or a backend supports. The tags of a
    part come from its declarations and its arguments alone, so they are known before `prepare`. A set that is empty
    means any value for `time`, `space`, `domains`, and `laws`, and a fixed structure for `structure`. Law kinds are
    `None` when the declarations do not show them, such as for a part that writes its laws in `_get_laws` without
    listing their classes, and the randomness is then `None` too, unless an argument shows it.

    Args:
        time:
            The times the part runs in, among `'continuous'`, `'discrete'`, and `'static'`.

        domains:
            The domains of the states, among `'real'`, `'integer'`, `'boolean'`, `'categorical'`, `'complex'`,
            `'vector'`, `'matrix group'`, and `'power series'`.

        space:
            The spaces, among `'none'`, `'continuous'`, `'lattice'`, `'network'`, `'hypergraph'`, `'cell complex'`,
            `'phase space'`, and `'curved spacetime'`.

        laws:
            The kinds of the laws, among `'rate'`, `'noise'`, `'jump'`, `'update'`, `'equation'`, and `'sampling'`, or
            `None` when the declarations do not show them.

        randomness:
            `'deterministic'`, `'stochastic'`, or `None` when the declarations do not show it.

        structure:
            How the structure changes, among `'variable membership'` and `'changing topology'`, empty for a fixed one.

    Examples:
        >>> from skcomplex.core.specs import Tags
        >>> tags = Tags(time=frozenset({'continuous'}), laws=frozenset({'rate', 'noise'}), randomness='stochastic')
        >>> sorted(tags.laws), tags.randomness
        (['noise', 'rate'], 'stochastic')
        >>> supported = Tags(time=frozenset({'continuous'}), laws=frozenset({'rate'}))
        >>> supported.missing(tags)
        {'laws': ['noise']}
        >>> tags.differentiable, supported.differentiable
        (False, True)
    """

    time: frozenset[str] = frozenset()
    domains: frozenset[str] = frozenset()
    space: frozenset[str] = frozenset()
    laws: frozenset[str] | None = frozenset()
    randomness: str | None = 'deterministic'
    structure: frozenset[str] = frozenset()

    def _check(self) -> None:
        """Raise an error for a tag that is none of the values of its field.

        Raises:
            ValidationError:
                A tag is unknown, named in the message with the values of its field.
        """
        fields = {
            'time': (self.time, TIMES),
            'domains': (self.domains, DOMAINS),
            'space': (self.space, SPACES),
            'laws': (self.laws or frozenset(), LAWS),
            'randomness': (frozenset() if self.randomness is None else {self.randomness}, RANDOMNESS),
            'structure': (self.structure, STRUCTURES),
        }
        for name, (values, known) in fields.items():
            unknown = sorted(set(values) - set(known))
            if unknown:
                error_msg = f'The {name} tag {unknown[0]!r} is unknown. Use one of {", ".join(map(repr, known))}.'
                raise ValidationError(error_msg)

    @property
    def differentiable(self) -> bool:
        """Whether runs of the part can be differentiated in its parameters.

        It holds for deterministic rate, update, and equation laws, which the JAX backend lowers to pure functions.
        """
        return (
            self.randomness == 'deterministic'
            and self.laws is not None
            and bool(self.laws)
            and self.laws <= DIFFERENTIABLE_LAWS
        )

    def missing(self, tags: Tags) -> dict[str, list[str]]:
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
