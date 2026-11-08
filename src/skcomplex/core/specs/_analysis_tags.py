"""Define the tags an analysis needs.

They are the kind of data it reads and the capability tags of the system behind it.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ValidationError
from ..types import DataKind
from ._base_spec import BaseSpec
from ._tags import RANDOMNESS, SPACES, TIMES, Tags


@dataclass(frozen=True)
class AnalysisTags(BaseSpec):
    """Tags an analysis needs.

    They are the kind of data it reads and the capability tags of the system that produced it. An analysis module
    declares its tags as the module constant `TAGS`, and a family of kinds declares the tags of the kinds that differ in
    `KIND_TAGS`, by kind. The capability tags are checked against the tags of a system, or of the system behind a
    trajectory, before the analysis does any work. A field that is empty or `None` needs nothing.

    Args:
        data:
            The kind of data the analysis reads.

        time:
            The times of which the system needs one, among `'continuous'`, `'discrete'`, and `'static'`.

        space:
            The spaces of which the system needs one, such as `'network'`.

        randomness:
            The randomness the system needs, `'deterministic'` or `'stochastic'`, or `None` for either.

    Examples:
        >>> from skcomplex.core.specs import AnalysisTags, Tags
        >>> needed = AnalysisTags('system', randomness='deterministic')
        >>> needed.missing(Tags(time=frozenset({'continuous'}), randomness='stochastic'))
        {'randomness': ['stochastic']}
    """

    data: DataKind | str
    time: frozenset[str] = frozenset()
    space: frozenset[str] = frozenset()
    randomness: str | None = None

    def _check(self) -> None:
        """Raise an error for a data kind or a tag that is none of the values of its field.

        Raises:
            ValidationError:
                The data kind or a tag is unknown, named in the message with the values of its field.
        """
        fields = {
            'data': ({self.data}, tuple(DataKind)),
            'time': (self.time, TIMES),
            'space': (self.space, SPACES),
            'randomness': (frozenset() if self.randomness is None else {self.randomness}, RANDOMNESS),
        }
        for name, (values, known) in fields.items():
            unknown = sorted(set(values) - set(known))
            if unknown:
                error_msg = f'The {name} tag {unknown[0]!r} is unknown. Use one of {", ".join(map(repr, known))}.'
                raise ValidationError(error_msg)

    def missing(self, tags: Tags) -> dict[str, list[str]]:
        """Return the tags of a system that do not give what the analysis needs, by field.

        The system needs one of the needed times and one of the needed spaces, and the needed randomness. A field of
        the system that is empty or `None` is unknown and not checked.

        Args:
            tags:
                The tags of a system.

        Returns:
            missing:
                The tags of the system of every field that misses what is needed, sorted.
        """
        found: dict[str, list[str]] = {}
        for name, needed, values in (('time', self.time, tags.time), ('space', self.space, tags.space)):
            if needed and values and not needed & values:
                found[name] = sorted(values)
        if self.randomness is not None and tags.randomness is not None and tags.randomness != self.randomness:
            found['randomness'] = [tags.randomness]
        return found
