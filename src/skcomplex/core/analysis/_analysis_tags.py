"""Define the tags that an analysis needs of its data and of the system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass

from ..base import BaseSpec
from ..components import Tags
from ..enums import DataKind, RandomnessTag, SpaceTag, TimeTag
from ..errors import ValidationError


@dataclass(frozen=True)
class AnalysisTags(BaseSpec):
    """A set of tags that an analysis needs of its data and of the system.

    It names the kind of data an analysis reads and the tags the system behind the data must have, so an analysis
    refuses data it cannot read before doing any work, such as `AnalysisTags('system', randomness='deterministic')` for
    an analysis of systems with no randomness. An analysis module declares it as the constant `TAGS`, and an empty field
    needs nothing. It keeps a stability analysis off a random epidemic and a network measure off a price series.

    Read more in the user guide, under Analyse.

    Args:
        data:
            The kind of data the analysis reads.

        time:
            The times of which the system needs one, among `'continuous'`, `'discrete'`, and `'static'`.

        space:
            The spaces of which the system needs one, such as `'network'`.

        randomness:
            The randomness the system needs, `'deterministic'` or `'stochastic'`, or `None` for either.

    See Also:
        applicable_analyses: Lists the analyses whose tags a source meets.

    Examples:
        >>> from skcomplex.core.analysis import AnalysisTags
        >>> from skcomplex.core.components import Tags
        >>> needed = AnalysisTags('system', randomness='deterministic')
        >>> needed.get_missing(Tags(time=frozenset({'continuous'}), randomness='stochastic'))
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
            'data': ({self.data}, DataKind),
            'time': (self.time, TimeTag),
            'space': (self.space, SpaceTag),
            'randomness': (frozenset() if self.randomness is None else {self.randomness}, RandomnessTag),
        }
        for name, (values, known) in fields.items():
            unknown = sorted(set(values) - set(known))
            if unknown:
                accepted = ', '.join(repr(str(item)) for item in known)
                error_msg = f'The {name} tag {unknown[0]!r} is unknown. Use one of {accepted}.'
                raise ValidationError(error_msg)

    def get_missing(self, tags: Tags) -> dict[str, list[str]]:
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
