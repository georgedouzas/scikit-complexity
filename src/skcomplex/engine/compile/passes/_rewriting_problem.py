"""Problem of the rewriting form, with the compiled rewriting system of every rewritten element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ...problems import (
    BaseProblem,
)
from .._compiled_word import CompiledWord
from ._compiled_graph import CompiledGraph
from ._compiled_patch import CompiledPatch

type Compiled = CompiledWord | CompiledPatch | CompiledGraph


@dataclass(frozen=True)
class RewritingProblem(BaseProblem):
    """Problem of the rewriting form, with the compiled rewriting system of every rewritten element.

    Args:
        rewriting:
            The compiled rewriting systems of the words, patches, and host graphs.
    """

    rewriting: tuple[Compiled, ...] = field(default=(), repr=False)
