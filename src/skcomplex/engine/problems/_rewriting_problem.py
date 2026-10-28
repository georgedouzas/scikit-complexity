"""Problem of the rewriting form, with the compiled rewriting system of every rewritten element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ._base import BaseProblem
from ._graph_kernel import GraphKernel
from ._patch_kernel import PatchKernel
from ._word_kernel import WordKernel

type Compiled = WordKernel | PatchKernel | GraphKernel


@dataclass(frozen=True)
class RewritingProblem(BaseProblem):
    """Problem of the rewriting form, with the compiled rewriting system of every rewritten element.

    Args:
        rewriting:
            The compiled rewriting systems of the words, patches, and host graphs.
    """

    rewriting: tuple[Compiled, ...] = field(default=(), repr=False)
