"""Define the rewriting problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._base_problem_spec import BaseProblemSpec

if TYPE_CHECKING:
    from ..engine.problems._graph_kernel import GraphKernel
    from ..engine.problems._patch_kernel import PatchKernel
    from ..engine.problems._word_kernel import WordKernel

type Compiled = WordKernel | PatchKernel | GraphKernel


@dataclass(frozen=True)
class RewritingProblemSpec(BaseProblemSpec):
    """Problem of the rewriting form, with the compiled rewriting system of every rewritten element.

    Args:
        rewriting:
            The compiled rewriting systems of the words, patches, and host graphs.
    """

    rewriting: tuple[Compiled, ...] = field(default=(), repr=False)
