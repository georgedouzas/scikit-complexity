"""Define the rewriting problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from dataclasses import dataclass, field

from ....engine.problems import BaseProblemSpec
from ._patch_kernel import PatchKernel
from ._word_kernel import WordKernel

type Compiled = WordKernel | PatchKernel


REWRITING = 'rewriting system'
"""Form of the problems of rewriting systems, which the rewriting compiler sets and the rewriting solver solves."""

@dataclass(frozen=True)
class RewritingProblemSpec(BaseProblemSpec):
    """Problem of the rewriting form, with the compiled rewriting system of every rewritten element.

    Args:
        rewriting:
            The compiled rewriting systems of the words and patches.
    """

    rewriting: tuple[Compiled, ...] = field(default=(), repr=False)
