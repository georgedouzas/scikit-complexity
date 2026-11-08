"""Define the rewriting problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass, field

from ._base_problem_spec import BaseProblemSpec
from ._patch_kernel import PatchKernel
from ._word_kernel import WordKernel


@dataclass(frozen=True)
class RewritingProblemSpec(BaseProblemSpec):
    """Problem of the rewriting form.

    It holds the compiled rewriting system of every rewritten element.

    Args:
        rewriting:
            The compiled rewriting systems of the words and patches.
    """

    rewriting: tuple[WordKernel | PatchKernel, ...] = field(default=(), repr=False)
