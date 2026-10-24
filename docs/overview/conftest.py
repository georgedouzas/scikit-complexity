"""Run the Python blocks of the user guide in a module that pickle can import."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
import types
from typing import Any

from pytest_markdown_docs._runners import DefaultRunner, register_runner
from pytest_markdown_docs.definitions import FenceTestDefinition

FENCE_MODULE = 'fence'


class _ImportableFence(types.ModuleType):
    """Module whose names are the globals of the running block, so pickle finds the functions a block defines."""

    def __init__(self, names: dict[str, Any]) -> None:
        super().__init__(FENCE_MODULE)
        self.__dict__.update(names)
        self._names = names

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Return a global of the running block."""
        try:
            return self._names[name]
        except KeyError as error:
            raise AttributeError(name) from error


@register_runner(default=True)
class ImportableRunner(DefaultRunner):
    """Runner that registers the running block's module, so functions the block defines pickle by reference."""

    def runtest(self, test: FenceTestDefinition, args: dict[str, Any], **options: Any) -> None:  # noqa: ANN401
        """Run a block with its module registered under its name."""
        sys.modules[FENCE_MODULE] = _ImportableFence(args)
        try:
            super().runtest(test, args, **options)
        finally:
            sys.modules.pop(FENCE_MODULE, None)
