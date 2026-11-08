"""Define the source loader."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
import importlib.util
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from ...base import BaseConfig
from ...errors import ValidationError

CACHE_DIR = Path.home() / '.cache' / 'skcomplex' / 'kernels'
"""The default directory of the generated sources and of Numba's cache."""


class SourceLoader(BaseConfig[str | Path | None, str]):
    """Loader of generated source, imported from a file of the cache directory named by its digest.

    `prepare(cache_dir)` sets the directory of the generated files, `~/.cache/skcomplex/kernels` by default, and
    `run(source)` writes the source there once and imports it into `module_`. Source without injected functions is
    kept in `sys.modules`, so the same source is imported once. Source with injected functions is imported afresh,
    with the functions set on the module before its code runs.

    Args:
        prefix:
            The start of the name of the module, a Python identifier.

        functions:
            The functions the source reads by name, injected into its module, or `None` for none.

    Attributes:
        directory_:
            The directory of the generated files.

        module_:
            The module of the last source run.

    Examples:
        >>> from skcomplex.core.engine.problems._source_loader import SourceLoader
        >>> loader = SourceLoader('skcomplex_example').prepare()
        >>> loader.run('def double(x): return 2 * x').module_.double(3)
        6
    """

    directory_: Path
    module_: ModuleType

    def __init__(self, prefix: str, functions: Mapping[str, object] | None = None) -> None:
        self.prefix = prefix
        self.functions = functions

    def _check(self) -> None:
        """Raise an error for a prefix that is not a Python identifier.

        Raises:
            ValidationError:
                The prefix is not a Python identifier.
        """
        if not self.prefix.isidentifier():
            error_msg = f'The prefix of a source loader is a Python identifier, not {self.prefix!r}.'
            raise ValidationError(error_msg)

    def _prepare(self, context: str | Path | None) -> None:
        """Set the directory of the generated files, the default cache directory when none is given.

        Args:
            context:
                The directory of the generated files, or `None` for `~/.cache/skcomplex/kernels`.
        """
        self.directory_ = Path(context) if context is not None else CACHE_DIR

    def _run(self, inputs: str | None) -> None:
        """Write the source to the cache directory once and import it into `module_`.

        Args:
            inputs:
                The generated source.

        Raises:
            ValidationError:
                No source was given.
            ImportError:
                The generated file cannot be imported.
        """
        if inputs is None:
            error_msg = 'A source loader runs on generated source, not None.'
            raise ValidationError(error_msg)
        functions = dict(self.functions or {})
        name = f'{self.prefix}_{hashlib.sha256(inputs.encode()).hexdigest()[:24]}'
        if not functions and name in sys.modules:
            self.module_ = sys.modules[name]
            return
        self.directory_.mkdir(parents=True, exist_ok=True)
        path = self.directory_ / f'{name}.py'
        if not path.exists():
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(inputs)
            temporary.replace(path)
        specification = importlib.util.spec_from_file_location(name, path)
        if specification is None or specification.loader is None:
            error_msg = f'The generated source at {path} cannot be imported.'
            raise ImportError(error_msg)
        module = importlib.util.module_from_spec(specification)
        vars(module).update(functions)
        if not functions:
            sys.modules[name] = module
        specification.loader.exec_module(module)
        self.module_ = module
