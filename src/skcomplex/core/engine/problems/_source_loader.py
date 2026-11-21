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


class SourceLoader(BaseConfig[str | Path | None, str]):
    """A loader that writes generated Python code to a file and imports it.

    Backends generate the code of the compiled laws of a system, and this loader writes it once to a file named by its
    digest in a cache directory, `~/.cache/skcomplex/kernels` by default, and imports it into `module_`, so the same
    code loads fast the next time. Code that reads injected functions is imported afresh with them set on its module. It
    loads the compiled rates of a population, a reactor, and a market.

    Read more in the user guide, under Backends.

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

    See Also:
        ExpressionPrinter: Prints the code the loader imports.

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

    def _prepare(self, context: str | Path | None) -> None:
        """Set the directory of the generated files.

        It first raises an error for a prefix that is not a Python identifier.

        Without one, it is the default cache directory.

        Args:
            context:
                The directory of the generated files, or `None` for `~/.cache/skcomplex/kernels`.

        Raises:
            ValidationError:
                The prefix is not a Python identifier.
        """
        if not self.prefix.isidentifier():
            error_msg = f'The prefix of a source loader is a Python identifier, not {self.prefix!r}.'
            raise ValidationError(error_msg)
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
