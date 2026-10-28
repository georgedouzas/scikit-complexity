# Proposal

## Why

The checks session fails:
- `black` would reformat 141 files.
- mypy reports about 3000 errors.
- pydoclint reports about 820 findings.

The code also carries 323 `noqa`, 119 `type: ignore`, and 42 `nosec` comments that hide findings instead of fixing them. On top of that, 43 modules use `if TYPE_CHECKING:` imports, because specs in the root package name classes of the layers above them.

## What Changes

- **Specs at their layer.** A spec lives in the lowest package that can name its field types:
  - `skcomplex.specs` for specs built only from root types;
  - `skcomplex.core.specs`, `skcomplex.engine.specs`, and `skcomplex.experiments.specs` for the others.

  Every spec still derives from `BaseSpec`. **BREAKING**: the import paths of the moved specs change.
- **No hidden imports.** No module uses `if TYPE_CHECKING:`, and the import-linter contracts drop `exclude_type_checking_imports`.
- **No suppressions.** The code holds no `noqa`, `type: ignore`, or `nosec` comment; every finding is fixed in the code. A rule that does not fit the project is changed in `pyproject.toml`, in the open.
- **The bases type their hooks precisely.**
  - `_get_laws` takes the namespace and returns the laws.
  - `BaseConfig` is generic in the context of `prepare` and the input of `run`, so a subclass types its hooks without breaking its base.
- **A green checks session.** `black` formats the whole tree, and mypy and pydoclint report nothing on `src`.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `package-layout`: specs live at the layer that names their types, and the code holds no type-checking imports or suppressions.

## Impact

- **Code:** every package, because of formatting, typing, docstrings, and the removed suppressions.
- **Spec imports:** the specs that move to `core.specs`, `engine.specs`, and `experiments.specs` are imported from there.
- **Configuration:** `pyproject.toml` loses `exclude_type_checking_imports`.
