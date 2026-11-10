## Why

Agüera y Arcas et al. show that self-replicating programs arise from random byte strings without any fitness function
(2406.19108). A soup of tapes is paired at random, each pair is concatenated and run by an interpreter of a minimal
language, BFF, whose code and data share one tape, and the tapes are split back. After a few thousand epochs a state
transition often happens, in which replicators take over the soup, and the high-order entropy of the soup, its Shannon
entropy minus its compressed size per byte, jumps. The core has the parts for this except two. An array update, which
hands a whole state to a compiled function, runs only on lattices and networks, so a well-mixed group of tapes has no
way to be updated as one array. And the entropies of the analysis layer have no estimate of Kolmogorov complexity by
compression, which the paper's measure needs.

## What Changes

- `ArrayUpdateLaw` runs on groups without a space, the well-mixed case, with one row per member and the components of
  a vector state on the last axis, as it does on networks.
- `entropy` gains the kind `'high-order'`, the Shannon entropy per symbol of a sequence of symbols minus its compressed
  size in bits per symbol, with the compressors brotli, lzma, zlib, and bz2. It reads a series or every frame of a
  group state as one string.
- New system `BffSoupSystem` in `computer_science/evolutionary_computation`: tapes of 64 bytes, random ordered pairs
  every epoch, the BFF interpreter compiled with Numba, a step limit, and background mutation. It records the
  high-order entropy of the soup as an observable.
- New runtime dependency `brotli`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `general-mechanisms`: array updates of well-mixed groups.
- `complexity-measures`: high-order entropy by compression.
- `model-library`: the BFF primordial soup.

## Impact

- Core: `core/components/laws/_array_update.py` (its requirement on the space) and the array-update lowering of the
  NumPy and Numba backends for a group with no space. `core/analysis/_entropy.py` (the new kind and its settings).
- Systems: a new module in `skcomplex.systems.computer_science.evolutionary_computation`.
- Dependencies: `brotli` 1.2.1, released 2026-10-22.
- Docs: the entropy section of the analysis guide, a gallery example, and the inventory, where the system uses the
  existing primitives `agents`, `draws`, and `updates` and the new `compression-entropy`.
- Independent of the other changes. The open-endedness measures of `add-diversity-search` can read the soup later.
