## Context

`ArrayUpdateLaw` in `core/components/laws/_array_update.py` hands a whole array state to a Python function at every
step of discrete time. Its requirement allows `LatticeSpace` and, for groups on edges and hyperedges, `NetworkSpace`.
The genome walks of `molecular-structures` and the sandpile drive use it. `entropy` in `core/analysis/_entropy.py` has
the kinds `'shannon'`, `'block'`, `'lempel-ziv'` (through antropy), and others, with settings per kind.

The BFF soup of 2406.19108 holds 2^17 tapes of 64 bytes. An epoch pairs them at random, runs every pair for up to 2^13
instructions, and mutates bytes. Written as agent rules on Mesa, one epoch would take minutes. Written as one compiled
function over the whole soup, it takes milliseconds.

## Goals / Non-Goals

**Goals:**
- Let a compiled function update a well-mixed group as one array, the missing case of `ArrayUpdateLaw`.
- Add the compression estimate of high-order entropy to the entropies.
- A BFF soup whose state transition matches the paper's statistics.

**Non-Goals:**
- The tracer tokens of the paper, which tag every byte with its origin. They need a second integer state per byte and
  the copy rules of the interpreter, and they are a later option of the system.
- The other languages of the paper (Forth, SUBLEQ, Z80, 8080) and the two-dimensional soup. They are further options
  of the same module once the first language is in.
- Narrow integer storage. States stay floating point arrays in the backends. A soup of 2^17 tapes uses 64 megabytes
  per frame, so the system records sparse frames by default.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `ArrayUpdateLaw` on groups without a space | 2, a law kind the engine handles | `core/components/laws/_array_update.py`, NumPy and Numba backends | artificial chemistries and digital evolution in computer science, well-mixed populations of genomes in biology, bags of molecules in chemistry, urn models in mathematics |
| `entropy(kind='high-order')` | 2, an analysis that reads series and groups by their tags | `core/analysis/_entropy.py` | origin of life and digital evolution, DNA and protein sequences in biology, configurations of cellular automata, texts and languages in social systems |
| `BffSoupSystem` and its interpreter | 3, one system | `computer_science/evolutionary_computation/_bff_soup.py` | computer science |

The interpreter is not a core primitive. It is a Numba function that the array update of one system calls, which the
core already runs once the well-mixed case exists. The pattern of pairing, concatenating, running, and splitting stays
in the module and moves to `computer_science/common/` when a second system of the field uses it.

### Array updates of well-mixed groups

`ArrayUpdateLaw.requirement` adds the case of a group whose system has no space. The lowering is the one of the network
case: the function receives an array of shape `(members, components)` and returns one of the same shape, which the step
checks. Variable groups are not covered, since their size changes between steps.

### High-order entropy

- `entropy(source, kind='high-order', compressor='brotli', quality=2)`. The symbols are cast to bytes when they lie in
  `[0, 255]`, and to 16-bit or 32-bit little-endian words otherwise, with the word size dividing the compressed bits.
- The value is `H - 8 C / n`, with `H` the plug-in Shannon entropy in bits of the symbol frequencies, `C` the compressed
  size in bytes, and `n` the number of symbols. Quality 2 of brotli is the setting of the paper.
- On a group state, every frame is the concatenation of the rows in member order, so the result holds one value per
  frame.

### Libraries surveyed

- `brotli` 1.2.1, released 2026-10-22, the reference binding of Google's compressor, is chosen, since the paper's
  numbers use brotli at quality 2.
- The standard library gives lzma, zlib, and bz2, which are offered as alternatives without a new dependency.
- antropy, already a dependency, gives Lempel-Ziv complexity but no compressed size, so it is not used for this kind.
- No maintained Python package runs BFF soups. The reference code of the paper is C++ and CUDA without a Python
  binding. The interpreter is a short Numba function in the system module.

### The BFF soup

- `BffSoupSystem(n_tapes=2**17, tape_length=64, max_steps=2**13, mutation_rate=0.00024, seeds=(), random_state=None)`.
  `seeds` is a sequence of byte strings placed in the first tapes, for the seeded runs of the paper.
- A group `soup` with one vector state `tape` of `tape_length` components in `[0, 255]`, and one `ArrayUpdateLaw` whose
  function permutes the rows with the run's generator, runs the pairs in a Numba kernel, and mutates bytes with a
  binomial draw of positions. The interpreter follows the instruction table of the paper: heads start at 0, the
  instruction pointer at 0, brackets match with nesting, an unmatched bracket halts, and heads wrap within the
  concatenated tape.
- The observable `high_order_entropy` reads the soup each frame through the same private function as the analysis.
- Primitives: `agents`, `groups`, `draws`, `updates`, and the new `compression-entropy`.

## Risks / Trade-offs

- [Statistical scenarios are slow] → The seeded-replicator scenario uses 2^12 tapes and 128 epochs, and is marked slow.
  Its bounds bracket the share of 22% that the paper reports for seeded soups, and are set from the first runs of the
  implementation before the test is fixed.
- [Head wrapping and bracket rules differ between implementations] → The design follows the paper's text and the
  scenario of the replicator of Figure 4, which fails if a rule differs.
- [Memory of frames] → The system records every 128 epochs by default, and the guide shows how to record only the
  observable.

## Migration Plan

None. Existing array updates on lattices and networks are unchanged.

## Open Questions

- Should the high-order entropy also be offered by `complexity`, beside the complexity-entropy plane? The design keeps
  one place, `entropy`.
