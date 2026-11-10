## ADDED Requirements

### Requirement: BFF primordial soup
The library SHALL provide `BffSoupSystem` in `computer_science/evolutionary_computation` (Agüera y Arcas et al. 2024,
arXiv 2406.19108): a soup of tapes of 64 bytes drawn uniformly, in which every epoch splits a random permutation into
ordered pairs, runs each concatenated pair with the BFF interpreter until it halts or reads 2^13 instructions, splits
it back, and replaces every byte by a uniform byte with the mutation rate.

#### Scenario: Instruction set
- **WHEN** the interpreter runs one tape for each of the ten instructions `< > { } - + . , [ ]` in a known context
- **THEN** the heads, the tape, and the instruction pointer change as the table of the paper states, and every other
  byte is a no-operation

#### Scenario: A replicator copies itself
- **WHEN** the self-replicator of Figure 4 of the paper is concatenated with a tape of zeros and run
- **THEN** both halves after the split equal the replicator

#### Scenario: Defaults of the paper
- **WHEN** a user builds the system with its defaults
- **THEN** it has 2^17 tapes of 64 bytes, a limit of 2^13 instructions, and the mutation rate 0.00024 per byte and epoch

#### Scenario: Seeded replicator spreads
- **WHEN** a soup of 2^12 tapes seeded with one replicator runs 128 epochs for 50 seeds without mutation
- **THEN** the high-order entropy of the final soup exceeds 1 bit in at least 5 runs and in at most 25 runs

### Requirement: The BFF soup is recorded in the inventory
The inventory SHALL record `bff_primordial_soup` as implemented with the source 2406.19108 and the primitive
`compression-entropy`, and the gallery SHALL hold its example.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for `bff_primordial_soup`
