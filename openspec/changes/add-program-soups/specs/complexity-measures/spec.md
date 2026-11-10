## ADDED Requirements

### Requirement: High-order entropy by compression
`entropy` SHALL take the kind `'high-order'`, which gives the Shannon entropy in bits per symbol of a sequence of
integer symbols minus its compressed size in bits per symbol, with the compressor `'brotli'`, `'lzma'`, `'zlib'`, or
`'bz2'` and its quality, after Agüera y Arcas et al. (2024, arXiv 2406.19108). A group state SHALL be read frame by
frame as the concatenation of the rows of the members.

#### Scenario: Random bytes
- **WHEN** a user measures 2^20 bytes drawn uniformly
- **THEN** the high-order entropy lies within 0.1 bits of 0

#### Scenario: Copies of one string
- **WHEN** a user measures 2^14 copies of one random string of 64 bytes
- **THEN** the high-order entropy exceeds 5 bits

#### Scenario: Unknown compressor
- **WHEN** a user passes `compressor='zip'`
- **THEN** it raises `AnalysisError`, whose message names the four compressors
