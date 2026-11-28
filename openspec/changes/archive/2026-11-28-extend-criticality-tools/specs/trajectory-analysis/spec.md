## ADDED Requirements

### Requirement: Power spectra take a segment length
`spectrum(kind='power', segment_length=n)` SHALL estimate the power spectral density with Welch segments of `n`
samples, or of the whole series when it is shorter, with 1024 as the default, and SHALL raise an `AnalysisError` for a
segment length below 2.

#### Scenario: Finer frequencies from longer segments
- **WHEN** the spectrum of a series of 10^5 samples is computed with `segment_length=8192`
- **THEN** it has 4097 frequencies, against 513 with the default

#### Scenario: Segment too short
- **WHEN** `segment_length=1` is given
- **THEN** an `AnalysisError` names `segment_length`
