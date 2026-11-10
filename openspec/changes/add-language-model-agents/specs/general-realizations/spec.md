## ADDED Requirements

### Requirement: Language models answer as oracles
`skcomplex.systems.general.language_models` SHALL provide `chat_oracle(model, ...)`, which returns an `OracleSpec` that
sends the request with the call seed to an endpoint compatible with the OpenAI chat API, reads the key from the
environment variable it names, and returns the text of the reply. It SHALL sit behind the optional extra
`language-models`.

#### Scenario: Extra not installed
- **WHEN** a user calls `chat_oracle('gpt-4o-mini')` without the extra `language-models`
- **THEN** it raises `MissingExtraError`, whose message names the extra

#### Scenario: Missing key
- **WHEN** a user calls the oracle of `chat_oracle` with `api_key_variable='NO_SUCH_KEY'` unset
- **THEN** it raises `ValidationError`, whose message names `NO_SUCH_KEY` and not any value

### Requirement: Options are read from free text
`skcomplex.systems.general.language_models` SHALL provide `parse_options(answer, options, count)`, which returns the
first `count` options that appear in an answer as whole tokens.

#### Scenario: Options in free text
- **WHEN** a user parses the answer `'I will pick Q, since Q won before.'` with the options `('Q', 'M', 'X')`
- **THEN** `parse_options` returns `('Q',)`

#### Scenario: No option in the answer
- **WHEN** a user parses an answer that names none of the options
- **THEN** `parse_options` raises `ValidationError` that quotes the answer
