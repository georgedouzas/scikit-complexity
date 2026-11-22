## ADDED Requirements

### Requirement: Docstrings of subclassed components point to the finder
The docstrings of `Element` and `Interaction`, the core classes users subclass to define components, SHALL tell the
user in their body to look for an existing component with `all_components` before writing a subclass, and SHALL name
`all_components` in their See Also section, within the scikit-learn layout.

#### Scenario: A user who wants a particle
- **WHEN** a reader who wants a particle reads the docstring of `Element`
- **THEN** it tells them to call `all_components` first, and its See Also section names `all_components`

### Requirement: The guide states the rules of agent members
The guide page on agent based systems SHALL state when the agent solver may remove a member, given its values tagged
with a quantity, and which names an agent value cannot take.

#### Scenario: Reading the agent rules
- **WHEN** a reader looks for why a member was not removed or why a value name is refused
- **THEN** the page on agent based systems states both rules with a runnable example
