## ADDED Requirements

### Requirement: Docstrings of subclassed parts point to the finder first
The docstrings of `Element` and `Interaction`, the core classes users subclass to define parts, SHALL say what the part
is and how a subclass declares it, and SHALL then tell the user to look for an existing part with `all_components`
before writing a subclass, with an example that finds one.

#### Scenario: A user who wants a particle
- **WHEN** a reader who wants a particle reads the docstring of `Element`
- **THEN** it tells them to call `all_components` first, and its example finds `MassPointElement`

### Requirement: Usage details of subclassed parts live in the guide
The docstrings of `Element` and `Interaction` SHALL NOT hold details of particular uses, such as ports and connections
or agent rules. The guide pages on those uses SHALL hold them.

#### Scenario: Ports and agents stay in the guide
- **WHEN** a reader reads the docstring of `Element`
- **THEN** it holds no paragraph on ports or agent rules, and the guide pages on ports and connections and on agent
  based systems hold those details
