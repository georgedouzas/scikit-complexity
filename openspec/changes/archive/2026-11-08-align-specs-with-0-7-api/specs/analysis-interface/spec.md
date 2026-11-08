## REMOVED Requirements

### Requirement: Results are fitted attributes
**Reason**: Analyses are functions that return frozen results, see "Analyses are functions of results". No analysis
keeps results in fitted attributes.
**Migration**: Call the analysis function on the result and read the fields of the frozen result it returns, such as
the exponent of a power-law fit.
