## REMOVED Requirements

### Requirement: Listing systems imports no system
**Reason**: Superseded by "Systems are discovered with all_systems", which reads the class definitions statically and
imports only the modules of the systems it returns. The `catalog()` function is removed.
**Migration**: Call `all_systems(field=...)` of `skcomplex.systems` instead of `catalog(field=...)`.
