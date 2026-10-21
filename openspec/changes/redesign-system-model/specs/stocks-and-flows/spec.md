# Spec Delta

## REMOVED Requirements

### Requirement: Flows move quantities between holders
**Reason**: Replaced by the transfer law of `system-modeling`, which works under every time class. `Flow`, `Source`, and
`Sink` were its system-dynamics realization.
**Migration**: Replace `Flow` with `Transfer`, and `Source` and `Sink` with a `None` source or sink. The
system-dynamics vocabulary is available from `skcomplex.models.engineering`.

### Requirement: Ledgers keep agents' balance sheets consistent
**Reason**: Replaced by transfer laws and invariants of `system-modeling`. `transfer` and `Identity` were their
accounting realization.
**Migration**: Replace `transfer` calls with `Transfer` laws, and `Identity` with `Invariant`.

### Requirement: Auxiliaries and lookups shape laws
**Reason**: Auxiliaries are named expressions, and lookups are numeric functions, both general expressions. Their
system-dynamics names are a realization.
**Migration**: Use named expressions and numeric functions in the core, or the system-dynamics helpers of
`skcomplex.models.engineering`.

### Requirement: Delays and smoothing add hidden stocks
**Reason**: `delay1`, `delay3`, `smooth`, and `smooth3` are system-dynamics realizations of a distributed delay of
order `k`.
**Migration**: Use the core's distributed-delay expression with the order as a parameter, or the named helpers of
`skcomplex.models.engineering`.
