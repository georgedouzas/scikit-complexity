"""Define the specs: frozen values that declare, record, or hold what the classes of the library read."""

from ._accelerated_spec import AcceleratedSpec
from ._affine_spec import AffineSpec
from ._agent_snapshot_spec import AgentSnapshotSpec
from ._agent_state_spec import AgentStateSpec
from ._arguments_spec import ArgumentsSpec
from ._axis_spec import AxisSpec
from ._base_record_spec import BaseRecordSpec
from ._base_spec import BaseSpec
from ._base_structure_spec import BaseStructureSpec
from ._batch_matching_spec import BatchMatchingSpec
from ._call_spec import CallSpec
from ._capacity_rows_spec import CapacityRowsSpec
from ._cardinality_spec import CardinalitySpec
from ._cascade_record_spec import CascadeRecordSpec
from ._clamp_spec import ClampSpec
from ._clause_spec import ClauseSpec
from ._connector_spec import ConnectorSpec
from ._continuous_matching_spec import ContinuousMatchingSpec
from ._control_solution_spec import ControlSolutionSpec
from ._covariance_adaptation_spec import CovarianceAdaptationSpec
from ._dataset_spec import DatasetSpec
from ._declared_variable_spec import DeclaredVariableSpec
from ._dialect_spec import DialectSpec
from ._differential_evolution_spec import DifferentialEvolutionSpec
from ._direction_spec import DirectionSpec
from ._draw_spec import DrawSpec
from ._entry_spec import EntrySpec
from ._event_record_spec import EventRecordSpec
from ._evolution_strategy_spec import EvolutionStrategySpec
from ._extremal_record_spec import ExtremalRecordSpec
from ._fractional_rows_spec import FractionalRowsSpec
from ._fractional_state_spec import FractionalStateSpec
from ._geometry_spec import TURTLE_SYMBOLS, GeometrySpec
from ._hilbert_spec import HilbertSpec
from ._holonomy_score_spec import HolonomyScoreSpec
from ._hyperedge_reduction_spec import HyperedgeReductionSpec
from ._interaction_requirement_spec import InteractionRequirementSpec
from ._issue_spec import IssueSpec
from ._iterative_matching_spec import IterativeMatchingSpec
from ._jump_path_spec import JumpPathSpec
from ._layer_spec import LayerSpec
from ._layout_spec import LayoutSpec
from ._linear_objective_spec import LinearObjectiveSpec
from ._linear_rule_spec import LinearRuleSpec
from ._local_operator_spec import LocalOperatorSpec
from ._lookup_spec import LookupSpec
from ._lowered_functions_spec import LoweredFunctionsSpec
from ._match_spec import MatchSpec
from ._matrix_group_spec import MatrixGroupSpec
from ._matrix_law_spec import MatrixLawSpec
from ._module_spec import Letters, ModuleSpec
from ._move_spec import MOVES, MoveSpec
from ._neighbor_read_spec import NeighborReadSpec
from ._neighbor_reduction_spec import NeighborReductionSpec
from ._network_rows_spec import NetworkRowsSpec
from ._network_snapshot_spec import NetworkSnapshotSpec
from ._noise_process_spec import NoiseProcessSpec
from ._objective_spec import ObjectiveSpec
from ._observable_spec import ObservableSpec
from ._operator_spec import OperatorSpec
from ._outer_totalistic_spec import OuterTotalisticSpec
from ._pair_read_spec import PairReadSpec
from ._parameter_spec import ParameterSpec
from ._parameters_spec import ParametersSpec
from ._particle_swarm_spec import ParticleSwarmSpec
from ._partner_spec import PartnerSpec
from ._partner_switch_spec import PartnerSwitchSpec
from ._patch_structure_spec import PatchStructureSpec
from ._phase_spec import PhaseSpec
from ._phases_schedule_spec import ACTIVATIONS, DEFAULT_PHASE, PhasesScheduleSpec
from ._power_series_spec import PowerSeriesSpec
from ._prototile_spec import PrototileSpec
from ._queue_spec import QueueSpec
from ._radius_spec import RadiusSpec
from ._random_search_spec import RandomSearchSpec
from ._read_spec import ReadSpec
from ._reduction_spec import ReductionSpec
from ._renewal_record_spec import RenewalRecordSpec
from ._requirement_spec import RequirementSpec
from ._role_spec import RoleSpec, Spec
from ._row_score_spec import RowScoreSpec
from ._run_settings_spec import RunSettingsSpec
from ._run_state_spec import RunStateSpec
from ._sampled_draw_spec import SampledDrawSpec
from ._sampler_spec import MONTE_CARLO_SCHEDULES, SAMPLERS, SamplerSpec
from ._samples_spec import SamplesSpec
from ._schedule_spec import ScheduleSpec
from ._sector_spec import SectorSpec
from ._series_spec import SeriesSpec
from ._signature_spec import SignatureSpec
from ._snapshots_spec import SnapshotsSpec
from ._stable_matching_spec import StableMatchingSpec
from ._state_spec import StateSpec
from ._step_spec import StepSpec
from ._structure_state_spec import StructureStateSpec
from ._subshift_spec import SubshiftSpec
from ._support_spec import SupportSpec
from ._symmetry_spec import SymmetrySpec
from ._table_spec import TableSpec
from ._table_values_spec import TableValuesSpec
from ._tensor_spec import TensorSpec
from ._topology_request_spec import EDGE_KINDS, ENDS, TOPOLOGY_KINDS, TopologyRequestSpec
from ._totalistic_spec import TotalisticSpec
from ._validation_report_spec import ValidationReportSpec
from ._variable_spec import VariableSpec
from ._visual_spec import VisualSpec
from ._wiener_block_spec import WienerBlockSpec
from ._wiener_process_spec import WienerProcessSpec
from ._word_structure_spec import WordStructureSpec

__all__ = [
    'ACTIVATIONS',
    'DEFAULT_PHASE',
    'EDGE_KINDS',
    'ENDS',
    'MONTE_CARLO_SCHEDULES',
    'MOVES',
    'SAMPLERS',
    'TOPOLOGY_KINDS',
    'TURTLE_SYMBOLS',
    'AcceleratedSpec',
    'AffineSpec',
    'AgentSnapshotSpec',
    'AgentStateSpec',
    'ArgumentsSpec',
    'AxisSpec',
    'BaseRecordSpec',
    'BaseSpec',
    'BaseStructureSpec',
    'BatchMatchingSpec',
    'CallSpec',
    'CapacityRowsSpec',
    'CardinalitySpec',
    'CascadeRecordSpec',
    'ClampSpec',
    'ClauseSpec',
    'ConnectorSpec',
    'ContinuousMatchingSpec',
    'ControlSolutionSpec',
    'CovarianceAdaptationSpec',
    'DatasetSpec',
    'DeclaredVariableSpec',
    'DialectSpec',
    'DifferentialEvolutionSpec',
    'DirectionSpec',
    'DrawSpec',
    'EntrySpec',
    'EventRecordSpec',
    'EvolutionStrategySpec',
    'ExtremalRecordSpec',
    'FractionalRowsSpec',
    'FractionalStateSpec',
    'GeometrySpec',
    'HilbertSpec',
    'HolonomyScoreSpec',
    'HyperedgeReductionSpec',
    'InteractionRequirementSpec',
    'IssueSpec',
    'IterativeMatchingSpec',
    'JumpPathSpec',
    'LayerSpec',
    'LayoutSpec',
    'Letters',
    'LinearObjectiveSpec',
    'LinearRuleSpec',
    'LocalOperatorSpec',
    'LookupSpec',
    'LoweredFunctionsSpec',
    'MatchSpec',
    'MatrixGroupSpec',
    'MatrixLawSpec',
    'ModuleSpec',
    'MoveSpec',
    'NeighborReadSpec',
    'NeighborReductionSpec',
    'NetworkRowsSpec',
    'NetworkSnapshotSpec',
    'NoiseProcessSpec',
    'ObjectiveSpec',
    'ObservableSpec',
    'OperatorSpec',
    'OuterTotalisticSpec',
    'PairReadSpec',
    'ParameterSpec',
    'ParametersSpec',
    'ParticleSwarmSpec',
    'PartnerSpec',
    'PartnerSwitchSpec',
    'PatchStructureSpec',
    'PhaseSpec',
    'PhasesScheduleSpec',
    'PowerSeriesSpec',
    'PrototileSpec',
    'QueueSpec',
    'RadiusSpec',
    'RandomSearchSpec',
    'ReadSpec',
    'ReductionSpec',
    'RenewalRecordSpec',
    'RequirementSpec',
    'RoleSpec',
    'RowScoreSpec',
    'RunSettingsSpec',
    'RunStateSpec',
    'SampledDrawSpec',
    'SamplerSpec',
    'SamplesSpec',
    'ScheduleSpec',
    'SectorSpec',
    'SeriesSpec',
    'SignatureSpec',
    'SnapshotsSpec',
    'Spec',
    'StableMatchingSpec',
    'StateSpec',
    'StepSpec',
    'StructureStateSpec',
    'SubshiftSpec',
    'SupportSpec',
    'SymmetrySpec',
    'TableSpec',
    'TableValuesSpec',
    'TensorSpec',
    'TopologyRequestSpec',
    'TotalisticSpec',
    'ValidationReportSpec',
    'VariableSpec',
    'VisualSpec',
    'WienerBlockSpec',
    'WienerProcessSpec',
    'WordStructureSpec',
]
