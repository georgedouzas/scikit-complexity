"""Define the specs.

A spec is a frozen value that declares, records, or holds what the classes of the library read.
"""

from ._accelerated_spec import AcceleratedSpec
from ._affine_spec import AffineSpec
from ._agent_snapshot_spec import AgentSnapshotSpec
from ._agent_state_spec import AgentStateSpec
from ._analysis_tags import AnalysisTags
from ._arguments_spec import ArgumentsSpec
from ._axis_spec import AxisSpec
from ._base_record_spec import BaseRecordSpec
from ._base_spec import BaseSpec
from ._base_structure_spec import BaseStructureSpec
from ._call_spec import CallSpec
from ._capacity_rows_spec import CapacityRowsSpec
from ._cascade_record_spec import CascadeRecordSpec
from ._clamp_spec import ClampSpec
from ._connector_spec import ConnectorSpec
from ._control_solution_spec import ControlSolutionSpec
from ._dataset_spec import DatasetSpec
from ._declared_variable_spec import DeclaredVariableSpec
from ._dialect_spec import DialectSpec
from ._direction_spec import DirectionSpec
from ._draw_log_spec import DrawLogSpec
from ._draw_record_spec import DrawRecordSpec
from ._draw_spec import DrawSpec
from ._entry_spec import EntrySpec
from ._event_record_spec import EventRecordSpec
from ._extremal_record_spec import ExtremalRecordSpec
from ._fractional_rows_spec import FractionalRowsSpec
from ._fractional_state_spec import FractionalStateSpec
from ._geometry_spec import TURTLE_SYMBOLS, GeometrySpec
from ._hilbert_spec import HilbertSpec
from ._holonomy_score_spec import HolonomyScoreSpec
from ._hyperedge_reduction_spec import HyperedgeReductionSpec
from ._interaction_requirement_spec import InteractionRequirementSpec
from ._issue_spec import IssueSpec
from ._jump_path_spec import JumpPathSpec
from ._layer_spec import LayerSpec
from ._layout_spec import LayoutSpec
from ._linear_objective_spec import LinearObjectiveSpec
from ._linear_rule_spec import LinearRuleSpec
from ._local_operator_spec import LocalOperatorSpec
from ._lookup_spec import LookupSpec
from ._lowered_functions_spec import LoweredFunctionsSpec
from ._matrix_group_spec import MatrixGroupSpec
from ._matrix_law_spec import MatrixLawSpec
from ._module_spec import Letters, ModuleSpec
from ._move_spec import MOVES, MoveSpec
from ._named_draw_spec import NamedDrawSpec
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
from ._partner_spec import PartnerSpec
from ._patch_structure_spec import PatchStructureSpec
from ._phase_spec import PhaseSpec
from ._phases_schedule_spec import ACTIVATIONS, DEFAULT_PHASE, PhasesScheduleSpec
from ._power_series_spec import PowerSeriesSpec
from ._prototile_spec import PrototileSpec
from ._radius_spec import RadiusSpec
from ._read_spec import ReadSpec
from ._reduction_spec import ReductionSpec
from ._region_score_spec import RegionScoreSpec
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
from ._state_spec import StateSpec
from ._step_spec import StepSpec
from ._structure_state_spec import StructureStateSpec
from ._subshift_spec import SubshiftSpec
from ._support_spec import SupportSpec
from ._symmetry_spec import SymmetrySpec
from ._table_spec import TableSpec
from ._table_values_spec import TableValuesSpec
from ._tags import DOMAINS, LAWS, RANDOMNESS, SPACES, STOCHASTIC_LAWS, STRUCTURES, TIMES, Tags
from ._tensor_spec import TensorSpec
from ._topology_request_spec import EDGE_KINDS, ENDS, HYPEREDGE_KINDS, TOPOLOGY_KINDS, TopologyRequestSpec
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
    'DOMAINS',
    'EDGE_KINDS',
    'ENDS',
    'HYPEREDGE_KINDS',
    'LAWS',
    'MONTE_CARLO_SCHEDULES',
    'MOVES',
    'RANDOMNESS',
    'SAMPLERS',
    'SPACES',
    'STOCHASTIC_LAWS',
    'STRUCTURES',
    'TIMES',
    'TOPOLOGY_KINDS',
    'TURTLE_SYMBOLS',
    'AcceleratedSpec',
    'AffineSpec',
    'AgentSnapshotSpec',
    'AgentStateSpec',
    'AnalysisTags',
    'ArgumentsSpec',
    'AxisSpec',
    'BaseRecordSpec',
    'BaseSpec',
    'BaseStructureSpec',
    'CallSpec',
    'CapacityRowsSpec',
    'CascadeRecordSpec',
    'ClampSpec',
    'ConnectorSpec',
    'ControlSolutionSpec',
    'DatasetSpec',
    'DeclaredVariableSpec',
    'DialectSpec',
    'DirectionSpec',
    'DrawLogSpec',
    'DrawRecordSpec',
    'DrawSpec',
    'EntrySpec',
    'EventRecordSpec',
    'ExtremalRecordSpec',
    'FractionalRowsSpec',
    'FractionalStateSpec',
    'GeometrySpec',
    'HilbertSpec',
    'HolonomyScoreSpec',
    'HyperedgeReductionSpec',
    'InteractionRequirementSpec',
    'IssueSpec',
    'JumpPathSpec',
    'LayerSpec',
    'LayoutSpec',
    'Letters',
    'LinearObjectiveSpec',
    'LinearRuleSpec',
    'LocalOperatorSpec',
    'LookupSpec',
    'LoweredFunctionsSpec',
    'MatrixGroupSpec',
    'MatrixLawSpec',
    'ModuleSpec',
    'MoveSpec',
    'NamedDrawSpec',
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
    'PartnerSpec',
    'PatchStructureSpec',
    'PhaseSpec',
    'PhasesScheduleSpec',
    'PowerSeriesSpec',
    'PrototileSpec',
    'RadiusSpec',
    'ReadSpec',
    'ReductionSpec',
    'RegionScoreSpec',
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
    'StateSpec',
    'StepSpec',
    'StructureStateSpec',
    'SubshiftSpec',
    'SupportSpec',
    'SymmetrySpec',
    'TableSpec',
    'TableValuesSpec',
    'Tags',
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
