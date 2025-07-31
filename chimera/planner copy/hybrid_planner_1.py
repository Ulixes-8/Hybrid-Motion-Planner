import logging
import time
from typing import List, Type, Optional, Dict
from dataclasses import dataclass
import traceback
import numpy as np
from enum import Enum

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, 
    PlannerInitialization, 
    PlannerInput
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# Import PDM components for scoring
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    StateIndex, MultiMetricIndex, WeightedMetricIndex
)

# Import our Leaky DDM switcher
from chimera.planner.leaky_ddm_switcher import LeakyDDMSwitcher, LeakyDDMConfig, PlannerType

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Detected scenario types based on empirical performance data"""
    # PDM-favorable scenarios
    FOLLOWING = "following"  # behind_long_vehicle, following_lane_with_lead
    HIGH_SPEED = "high_speed"  # high_magnitude_speed
    CONGESTED = "congested"  # near_multiple_vehicles, stationary_in_traffic
    STOPPING = "stopping"  # stopping_with_lead
    RIGHT_TURN = "right_turn"  # starting_right_turn
    PEDESTRIAN = "pedestrian"  # waiting_for_pedestrian_to_cross
    
    # Diffusion-favorable scenarios
    LANE_CHANGE = "lane_change"  # changing_lane
    HIGH_LATERAL = "high_lateral"  # high_lateral_acceleration
    LOW_SPEED_MANEUVER = "low_speed_maneuver"  # low_magnitude_speed
    LEFT_TURN = "left_turn"  # starting_left_turn
    INTERSECTION = "intersection"  # starting_straight_traffic_light_intersection_traversal
    PICKUP_DROPOFF = "pickup_dropoff"  # traversing_pickup_dropoff
    
    # Neutral
    UNKNOWN = "unknown"


@dataclass
class ScenarioDetectionResult:
    """Result of scenario detection"""
    scenario_type: ScenarioType
    confidence: float  # 0-1
    bias: float  # -1 (strong PDM) to +1 (strong Diffusion)
    features: Dict[str, float]


@dataclass
class TrajectoryWithMetadata:
    """Container for trajectory with additional metadata."""
    trajectory: AbstractTrajectory
    planner_name: str
    computation_time: float
    timestamp: float
    score: Optional[float] = None
    progress: Optional[float] = None
    is_valid: Optional[bool] = None
    scenario_type: Optional[ScenarioType] = None


class HybridPlanner(AbstractPlanner):
    """
    Hybrid planner using Leaky DDM switching between PDM-Closed and Diffusion Planner.
    Now enhanced with scenario-aware switching based on empirical performance data.
    """
    
    # Inherited property
    requires_scenario: bool = False
    
    # Empirical performance data (win rates from your analysis)
    SCENARIO_PERFORMANCE = {
        # PDM wins
        ScenarioType.FOLLOWING: {'pdm': 0.933, 'diffusion': 0.067},  # Avg of behind/following
        ScenarioType.HIGH_SPEED: {'pdm': 0.919, 'diffusion': 0.081},
        ScenarioType.CONGESTED: {'pdm': 0.764, 'diffusion': 0.236},  # Avg of multiple/stationary
        ScenarioType.STOPPING: {'pdm': 0.984, 'diffusion': 0.016},
        ScenarioType.RIGHT_TURN: {'pdm': 0.693, 'diffusion': 0.307},
        ScenarioType.PEDESTRIAN: {'pdm': 0.882, 'diffusion': 0.118},
        
        # Diffusion wins
        ScenarioType.LANE_CHANGE: {'pdm': 0.233, 'diffusion': 0.767},
        ScenarioType.HIGH_LATERAL: {'pdm': 0.427, 'diffusion': 0.573},
        ScenarioType.LOW_SPEED_MANEUVER: {'pdm': 0.520, 'diffusion': 0.480},
        ScenarioType.LEFT_TURN: {'pdm': 0.330, 'diffusion': 0.670},
        ScenarioType.INTERSECTION: {'pdm': 0.100, 'diffusion': 0.900},
        ScenarioType.PICKUP_DROPOFF: {'pdm': 0.315, 'diffusion': 0.685},
        
        # Unknown - no bias
        ScenarioType.UNKNOWN: {'pdm': 0.5, 'diffusion': 0.5},
    }
    
    def __init__(
        self,
        pdm_planner: AbstractPlanner,
        diffusion_planner: AbstractPlanner,
        pdm_frequency: float = 10.0,  # Hz
        diffusion_frequency: float = 2.0,  # Hz
        leaky_ddm_config: Optional[Dict] = None,
        enable_scenario_detection: bool = True,  # New parameter
    ):
        """
        Initialize the hybrid planner.
        
        :param pdm_planner: Instance of PDM-Closed planner
        :param diffusion_planner: Instance of Diffusion planner
        :param pdm_frequency: Frequency for PDM planner execution (Hz)
        :param diffusion_frequency: Frequency for Diffusion planner execution (Hz)
        :param leaky_ddm_config: Configuration for Leaky DDM switcher
        :param enable_scenario_detection: Whether to use scenario-aware switching
        """
        self.pdm_planner = pdm_planner
        self.diffusion_planner = diffusion_planner
        
        # Timing control
        self.pdm_period = 1.0 / pdm_frequency
        self.diffusion_period = 1.0 / diffusion_frequency
        self.last_pdm_time = 0.0
        self.last_diffusion_time = 0.0
        
        # Cache for trajectories
        self.current_pdm_trajectory: Optional[TrajectoryWithMetadata] = None
        self.current_diffusion_trajectory: Optional[TrajectoryWithMetadata] = None
        
        # State tracking
        self._iteration = 0
        self._start_time = None
        self._enable_scenario_detection = enable_scenario_detection
        self._current_scenario = ScenarioType.UNKNOWN
        
        # Initialize Leaky DDM switcher
        config = LeakyDDMConfig(**leaky_ddm_config) if leaky_ddm_config else LeakyDDMConfig()
        self._switcher = LeakyDDMSwitcher(config)
        
        # PDM scoring components
        self._scorer = None
        self._observation = None
        self._route_lane_dict = None
        self._drivable_area_map = None
        self._centerline = None
        self._map_api = None
        
        # Trajectory sampling for scoring
        self._trajectory_sampling = TrajectorySampling(
            num_poses=80,
            interval_length=0.1
        )
        self._proposal_sampling = TrajectorySampling(
            num_poses=40,
            interval_length=0.1
        )

        logger.info(f"Initialized HybridPlanner with Leaky DDM switching (scenario detection: {enable_scenario_detection})")
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize both planners."""
        logger.info("Initializing hybrid planner components...")
        
        # Store map API
        self._map_api = initialization.map_api
        
        # Initialize PDM scorer
        self._scorer = PDMScorer(proposal_sampling=self._proposal_sampling)
        
        # Initialize route information
        self._initialize_route_info(initialization.route_roadblock_ids)
        
        # Initialize PDM observation
        self._observation = PDMObservation(
            trajectory_sampling=self._trajectory_sampling,
            proposal_sampling=self._proposal_sampling,
            map_radius=50.0
        )
        
        # Initialize PDM
        try:
            self.pdm_planner.initialize(initialization)
            logger.info("PDM planner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PDM planner: {e}")
            raise
        
        # Initialize Diffusion
        try:
            self.diffusion_planner.initialize(initialization)
            logger.info("Diffusion planner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Diffusion planner: {e}")
            traceback.print_exc()
            raise
        
        self._iteration = 0
        self._start_time = time.time()
    
    def _initialize_route_info(self, route_roadblock_ids: List[str]) -> None:
        """Initialize route-related information for scoring."""
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import StateSE2
        
        self._route_lane_dict = {}
        centerline_discrete_path = []
        
        for id_ in route_roadblock_ids:
            try:
                block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
                if block:
                    for edge in block.interior_edges:
                        self._route_lane_dict[edge.id] = edge
                        centerline_discrete_path.extend(edge.baseline_path.discrete_path)
            except Exception as e:
                logger.warning(f"Could not get roadblock {id_}: {e}")
                continue
        
        # Create centerline path
        if centerline_discrete_path:
            states_se2 = [
                StateSE2(point.x, point.y, point.heading) 
                for point in centerline_discrete_path
            ]
            self._centerline = PDMPath(states_se2)
            logger.info(f"Created centerline with {len(states_se2)} points")
        else:
            logger.warning("No centerline path could be created from route")
            dummy_states = [StateSE2(0, 0, 0), StateSE2(10, 0, 0), StateSE2(20, 0, 0)]
            self._centerline = PDMPath(dummy_states)
    
    def _detect_scenario(self, ego_state, observation, traffic_light_data) -> ScenarioDetectionResult:
        """
        Detect the current driving scenario based on state and environment.
        """
        features = {}
        
        # Extract ego motion features
        ego_speed = float(np.hypot(
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        ))
        features['ego_speed'] = ego_speed
        
        # Lateral acceleration (indicator of lane change/turning)
        lateral_accel = abs(ego_state.dynamic_car_state.rear_axle_acceleration_2d.y)
        features['lateral_accel'] = lateral_accel
        
        # Steering angle (turning indicator)
        steering_angle = abs(ego_state.tire_steering_angle)
        features['steering_angle'] = steering_angle
        
        # Count nearby vehicles
        nearby_vehicles = 0
        leading_vehicle_dist = float('inf')
        
        if hasattr(observation, 'tracked_objects'):
            for obj in observation.tracked_objects:
                if obj.tracked_object_type.name == 'VEHICLE':
                    nearby_vehicles += 1
                    
                    # Check for leading vehicle
                    rel_x = obj.center.x - ego_state.rear_axle.x
                    rel_y = obj.center.y - ego_state.rear_axle.y
                    # Transform to ego frame
                    cos_h = np.cos(ego_state.rear_axle.heading)
                    sin_h = np.sin(ego_state.rear_axle.heading)
                    ego_x = rel_x * cos_h + rel_y * sin_h
                    ego_y = -rel_x * sin_h + rel_y * cos_h
                    
                    # Check if vehicle is ahead and in same lane
                    if ego_x > 0 and abs(ego_y) < 2.0:  # roughly in same lane
                        dist = np.hypot(ego_x, ego_y)
                        leading_vehicle_dist = min(leading_vehicle_dist, dist)
        
        features['num_nearby_vehicles'] = nearby_vehicles
        features['leading_vehicle_dist'] = leading_vehicle_dist
        
        # Traffic light state
        has_red_light = False
        if traffic_light_data:
            for light in traffic_light_data:
                if hasattr(light, 'status') and light.status.name in ['RED', 'YELLOW']:
                    has_red_light = True
                    break
        features['has_red_light'] = has_red_light
        
        # Classify scenario
        scenario = ScenarioType.UNKNOWN
        confidence = 0.5
        
        # Decision tree for scenario classification
        if ego_speed > 15.0:  # m/s (~54 km/h)
            scenario = ScenarioType.HIGH_SPEED
            confidence = min(ego_speed / 25.0, 0.9)
        
        elif ego_speed < 5.0 and lateral_accel > 0.5:
            scenario = ScenarioType.LOW_SPEED_MANEUVER
            confidence = 0.8
        
        elif lateral_accel > 2.0 and ego_speed > 8.0:
            scenario = ScenarioType.LANE_CHANGE
            confidence = min(lateral_accel / 3.0, 0.9)
        
        elif steering_angle > 0.2:  # radians
            # Simplified turning detection
            if ego_state.dynamic_car_state.angular_velocity > 0:
                scenario = ScenarioType.LEFT_TURN
            else:
                scenario = ScenarioType.RIGHT_TURN
            confidence = 0.7
        
        elif leading_vehicle_dist < 50.0 and ego_speed > 5.0:
            scenario = ScenarioType.FOLLOWING
            confidence = 0.8
        
        elif nearby_vehicles > 5:
            scenario = ScenarioType.CONGESTED
            confidence = min(nearby_vehicles / 10.0, 0.9)
        
        elif ego_speed < 2.0 and (has_red_light or leading_vehicle_dist < 20.0):
            scenario = ScenarioType.STOPPING
            confidence = 0.85
        
        elif has_red_light and ego_speed < 10.0:
            scenario = ScenarioType.INTERSECTION
            confidence = 0.8
        
        # Calculate planner bias based on empirical performance
        perf = self.SCENARIO_PERFORMANCE[scenario]
        # Convert win rate to bias: -1 (PDM) to +1 (Diffusion)
        bias = 2.0 * perf['diffusion'] - 1.0
        
        return ScenarioDetectionResult(
            scenario_type=scenario,
            confidence=confidence,
            bias=bias,
            features=features
        )
    
    def name(self) -> str:
        """Return planner name."""
        return "HybridPlanner_LeakyDDM"
    
    def observation_type(self) -> Type[Observation]:
        """Return observation type."""
        return DetectionsTracks
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Main planning function with dual-frequency execution and Leaky DDM switching.
        """
        # Store current input
        self._current_planner_input = current_input
        
        # Get current time relative to start
        current_time = time.time() - self._start_time if self._start_time else 0.0
        
        # Update observation for scoring
        ego_state, observation = current_input.history.current_state
        
        # Detect scenario if enabled
        scenario_result = None
        if self._enable_scenario_detection:
            scenario_result = self._detect_scenario(
                ego_state, 
                observation, 
                current_input.traffic_light_data
            )
            self._current_scenario = scenario_result.scenario_type
            
            # Log scenario detection occasionally
            if self._iteration % 10 == 0:
                logger.info(
                    f"Detected scenario: {scenario_result.scenario_type.value} "
                    f"(confidence: {scenario_result.confidence:.2f}, bias: {scenario_result.bias:+.2f})"
                )
        
        # Update PDM observation
        self._observation.update(
            ego_state=ego_state,
            observation=observation,
            traffic_light_data=current_input.traffic_light_data,
            route_lane_dict=self._route_lane_dict
        )
        
        # Update drivable area map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, map_radius=50.0
        )
        
        # Check if we should run PDM planner
        if self._iteration == 0 or current_time - self.last_pdm_time >= self.pdm_period:
            start = time.time()
            try:
                trajectory = self.pdm_planner.compute_planner_trajectory(current_input)
                computation_time = time.time() - start
                
                # Score the trajectory
                score_data = self._score_trajectory(trajectory, 'pdm')
                
                self.current_pdm_trajectory = TrajectoryWithMetadata(
                    trajectory=trajectory,
                    planner_name='pdm',
                    computation_time=computation_time,
                    timestamp=current_time,
                    score=score_data['total_score'],
                    progress=score_data['progress'],
                    is_valid=score_data['is_valid'],
                    scenario_type=self._current_scenario if scenario_result else None
                )
                self.last_pdm_time = current_time
                
                logger.debug(f"PDM trajectory computed in {computation_time:.3f}s, score: {score_data['total_score']:.3f}")
            except Exception as e:
                logger.error(f"PDM planner failed: {e}")
                traceback.print_exc()
                raise
                
        # Check if we should run Diffusion planner
        if self._iteration == 0 or current_time - self.last_diffusion_time >= self.diffusion_period:
            start = time.time()
            try:
                trajectory = self.diffusion_planner.compute_planner_trajectory(current_input)
                computation_time = time.time() - start
                
                # Score the trajectory
                score_data = self._score_trajectory(trajectory, 'diffusion')
                
                self.current_diffusion_trajectory = TrajectoryWithMetadata(
                    trajectory=trajectory,
                    planner_name='diffusion',
                    computation_time=computation_time,
                    timestamp=current_time,
                    score=score_data['total_score'],
                    progress=score_data['progress'],
                    is_valid=score_data['is_valid'],
                    scenario_type=self._current_scenario if scenario_result else None
                )
                self.last_diffusion_time = current_time
                
                logger.info(f"Diffusion trajectory computed in {computation_time:.3f}s, score: {score_data['total_score']:.3f}")
            except Exception as e:
                logger.error(f"Diffusion planner failed: {e}")
                # Continue with PDM on diffusion failure
                
        # Select trajectory using Leaky DDM
        selected_trajectory = self._select_trajectory_leaky_ddm(scenario_result)
        
        self._iteration += 1
        
        return selected_trajectory
    
    def _score_trajectory(self, trajectory: AbstractTrajectory, planner_name: str) -> Dict:
        """Score a trajectory using PDM scorer."""
        try:
            # Convert trajectory to states for PDM scorer
            states = self._trajectory_to_states(trajectory)
            
            # Get current ego state
            ego_state = self._current_planner_input.history.current_state[0]
            
            # Score using PDM scorer
            scores = self._scorer.score_proposals(
                states=states,
                initial_ego_state=ego_state,
                observation=self._observation,
                centerline=self._centerline,
                route_lane_dict=self._route_lane_dict,
                drivable_area_map=self._drivable_area_map,
                map_api=self._map_api
            )
            
            # Extract individual metrics
            total_score = float(scores[0])
            progress_raw = float(self._scorer._progress_raw[0])
            
            # Get normalized progress from weighted metrics
            progress_normalized = float(self._scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 0])
            
            # Get detailed metrics
            no_collision = float(self._scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 0])
            drivable_area = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 0])
            driving_direction = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, 0])
            
            ttc = float(self._scorer._weighted_metrics[WeightedMetricIndex.TTC, 0])
            comfort = float(self._scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, 0])
            
            # Check validity (all multiplicative metrics must pass)
            is_valid = no_collision > 0.5 and drivable_area > 0.5 and driving_direction > 0.5
            
            return {
                'total_score': total_score,
                'progress': progress_normalized,  # Use normalized progress for trapped detection
                'progress_raw': progress_raw,
                'is_valid': is_valid,
                'no_collision': no_collision,
                'drivable_area': drivable_area,
                'driving_direction': driving_direction,
                'ttc': ttc,
                'comfort': comfort
            }
            
        except Exception as e:
            logger.error(f"Error scoring {planner_name} trajectory: {e}")
            traceback.print_exc()
            # Return zero scores on error
            return {
                'total_score': 0.0,
                'progress': 0.0,
                'progress_raw': 0.0,
                'is_valid': False,
                'no_collision': 0.0,
                'drivable_area': 0.0,
                'driving_direction': 0.0,
                'ttc': 0.0,
                'comfort': 0.0
            }
    
    def _trajectory_to_states(self, trajectory: AbstractTrajectory) -> np.ndarray:
        """Convert trajectory to state array for PDM scorer."""
        # Sample trajectory
        sampled_states = trajectory.get_sampled_trajectory()
        
        # Create state array with shape (1, num_poses+1, state_dim)
        num_states = min(len(sampled_states), self._proposal_sampling.num_poses + 1)
        states = np.zeros((1, num_states, StateIndex.size()), dtype=np.float64)
        
        for i in range(num_states):
            if i < len(sampled_states):
                state = sampled_states[i]
                # Position and heading
                states[0, i, StateIndex.X] = state.rear_axle.x
                states[0, i, StateIndex.Y] = state.rear_axle.y
                states[0, i, StateIndex.HEADING] = state.rear_axle.heading
                
                # Velocities
                states[0, i, StateIndex.VELOCITY_X] = state.dynamic_car_state.rear_axle_velocity_2d.x
                states[0, i, StateIndex.VELOCITY_Y] = state.dynamic_car_state.rear_axle_velocity_2d.y
                
                # Accelerations
                states[0, i, StateIndex.ACCELERATION_X] = state.dynamic_car_state.rear_axle_acceleration_2d.x
                states[0, i, StateIndex.ACCELERATION_Y] = state.dynamic_car_state.rear_axle_acceleration_2d.y
                
                # Steering
                states[0, i, StateIndex.STEERING_ANGLE] = state.tire_steering_angle
                states[0, i, StateIndex.STEERING_RATE] = 0.0
                
                # Angular motion
                states[0, i, StateIndex.ANGULAR_VELOCITY] = state.dynamic_car_state.angular_velocity
                states[0, i, StateIndex.ANGULAR_ACCELERATION] = state.dynamic_car_state.angular_acceleration
            else:
                # Pad with last state
                states[0, i] = states[0, i-1]
        
        return states
    
    def _select_trajectory_leaky_ddm(self, scenario_result: Optional[ScenarioDetectionResult]) -> AbstractTrajectory:
        """Select trajectory using Leaky DDM switcher."""
        if self.current_pdm_trajectory is None:
            raise RuntimeError("No PDM trajectory available!")
        
        # If no diffusion trajectory, use PDM
        if self.current_diffusion_trajectory is None:
            return self.current_pdm_trajectory.trajectory
        
        # Get scores
        pdm_score = self.current_pdm_trajectory.score or 0.0
        diffusion_score = self.current_diffusion_trajectory.score or 0.0
        pdm_progress = self.current_pdm_trajectory.progress
        
        # Check if diffusion was vetoed by safety
        safety_vetoed = (
            self.current_diffusion_trajectory.is_valid is False and
            self.current_diffusion_trajectory.score < 0.1
        )
        
        # Prepare scenario parameters
        scenario_bias = None
        scenario_confidence = None
        if scenario_result is not None:
            scenario_bias = scenario_result.bias
            scenario_confidence = scenario_result.confidence
        
        # Use Leaky DDM to select planner
        selected_planner, metadata = self._switcher.update_and_select(
            pdm_score=pdm_score,
            diffusion_score=diffusion_score,
            pdm_progress=pdm_progress,
            safety_vetoed=safety_vetoed,
            scenario_bias=scenario_bias,
            scenario_confidence=scenario_confidence
        )
        
        # Log decision
        if scenario_result:
            logger.info(
                f"LeakyDDM decision: {selected_planner.value} | "
                f"Scenario: {scenario_result.scenario_type.value} | "
                f"P={metadata['P']:.3f} S={metadata['S']:.3f} | "
                f"PDM={pdm_score:.3f} Diff={diffusion_score:.3f} | "
                f"Bias={scenario_result.bias:+.3f}"
            )
        else:
            logger.info(
                f"LeakyDDM decision: {selected_planner.value} | "
                f"P={metadata['P']:.3f} | "
                f"PDM={pdm_score:.3f} Diff={diffusion_score:.3f}"
            )
        
        # Return selected trajectory
        if selected_planner == PlannerType.PDM:
            return self.current_pdm_trajectory.trajectory
        else:
            return self.current_diffusion_trajectory.trajectory