import logging
import time
from typing import List, Type, Optional, Dict, Tuple
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
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

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
    """Detected scenario types based on nuPlan definitions and empirical performance data"""
    # PDM-favorable scenarios (based on your data)
    FOLLOWING_LANE_WITH_LEAD = "following_lane_with_lead"  # PDM: 0.939
    HIGH_MAGNITUDE_SPEED = "high_magnitude_speed"  # PDM: 0.919
    BEHIND_LONG_VEHICLE = "behind_long_vehicle"  # PDM: 0.926
    STATIONARY_IN_TRAFFIC = "stationary_in_traffic"  # PDM: 0.868
    STOPPING_WITH_LEAD = "stopping_with_lead"  # PDM: 0.984
    WAITING_FOR_PEDESTRIAN = "waiting_for_pedestrian_to_cross"  # PDM: 0.882
    NEAR_MULTIPLE_VEHICLES = "near_multiple_vehicles"  # PDM: 0.659
    
    # Diffusion-favorable scenarios
    TRAVERSING_PICKUP_DROPOFF = "traversing_pickup_dropoff"  # Diffusion: 0.685
    STARTING_STRAIGHT_INTERSECTION = "starting_straight_traffic_light_intersection_traversal"  # Diffusion: 0.900
    STARTING_LEFT_TURN = "starting_left_turn"  # Diffusion: 0.670
    CHANGING_LANE = "changing_lane"  # Diffusion: 0.767
    HIGH_LATERAL_ACCELERATION = "high_lateral_acceleration"  # Diffusion: 0.573
    LOW_MAGNITUDE_SPEED = "low_magnitude_speed"  # Diffusion: 0.480
    
    # Neutral (close performance)
    STARTING_RIGHT_TURN = "starting_right_turn"  # Mixed results
    UNKNOWN = "unknown"


@dataclass
class ScenarioDetectionResult:
    """Result of scenario detection with confidence"""
    scenario_type: ScenarioType
    confidence: float  # 0-1
    bias: float  # -1 (strong PDM) to +1 (strong Diffusion)
    features: Dict[str, float]
    debug_info: str = ""


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


class EnhancedScenarioDetector:
    """Enhanced scenario detection based on nuPlan definitions"""
    
    def __init__(self, map_api):
        self.map_api = map_api
        # Empirical performance data from your results
        self.SCENARIO_PERFORMANCE = {
            # Strong PDM scenarios
            ScenarioType.STOPPING_WITH_LEAD: {'pdm': 0.984, 'diffusion': 0.976},
            ScenarioType.FOLLOWING_LANE_WITH_LEAD: {'pdm': 0.939, 'diffusion': 0.797},
            ScenarioType.BEHIND_LONG_VEHICLE: {'pdm': 0.926, 'diffusion': 0.846},
            ScenarioType.HIGH_MAGNITUDE_SPEED: {'pdm': 0.919, 'diffusion': 0.783},
            ScenarioType.WAITING_FOR_PEDESTRIAN: {'pdm': 0.882, 'diffusion': 0.494},
            ScenarioType.STATIONARY_IN_TRAFFIC: {'pdm': 0.868, 'diffusion': 0.678},
            ScenarioType.NEAR_MULTIPLE_VEHICLES: {'pdm': 0.659, 'diffusion': 0.505},
            
            # Strong Diffusion scenarios
            ScenarioType.STARTING_STRAIGHT_INTERSECTION: {'pdm': 0.874, 'diffusion': 0.900},
            ScenarioType.CHANGING_LANE: {'pdm': 0.730, 'diffusion': 0.767},
            ScenarioType.TRAVERSING_PICKUP_DROPOFF: {'pdm': 0.505, 'diffusion': 0.685},
            ScenarioType.STARTING_LEFT_TURN: {'pdm': 0.640, 'diffusion': 0.670},
            ScenarioType.HIGH_LATERAL_ACCELERATION: {'pdm': 0.547, 'diffusion': 0.573},
            ScenarioType.LOW_MAGNITUDE_SPEED: {'pdm': 0.432, 'diffusion': 0.480},
            
            # Neutral
            ScenarioType.STARTING_RIGHT_TURN: {'pdm': 0.693, 'diffusion': 0.577},
            ScenarioType.UNKNOWN: {'pdm': 0.5, 'diffusion': 0.5},
        }
    
    def detect_scenario(
        self, 
        ego_state, 
        observation, 
        traffic_light_data,
        route_lane_dict: Dict
    ) -> ScenarioDetectionResult:
        """
        Enhanced scenario detection following nuPlan definitions exactly.
        """
        features = {}
        debug_info = []
        
        # Extract ego motion features
        ego_speed = float(np.hypot(
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        ))
        ego_accel_x = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        ego_accel_y = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
        ego_accel_mag = float(np.hypot(ego_accel_x, ego_accel_y))
        lateral_accel = abs(ego_accel_y)
        longitudinal_accel = ego_accel_x
        
        features['ego_speed'] = ego_speed
        features['ego_accel_mag'] = ego_accel_mag
        features['lateral_accel'] = lateral_accel
        features['longitudinal_accel'] = longitudinal_accel
        features['yaw_rate'] = abs(ego_state.dynamic_car_state.angular_velocity)
        
        # Process tracked objects
        vehicles = []
        pedestrians = []
        leading_vehicle = None
        long_vehicles = []
        
        if hasattr(observation, 'tracked_objects'):
            for obj in observation.tracked_objects:
                # Get object position relative to ego
                rel_x = obj.center.x - ego_state.rear_axle.x
                rel_y = obj.center.y - ego_state.rear_axle.y
                
                # Transform to ego frame
                cos_h = np.cos(ego_state.rear_axle.heading)
                sin_h = np.sin(ego_state.rear_axle.heading)
                ego_x = rel_x * cos_h + rel_y * sin_h
                ego_y = -rel_x * sin_h + rel_y * cos_h
                
                distance = np.hypot(ego_x, ego_y)
                
                if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                    obj_speed = float(np.hypot(obj.velocity.x, obj.velocity.y))
                    vehicles.append({
                        'distance': distance,
                        'ego_x': ego_x,
                        'ego_y': ego_y,
                        'speed': obj_speed,
                        'length': obj.box.length,
                        'width': obj.box.width
                    })
                    
                    # Check for long vehicle (length > 8m)
                    if obj.box.length > 8.0:
                        long_vehicles.append(vehicles[-1])
                    
                    # Check for leading vehicle
                    if (ego_x > 0 and ego_x < 10.0 and  # In front, within 10m
                        abs(ego_y) < 0.5 and  # Same lane (lateral < 0.5m)
                        (leading_vehicle is None or ego_x < leading_vehicle['ego_x'])):
                        leading_vehicle = vehicles[-1]
                        
                elif obj.tracked_object_type in [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]:
                    pedestrians.append({
                        'distance': distance,
                        'ego_x': ego_x,
                        'ego_y': ego_y
                    })
        
        features['num_vehicles'] = len(vehicles)
        features['num_pedestrians'] = len(pedestrians)
        
        # Count nearby moving vehicles (distance < 8m, speed > 6 m/s)
        nearby_moving = sum(1 for v in vehicles if v['distance'] < 8.0 and v['speed'] > 6.0)
        features['nearby_moving_vehicles'] = nearby_moving
        
        # Check map features
        is_at_intersection = self._check_intersection(ego_state)
        is_at_crosswalk = self._check_crosswalk(ego_state)
        is_at_pickup_dropoff = self._check_pickup_dropoff(ego_state)
        
        features['at_intersection'] = is_at_intersection
        features['at_crosswalk'] = is_at_crosswalk
        features['at_pickup_dropoff'] = is_at_pickup_dropoff
        
        # Traffic light state
        has_traffic_light = len(traffic_light_data) > 0 if traffic_light_data else False
        red_light = False
        if traffic_light_data:
            for light in traffic_light_data:
                if hasattr(light, 'status') and light.status.name in ['RED', 'YELLOW']:
                    red_light = True
                    break
        
        features['has_traffic_light'] = has_traffic_light
        features['red_light'] = red_light
        
        # Now detect scenarios based on nuPlan definitions
        
        # 1. stopping_with_lead: Decel < -0.6 m/s², speed < 0.3 m/s, lead < 6m
        if (longitudinal_accel < -0.6 and ego_speed < 0.3 and 
            leading_vehicle and leading_vehicle['ego_x'] < 6.0):
            debug_info.append("Stopping with lead vehicle")
            return self._create_result(ScenarioType.STOPPING_WITH_LEAD, 0.9, features, debug_info)
        
        # 2. stationary_in_traffic: Stationary with >6 vehicles within 8m
        if ego_speed < 0.1 and len([v for v in vehicles if v['distance'] < 8.0]) > 6:
            debug_info.append("Stationary in traffic")
            return self._create_result(ScenarioType.STATIONARY_IN_TRAFFIC, 0.9, features, debug_info)
        
        # 3. waiting_for_pedestrian_to_cross: Near crosswalk, ped < 8m
        if (is_at_crosswalk and ego_speed > 0.1 and 
            any(p['distance'] < 8.0 for p in pedestrians)):
            debug_info.append("Waiting for pedestrian")
            return self._create_result(ScenarioType.WAITING_FOR_PEDESTRIAN, 0.85, features, debug_info)
        
        # 4. traversing_pickup_dropoff: In pickup/dropoff area, not stopped
        if is_at_pickup_dropoff and ego_speed > 0.1:
            debug_info.append("Traversing pickup/dropoff")
            return self._create_result(ScenarioType.TRAVERSING_PICKUP_DROPOFF, 0.9, features, debug_info)
        
        # 5. following_lane_with_lead: Speed > 3.5 m/s, lead < 7.5m, same lane
        if (ego_speed > 3.5 and leading_vehicle and 
            leading_vehicle['speed'] > 3.5 and 
            leading_vehicle['ego_x'] < 7.5 and 
            abs(leading_vehicle['ego_y']) < 0.5):
            debug_info.append("Following lane with lead")
            return self._create_result(ScenarioType.FOLLOWING_LANE_WITH_LEAD, 0.9, features, debug_info)
        
        # 6. behind_long_vehicle: 3m < dist < 10m, length > 8m, same lane
        if long_vehicles:
            for lv in long_vehicles:
                if (3.0 < lv['ego_x'] < 10.0 and abs(lv['ego_y']) < 0.5):
                    debug_info.append("Behind long vehicle")
                    return self._create_result(ScenarioType.BEHIND_LONG_VEHICLE, 0.9, features, debug_info)
        
        # 7. high_magnitude_speed: Speed > 9 m/s, low accel
        if ego_speed > 9.0 and ego_accel_mag < 1.0:
            debug_info.append("High magnitude speed")
            return self._create_result(ScenarioType.HIGH_MAGNITUDE_SPEED, 0.9, features, debug_info)
        
        # 8. low_magnitude_speed: 0.3 < speed < 1.2 m/s, low accel
        if 0.3 < ego_speed < 1.2 and ego_accel_mag < 1.0 and not is_at_pickup_dropoff:
            debug_info.append("Low magnitude speed")
            return self._create_result(ScenarioType.LOW_MAGNITUDE_SPEED, 0.8, features, debug_info)
        
        # 9. high_lateral_acceleration: 1.5 < accel < 3 m/s², high yaw, not turning
        if (1.5 < lateral_accel < 3.0 and features['yaw_rate'] > 0.2 and 
            not is_at_intersection):
            debug_info.append("High lateral acceleration")
            return self._create_result(ScenarioType.HIGH_LATERAL_ACCELERATION, 0.85, features, debug_info)
        
        # 10. changing_lane: Detect lane change (need more sophisticated check)
        if lateral_accel > 0.5 and ego_speed > 3.0 and features['yaw_rate'] < 0.1:
            debug_info.append("Possible lane change")
            return self._create_result(ScenarioType.CHANGING_LANE, 0.7, features, debug_info)
        
        # 11. near_multiple_vehicles: >6 vehicles < 8m, ego > 6 m/s
        if ego_speed > 6.0 and nearby_moving > 6:
            debug_info.append("Near multiple vehicles")
            return self._create_result(ScenarioType.NEAR_MULTIPLE_VEHICLES, 0.85, features, debug_info)
        
        # 12-14. Intersection scenarios
        if is_at_intersection and ego_speed > 0.1:
            # Determine turn direction based on yaw rate
            if features['yaw_rate'] > 0.15:  # Left turn
                debug_info.append("Starting left turn")
                return self._create_result(ScenarioType.STARTING_LEFT_TURN, 0.8, features, debug_info)
            elif features['yaw_rate'] < -0.15:  # Right turn
                debug_info.append("Starting right turn")
                return self._create_result(ScenarioType.STARTING_RIGHT_TURN, 0.8, features, debug_info)
            elif has_traffic_light:  # Straight through intersection
                debug_info.append("Straight through traffic light intersection")
                return self._create_result(ScenarioType.STARTING_STRAIGHT_INTERSECTION, 0.85, features, debug_info)
        
        # Default
        debug_info.append("No specific scenario detected")
        return self._create_result(ScenarioType.UNKNOWN, 0.5, features, debug_info)
    
    def _check_intersection(self, ego_state) -> bool:
        """Check if ego is at/near an intersection"""
        try:
            # Get nearby lane connectors (indicate intersections)
            connectors = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=15.0,
                layers=[SemanticMapLayer.LANE_CONNECTOR]
            )
            return len(connectors[SemanticMapLayer.LANE_CONNECTOR]) > 0
        except:
            return False
    
    def _check_crosswalk(self, ego_state) -> bool:
        """Check if ego is near a crosswalk"""
        try:
            crosswalks = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=10.0,
                layers=[SemanticMapLayer.CROSSWALK]
            )
            return len(crosswalks[SemanticMapLayer.CROSSWALK]) > 0
        except:
            return False
    
    def _check_pickup_dropoff(self, ego_state) -> bool:
        """Check if ego is in a pickup/dropoff area"""
        try:
            # Check for pickup/dropoff lanes or areas
            lanes = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=5.0,
                layers=[SemanticMapLayer.LANE]
            )
            
            # Check lane types for pickup/dropoff designation
            for lane in lanes[SemanticMapLayer.LANE]:
                # This is map-specific - adjust based on your map format
                if hasattr(lane, 'lane_type'):
                    if 'pickup' in str(lane.lane_type).lower() or 'dropoff' in str(lane.lane_type).lower():
                        return True
            
            # Alternative: check for specific map regions
            # You might need to adjust this based on your specific map API
            return False
        except:
            return False
    
    def _create_result(
        self, 
        scenario: ScenarioType, 
        confidence: float, 
        features: Dict,
        debug_info: List[str]
    ) -> ScenarioDetectionResult:
        """Create scenario detection result with bias calculation"""
        perf = self.SCENARIO_PERFORMANCE[scenario]
        # Convert win rate to bias: -1 (PDM) to +1 (Diffusion)
        # Amplify the bias for scenarios with large performance gaps
        pdm_win = perf['pdm']
        diff_win = perf['diffusion']
        
        # Calculate bias with amplification for large gaps
        raw_bias = diff_win - pdm_win
        if abs(raw_bias) > 0.1:  # Significant gap
            bias = np.sign(raw_bias) * min(abs(raw_bias) * 2.0, 1.0)
        else:
            bias = raw_bias
        
        return ScenarioDetectionResult(
            scenario_type=scenario,
            confidence=confidence,
            bias=bias,
            features=features,
            debug_info="; ".join(debug_info)
        )


class HybridPlanner(AbstractPlanner):
    """
    Hybrid planner with enhanced scenario detection for better switching.
    """
    
    requires_scenario: bool = False
    
    def __init__(
        self,
        pdm_planner: AbstractPlanner,
        diffusion_planner: AbstractPlanner,
        pdm_frequency: float = 10.0,
        diffusion_frequency: float = 2.0,
        leaky_ddm_config: Optional[Dict] = None,
        enable_scenario_detection: bool = True,
    ):
        """Initialize the hybrid planner"""
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
        self._scenario_detector = None
        
        # Initialize Leaky DDM switcher with enhanced config
        if leaky_ddm_config is None:
            leaky_ddm_config = {
                'scenario_scale': 0.25,  # Increased from 0.15
                'scenario_confidence_threshold': 0.7,  # Increased from 0.6
                'scenario_decay': 0.85,  # Slower decay
            }
        config = LeakyDDMConfig(**leaky_ddm_config)
        self._switcher = LeakyDDMSwitcher(config)
        
        # PDM scoring components
        self._scorer = None
        self._observation = None
        self._route_lane_dict = None
        self._drivable_area_map = None
        self._centerline = None
        self._map_api = None
        
        # Trajectory sampling
        self._trajectory_sampling = TrajectorySampling(num_poses=80, interval_length=0.1)
        self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        
        logger.info("Initialized HybridPlanner with enhanced scenario detection")
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize both planners and components"""
        logger.info("Initializing hybrid planner components...")
        
        # Store map API
        self._map_api = initialization.map_api
        
        # Initialize scenario detector
        self._scenario_detector = EnhancedScenarioDetector(self._map_api)
        
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
        
        # Initialize planners
        try:
            self.pdm_planner.initialize(initialization)
            logger.info("PDM planner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PDM planner: {e}")
            raise
        
        try:
            self.diffusion_planner.initialize(initialization)
            logger.info("Diffusion planner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Diffusion planner: {e}")
            traceback.print_exc()
            raise
        
        self._iteration = 0
        self._start_time = time.time()
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Main planning function with enhanced scenario detection"""
        self._current_planner_input = current_input
        current_time = time.time() - self._start_time if self._start_time else 0.0
        
        # Update observation
        ego_state, observation = current_input.history.current_state
        
        # Enhanced scenario detection
        scenario_result = None
        if self._enable_scenario_detection and self._scenario_detector:
            scenario_result = self._scenario_detector.detect_scenario(
                ego_state, 
                observation, 
                current_input.traffic_light_data,
                self._route_lane_dict
            )
            self._current_scenario = scenario_result.scenario_type
            
            # Log scenario detection
            if self._iteration % 10 == 0:
                logger.info(
                    f"Scenario: {scenario_result.scenario_type.value} "
                    f"(conf: {scenario_result.confidence:.2f}, bias: {scenario_result.bias:+.3f}) "
                    f"[{scenario_result.debug_info}]"
                )
        
        # Update PDM observation
        self._observation.update(
            ego_state=ego_state,
            observation=observation,
            traffic_light_data=current_input.traffic_light_data,
            route_lane_dict=self._route_lane_dict
        )
        
        # Update drivable area
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, map_radius=50.0
        )
        
        # Execute planners at their frequencies
        if self._iteration == 0 or current_time - self.last_pdm_time >= self.pdm_period:
            self._execute_pdm_planner(current_input, current_time, scenario_result)
        
        if self._iteration == 0 or current_time - self.last_diffusion_time >= self.diffusion_period:
            self._execute_diffusion_planner(current_input, current_time, scenario_result)
        
        # Select trajectory with enhanced scenario awareness
        selected_trajectory = self._select_trajectory_enhanced(scenario_result)
        
        self._iteration += 1
        return selected_trajectory
    
    def _execute_pdm_planner(self, current_input, current_time, scenario_result):
        """Execute PDM planner and score trajectory"""
        start = time.time()
        try:
            trajectory = self.pdm_planner.compute_planner_trajectory(current_input)
            computation_time = time.time() - start
            score_data = self._score_trajectory(trajectory, 'pdm')
            
            self.current_pdm_trajectory = TrajectoryWithMetadata(
                trajectory=trajectory,
                planner_name='pdm',
                computation_time=computation_time,
                timestamp=current_time,
                score=score_data['total_score'],
                progress=score_data['progress'],
                is_valid=score_data['is_valid'],
                scenario_type=scenario_result.scenario_type if scenario_result else None
            )
            self.last_pdm_time = current_time
            
        except Exception as e:
            logger.error(f"PDM planner failed: {e}")
            traceback.print_exc()
            raise
    
    def _execute_diffusion_planner(self, current_input, current_time, scenario_result):
        """Execute Diffusion planner and score trajectory"""
        start = time.time()
        try:
            trajectory = self.diffusion_planner.compute_planner_trajectory(current_input)
            computation_time = time.time() - start
            score_data = self._score_trajectory(trajectory, 'diffusion')
            
            self.current_diffusion_trajectory = TrajectoryWithMetadata(
                trajectory=trajectory,
                planner_name='diffusion',
                computation_time=computation_time,
                timestamp=current_time,
                score=score_data['total_score'],
                progress=score_data['progress'],
                is_valid=score_data['is_valid'],
                scenario_type=scenario_result.scenario_type if scenario_result else None
            )
            self.last_diffusion_time = current_time
            
        except Exception as e:
            logger.error(f"Diffusion planner failed: {e}")
    
    def _select_trajectory_enhanced(self, scenario_result: Optional[ScenarioDetectionResult]) -> AbstractTrajectory:
        """Enhanced trajectory selection with stronger scenario influence"""
        if self.current_pdm_trajectory is None:
            raise RuntimeError("No PDM trajectory available!")
        
        if self.current_diffusion_trajectory is None:
            return self.current_pdm_trajectory.trajectory
        
        # Get scores
        pdm_score = self.current_pdm_trajectory.score or 0.0
        diffusion_score = self.current_diffusion_trajectory.score or 0.0
        pdm_progress = self.current_pdm_trajectory.progress
        
        # Safety check
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
            
            # Log detailed switching info for critical scenarios
            if abs(scenario_bias) > 0.5 and scenario_confidence > 0.7:
                logger.info(
                    f"Strong scenario signal: {scenario_result.scenario_type.value} "
                    f"(bias={scenario_bias:+.3f}, conf={scenario_confidence:.2f})"
                )
        
        # Use enhanced DDM
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
                f"Decision: {selected_planner.value} | "
                f"Scenario: {scenario_result.scenario_type.value} | "
                f"Scores: PDM={pdm_score:.3f} Diff={diffusion_score:.3f} | "
                f"P={metadata['P']:.3f} S={metadata['S']:.3f}"
            )
        
        # Return selected trajectory
        if selected_planner == PlannerType.PDM:
            return self.current_pdm_trajectory.trajectory
        else:
            return self.current_diffusion_trajectory.trajectory
    
    # ... [Include all the other methods like _initialize_route_info, _score_trajectory, etc.]
    # These remain the same as before
    
    def name(self) -> str:
        return "HybridPlanner_EnhancedScenarioDetection"
    
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks
    
    def _initialize_route_info(self, route_roadblock_ids: List[str]) -> None:
        """Initialize route-related information"""
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
        
        if centerline_discrete_path:
            states_se2 = [StateSE2(point.x, point.y, point.heading) 
                         for point in centerline_discrete_path]
            self._centerline = PDMPath(states_se2)
        else:
            dummy_states = [StateSE2(0, 0, 0), StateSE2(10, 0, 0), StateSE2(20, 0, 0)]
            self._centerline = PDMPath(dummy_states)
    
    def _score_trajectory(self, trajectory: AbstractTrajectory, planner_name: str) -> Dict:
        """Score trajectory using PDM scorer"""
        try:
            states = self._trajectory_to_states(trajectory)
            ego_state = self._current_planner_input.history.current_state[0]
            
            scores = self._scorer.score_proposals(
                states=states,
                initial_ego_state=ego_state,
                observation=self._observation,
                centerline=self._centerline,
                route_lane_dict=self._route_lane_dict,
                drivable_area_map=self._drivable_area_map,
                map_api=self._map_api
            )
            
            total_score = float(scores[0])
            progress_raw = float(self._scorer._progress_raw[0])
            progress_normalized = float(self._scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 0])
            
            no_collision = float(self._scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 0])
            drivable_area = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 0])
            driving_direction = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, 0])
            
            is_valid = no_collision > 0.5 and drivable_area > 0.5 and driving_direction > 0.5
            
            return {
                'total_score': total_score,
                'progress': progress_normalized,
                'progress_raw': progress_raw,
                'is_valid': is_valid,
                'no_collision': no_collision,
                'drivable_area': drivable_area,
                'driving_direction': driving_direction,
            }
            
        except Exception as e:
            logger.error(f"Error scoring {planner_name} trajectory: {e}")
            return {
                'total_score': 0.0,
                'progress': 0.0,
                'progress_raw': 0.0,
                'is_valid': False,
                'no_collision': 0.0,
                'drivable_area': 0.0,
                'driving_direction': 0.0,
            }
    
    def _trajectory_to_states(self, trajectory: AbstractTrajectory) -> np.ndarray:
        """Convert trajectory to state array"""
        sampled_states = trajectory.get_sampled_trajectory()
        num_states = min(len(sampled_states), self._proposal_sampling.num_poses + 1)
        states = np.zeros((1, num_states, StateIndex.size()), dtype=np.float64)
        
        for i in range(num_states):
            if i < len(sampled_states):
                state = sampled_states[i]
                states[0, i, StateIndex.X] = state.rear_axle.x
                states[0, i, StateIndex.Y] = state.rear_axle.y
                states[0, i, StateIndex.HEADING] = state.rear_axle.heading
                states[0, i, StateIndex.VELOCITY_X] = state.dynamic_car_state.rear_axle_velocity_2d.x
                states[0, i, StateIndex.VELOCITY_Y] = state.dynamic_car_state.rear_axle_velocity_2d.y
                states[0, i, StateIndex.ACCELERATION_X] = state.dynamic_car_state.rear_axle_acceleration_2d.x
                states[0, i, StateIndex.ACCELERATION_Y] = state.dynamic_car_state.rear_axle_acceleration_2d.y
                states[0, i, StateIndex.STEERING_ANGLE] = state.tire_steering_angle
                states[0, i, StateIndex.STEERING_RATE] = 0.0
                states[0, i, StateIndex.ANGULAR_VELOCITY] = state.dynamic_car_state.angular_velocity
                states[0, i, StateIndex.ANGULAR_ACCELERATION] = state.dynamic_car_state.angular_acceleration
            else:
                states[0, i] = states[0, i-1]
        
        return states