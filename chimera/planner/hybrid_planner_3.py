"""
Focused Hybrid Planner with Precise Scenario Detection
Prioritizes accurate scenario recognition to switch to the empirically better planner.
"""

import logging
import time
from typing import List, Type, Optional, Dict, Tuple
from dataclasses import dataclass
import traceback
import numpy as np

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, 
    PlannerInitialization, 
    PlannerInput
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.state_representation import StateSE2

# Import PDM components
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    StateIndex, MultiMetricIndex, WeightedMetricIndex
)

# Import our focused switcher
from chimera.planner.leaky_ddm_switcher import (
    LeakyDDMSwitcher, LeakyDDMConfig, PlannerType, ScenarioType
)

logger = logging.getLogger(__name__)


@dataclass
class ScenarioDetectionResult:
    """Result of scenario detection"""
    scenario_type: ScenarioType
    confidence: float
    features: Dict[str, float]
    debug_info: str = ""


@dataclass
class TrajectoryWithMetadata:
    """Container for trajectory with metadata"""
    trajectory: AbstractTrajectory
    planner_name: str
    computation_time: float
    timestamp: float
    score: Optional[float] = None
    progress: Optional[float] = None


class FocusedScenarioDetector:
    """
    Precise scenario detection matching nuPlan definitions exactly.
    Focus on detecting the 14 scenario types accurately.
    """
    
    def __init__(self, map_api):
        self.map_api = map_api
        
    def detect_scenario(
        self, 
        ego_state, 
        observation, 
        traffic_light_data,
        route_lane_dict: Dict
    ) -> ScenarioDetectionResult:
        """
        Detect scenario following exact nuPlan definitions.
        Priority order matters - check most specific scenarios first.
        """
        # Extract all features we need
        features = self._extract_features(ego_state, observation, traffic_light_data)
        
        # 1. STOPPING_WITH_LEAD (highest priority - safety critical)
        # "Decel < -0.6 m/s², velocity < 0.3 m/s, lead < 6m"
        if (features['longitudinal_accel'] < -0.6 and 
            features['ego_speed'] < 0.3 and
            features['leading_vehicle'] is not None and
            features['leading_vehicle']['distance'] < 6.0):
            return ScenarioDetectionResult(
                scenario_type=ScenarioType.STOPPING_WITH_LEAD,
                confidence=0.95,
                features=features,
                debug_info="Stopping with lead vehicle"
            )
        
        # 2. STATIONARY_IN_TRAFFIC
        # "Stationary with >6 vehicles within 8m"
        if features['ego_speed'] < 0.1 and features['vehicles_within_8m'] > 6:
            return ScenarioDetectionResult(
                scenario_type=ScenarioType.STATIONARY_IN_TRAFFIC,
                confidence=0.95,
                features=features,
                debug_info=f"Stationary with {features['vehicles_within_8m']} vehicles"
            )
        
        # 3. WAITING_FOR_PEDESTRIAN_TO_CROSS
        # "Distance < 8m, time to intersection < 1.5s, ego not stopped, near crosswalk"
        if features['ego_speed'] > 0.1 and features['near_crosswalk']:
            for ped in features['pedestrians']:
                if ped['distance'] < 8.0:
                    tti = self._calculate_time_to_intersection(ego_state, ped, features['ego_speed'])
                    if tti < 1.5:
                        features['min_pedestrian_distance'] = ped['distance']
                        return ScenarioDetectionResult(
                            scenario_type=ScenarioType.WAITING_FOR_PEDESTRIAN,
                            confidence=0.9,
                            features=features,
                            debug_info=f"Pedestrian at {ped['distance']:.1f}m, TTI={tti:.1f}s"
                        )
        
        # 4. FOLLOWING_LANE_WITH_LEAD
        # "Speed > 3.5 m/s, lead speed > 3.5 m/s, distance < 7.5m, same lane"
        if (features['ego_speed'] > 3.5 and features['leading_vehicle'] is not None):
            lead = features['leading_vehicle']
            if (lead['speed'] > 3.5 and 
                lead['distance'] < 7.5 and
                abs(lead['lateral_offset']) < 0.5):
                return ScenarioDetectionResult(
                    scenario_type=ScenarioType.FOLLOWING_LANE_WITH_LEAD,
                    confidence=0.95,
                    features=features,
                    debug_info=f"Following at {lead['distance']:.1f}m"
                )
        
        # 5. BEHIND_LONG_VEHICLE
        # "3m < longitudinal distance < 10m, length > 8m, same lane"
        for vehicle in features['vehicles']:
            if (vehicle['length'] > 8.0 and
                3.0 < vehicle['longitudinal_distance'] < 10.0 and
                abs(vehicle['lateral_offset']) < 0.5):
                return ScenarioDetectionResult(
                    scenario_type=ScenarioType.BEHIND_LONG_VEHICLE,
                    confidence=0.9,
                    features=features,
                    debug_info=f"Behind {vehicle['length']:.1f}m vehicle"
                )
        
        # 6. TRAVERSING_PICKUP_DROPOFF
        # "In pickup/dropoff area while not stopped"
        if features['in_pickup_dropoff'] and features['ego_speed'] > 0.1:
            return ScenarioDetectionResult(
                scenario_type=ScenarioType.TRAVERSING_PICKUP_DROPOFF,
                confidence=0.9,
                features=features,
                debug_info="Traversing pickup/dropoff area"
            )
        
        # 7. HIGH_MAGNITUDE_SPEED
        # "Speed > 9 m/s with low acceleration"
        if features['ego_speed'] > 9.0 and features['acceleration_magnitude'] < 1.0:
            return ScenarioDetectionResult(
                scenario_type=ScenarioType.HIGH_MAGNITUDE_SPEED,
                confidence=0.95,
                features=features,
                debug_info=f"High speed: {features['ego_speed']:.1f} m/s"
            )
        
        # 8. LOW_MAGNITUDE_SPEED
        # "0.3 < speed < 1.2 m/s with low acceleration, not in pickup/dropoff"
        if (0.3 < features['ego_speed'] < 1.2 and 
            features['acceleration_magnitude'] < 1.0 and
            not features['in_pickup_dropoff']):
            return ScenarioDetectionResult(
                scenario_type=ScenarioType.LOW_MAGNITUDE_SPEED,
                confidence=0.85,
                features=features,
                debug_info="Low speed maneuver"
            )
        
        # 9. HIGH_LATERAL_ACCELERATION
        # "1.5 < lateral_accel < 3 m/s² with high yaw rate, not turning"
        if (1.5 < features['lateral_accel'] < 3.0 and 
            features['yaw_rate'] > 0.2 and
            not features['at_intersection']):
            return ScenarioDetectionResult(
                scenario_type=ScenarioType.HIGH_LATERAL_ACCELERATION,
                confidence=0.85,
                features=features,
                debug_info=f"High lateral: {features['lateral_accel']:.2f} m/s²"
            )
        
        # 10. CHANGING_LANE
        # "At the start of a lane change"
        if self._detect_lane_change(features):
            return ScenarioDetectionResult(
                scenario_type=ScenarioType.CHANGING_LANE,
                confidence=0.85,
                features=features,
                debug_info="Lane change detected"
            )
        
        # 11. NEAR_MULTIPLE_VEHICLES
        # "Nearby (distance < 8m) >6 moving vehicles, ego speed > 6 m/s"
        moving_nearby = sum(1 for v in features['vehicles'] 
                          if v['distance'] < 8.0 and v['speed'] > 6.0)
        if features['ego_speed'] > 6.0 and moving_nearby > 6:
            return ScenarioDetectionResult(
                scenario_type=ScenarioType.NEAR_MULTIPLE_VEHICLES,
                confidence=0.85,
                features=features,
                debug_info=f"{moving_nearby} moving vehicles nearby"
            )
        
        # 12-14. INTERSECTION SCENARIOS
        # "At the start of traversal at intersection"
        if features['at_intersection'] and features['ego_speed'] > 0.1:
            # Determine turn type based on yaw rate and steering
            if features['yaw_rate'] > 0.15 or features['steering_angle'] > 0.1:
                return ScenarioDetectionResult(
                    scenario_type=ScenarioType.STARTING_LEFT_TURN,
                    confidence=0.85,
                    features=features,
                    debug_info="Starting left turn"
                )
            elif features['yaw_rate'] < -0.15 or features['steering_angle'] < -0.1:
                return ScenarioDetectionResult(
                    scenario_type=ScenarioType.STARTING_RIGHT_TURN,
                    confidence=0.85,
                    features=features,
                    debug_info="Starting right turn"
                )
            elif features['has_traffic_light']:
                return ScenarioDetectionResult(
                    scenario_type=ScenarioType.STARTING_STRAIGHT_INTERSECTION,
                    confidence=0.85,
                    features=features,
                    debug_info="Straight through traffic light"
                )
        
        # Default
        return ScenarioDetectionResult(
            scenario_type=ScenarioType.UNKNOWN,
            confidence=0.5,
            features=features,
            debug_info="No specific scenario"
        )
    
    def _extract_features(self, ego_state, observation, traffic_light_data) -> Dict[str, float]:
        """Extract all features needed for scenario detection"""
        features = {}
        
        # Ego motion
        features['ego_speed'] = float(np.hypot(
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        ))
        features['longitudinal_accel'] = float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x)
        features['lateral_accel'] = float(abs(ego_state.dynamic_car_state.rear_axle_acceleration_2d.y))
        features['acceleration_magnitude'] = float(np.hypot(
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
        ))
        features['yaw_rate'] = float(abs(ego_state.dynamic_car_state.angular_velocity))
        features['steering_angle'] = float(ego_state.tire_steering_angle)
        
        # Process tracked objects
        vehicles = []
        pedestrians = []
        leading_vehicle = None
        vehicles_within_8m = 0
        
        if hasattr(observation, 'tracked_objects'):
            for obj in observation.tracked_objects:
                # Transform to ego frame
                rel_x = obj.center.x - ego_state.rear_axle.x
                rel_y = obj.center.y - ego_state.rear_axle.y
                
                cos_h = np.cos(ego_state.rear_axle.heading)
                sin_h = np.sin(ego_state.rear_axle.heading)
                ego_x = rel_x * cos_h + rel_y * sin_h
                ego_y = -rel_x * sin_h + rel_y * cos_h
                
                distance = float(np.hypot(ego_x, ego_y))
                
                if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                    obj_speed = float(np.hypot(obj.velocity.x, obj.velocity.y))
                    
                    vehicle_info = {
                        'distance': distance,
                        'longitudinal_distance': ego_x,
                        'lateral_offset': ego_y,
                        'speed': obj_speed,
                        'length': float(obj.box.length),
                        'width': float(obj.box.width)
                    }
                    vehicles.append(vehicle_info)
                    
                    if distance < 8.0:
                        vehicles_within_8m += 1
                    
                    # Leading vehicle (ahead in same lane)
                    if (ego_x > 0 and ego_x < 20.0 and abs(ego_y) < 1.0):
                        if leading_vehicle is None or ego_x < leading_vehicle['distance']:
                            leading_vehicle = vehicle_info
                            
                elif obj.tracked_object_type in [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]:
                    pedestrians.append({
                        'distance': distance,
                        'position': {'x': ego_x, 'y': ego_y},
                        'velocity': {'x': obj.velocity.x, 'y': obj.velocity.y}
                    })
        
        features['vehicles'] = vehicles
        features['pedestrians'] = pedestrians
        features['leading_vehicle'] = leading_vehicle
        features['vehicles_within_8m'] = vehicles_within_8m
        
        # Map features
        features['at_intersection'] = self._check_intersection(ego_state)
        features['near_crosswalk'] = self._check_crosswalk(ego_state)
        features['in_pickup_dropoff'] = self._check_pickup_dropoff(ego_state)
        
        # Traffic light
        features['has_traffic_light'] = len(traffic_light_data) > 0 if traffic_light_data else False
        
        # TTC for safety
        if leading_vehicle:
            features['ttc'] = self._calculate_ttc(ego_state, leading_vehicle, features['ego_speed'])
        else:
            features['ttc'] = float('inf')
        
        return features
    
    def _detect_lane_change(self, features) -> bool:
        """Detect if ego is changing lanes"""
        # Lane change indicators:
        # - Moderate lateral acceleration
        # - Speed > 3 m/s
        # - Low yaw rate (not turning)
        # - Lateral velocity component
        return (features['lateral_accel'] > 0.5 and 
                features['ego_speed'] > 3.0 and
                features['yaw_rate'] < 0.15)
    
    def _calculate_time_to_intersection(self, ego_state, pedestrian, ego_speed) -> float:
        """Calculate time until paths intersect"""
        if ego_speed < 0.1:
            return float('inf')
        return pedestrian['distance'] / max(ego_speed, 0.1)
    
    def _calculate_ttc(self, ego_state, lead_vehicle, ego_speed) -> float:
        """Calculate time to collision"""
        relative_speed = ego_speed - lead_vehicle['speed']
        if relative_speed <= 0:
            return float('inf')
        return lead_vehicle['distance'] / relative_speed
    
    def _check_intersection(self, ego_state) -> bool:
        """Check if at intersection"""
        try:
            connectors = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=15.0,
                layers=[SemanticMapLayer.LANE_CONNECTOR]
            )
            return len(connectors.get(SemanticMapLayer.LANE_CONNECTOR, [])) > 0
        except:
            return False
    
    def _check_crosswalk(self, ego_state) -> bool:
        """Check if near crosswalk"""
        try:
            crosswalks = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=10.0,
                layers=[SemanticMapLayer.CROSSWALK]
            )
            return len(crosswalks.get(SemanticMapLayer.CROSSWALK, [])) > 0
        except:
            return False
    
    def _check_pickup_dropoff(self, ego_state) -> bool:
        """Check if in pickup/dropoff area"""
        try:
            lanes = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=5.0,
                layers=[SemanticMapLayer.LANE]
            )
            
            for lane in lanes.get(SemanticMapLayer.LANE, []):
                # Check for pickup/dropoff designation
                if hasattr(lane, 'lane_type_fid'):
                    lane_type_str = str(lane.lane_type_fid).lower()
                    if any(keyword in lane_type_str for keyword in 
                          ['pickup', 'dropoff', 'loading', 'passenger']):
                        return True
                # Alternative: check lane ID patterns
                if hasattr(lane, 'id'):
                    lane_id_str = str(lane.id).lower()
                    if any(keyword in lane_id_str for keyword in 
                          ['pickup', 'dropoff', 'loading']):
                        return True
            
            return False
        except:
            return False


class HybridPlanner(AbstractPlanner):
    """
    Focused Hybrid Planner that switches based on scenario detection.
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
        """Initialize focused hybrid planner"""
        self.pdm_planner = pdm_planner
        self.diffusion_planner = diffusion_planner
        
        # Timing
        self.pdm_period = 1.0 / pdm_frequency
        self.diffusion_period = 1.0 / diffusion_frequency
        self.last_pdm_time = 0.0
        self.last_diffusion_time = 0.0
        
        # Trajectory cache
        self.current_pdm_trajectory: Optional[TrajectoryWithMetadata] = None
        self.current_diffusion_trajectory: Optional[TrajectoryWithMetadata] = None
        
        # State
        self._iteration = 0
        self._start_time = None
        self._enable_scenario_detection = enable_scenario_detection
        self._scenario_detector = None
        
        # Initialize focused switcher
        config = LeakyDDMConfig(**leaky_ddm_config) if leaky_ddm_config else LeakyDDMConfig()
        self._switcher = LeakyDDMSwitcher(config)
        
        # PDM components for scoring
        self._scorer = None
        self._observation = None
        self._route_lane_dict = None
        self._drivable_area_map = None
        self._centerline = None
        self._map_api = None
        
        # Sampling
        self._trajectory_sampling = TrajectorySampling(num_poses=80, interval_length=0.1)
        self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        
        logger.info("Initialized Focused HybridPlanner")
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize all components"""
        logger.info("Initializing focused hybrid planner...")
        
        # Store map API
        self._map_api = initialization.map_api
        
        # Initialize scenario detector
        self._scenario_detector = FocusedScenarioDetector(self._map_api)
        
        # Initialize PDM scorer
        self._scorer = PDMScorer(proposal_sampling=self._proposal_sampling)
        
        # Initialize route
        self._initialize_route_info(initialization.route_roadblock_ids)
        
        # Initialize observation
        self._observation = PDMObservation(
            trajectory_sampling=self._trajectory_sampling,
            proposal_sampling=self._proposal_sampling,
            map_radius=50.0
        )
        
        # Initialize planners
        try:
            self.pdm_planner.initialize(initialization)
            logger.info("PDM planner initialized")
        except Exception as e:
            logger.error(f"PDM initialization failed: {e}")
            raise
        
        try:
            self.diffusion_planner.initialize(initialization)
            logger.info("Diffusion planner initialized")
        except Exception as e:
            logger.error(f"Diffusion initialization failed: {e}")
            traceback.print_exc()
            raise
        
        self._iteration = 0
        self._start_time = time.time()
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Main planning function focused on scenario-based switching"""
        self._current_planner_input = current_input
        current_time = time.time() - self._start_time if self._start_time else 0.0
        
        # Extract current state
        ego_state, observation = current_input.history.current_state
        
        # SCENARIO DETECTION (CRITICAL)
        scenario_result = None
        if self._enable_scenario_detection and self._scenario_detector:
            scenario_result = self._scenario_detector.detect_scenario(
                ego_state, 
                observation, 
                current_input.traffic_light_data,
                self._route_lane_dict
            )
            
            # Log scenario every second
            if self._iteration % 10 == 0:
                logger.info(
                    f"SCENARIO: {scenario_result.scenario_type.value} "
                    f"(conf={scenario_result.confidence:.2f}) - "
                    f"{scenario_result.debug_info}"
                )
        
        # Update observation for scoring
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
            self._execute_pdm_planner(current_input, current_time)
        
        if self._iteration == 0 or current_time - self.last_diffusion_time >= self.diffusion_period:
            self._execute_diffusion_planner(current_input, current_time)
        
        # SELECT BASED ON SCENARIO
        selected_trajectory = self._select_trajectory(scenario_result)
        
        self._iteration += 1
        return selected_trajectory
    
    def _execute_pdm_planner(self, current_input, current_time):
        """Execute PDM and score"""
        start = time.time()
        try:
            trajectory = self.pdm_planner.compute_planner_trajectory(current_input)
            computation_time = time.time() - start
            
            # Score trajectory
            score_data = self._score_trajectory(trajectory, 'pdm')
            
            self.current_pdm_trajectory = TrajectoryWithMetadata(
                trajectory=trajectory,
                planner_name='pdm',
                computation_time=computation_time,
                timestamp=current_time,
                score=score_data['total_score'],
                progress=score_data['progress']
            )
            self.last_pdm_time = current_time
            
        except Exception as e:
            logger.error(f"PDM failed: {e}")
            traceback.print_exc()
            raise
    
    def _execute_diffusion_planner(self, current_input, current_time):
        """Execute Diffusion and score"""
        start = time.time()
        try:
            trajectory = self.diffusion_planner.compute_planner_trajectory(current_input)
            computation_time = time.time() - start
            
            # Score trajectory
            score_data = self._score_trajectory(trajectory, 'diffusion')
            
            self.current_diffusion_trajectory = TrajectoryWithMetadata(
                trajectory=trajectory,
                planner_name='diffusion',
                computation_time=computation_time,
                timestamp=current_time,
                score=score_data['total_score'],
                progress=score_data['progress']
            )
            self.last_diffusion_time = current_time
            
        except Exception as e:
            logger.error(f"Diffusion failed: {e}")
            # Continue with PDM on failure
    
    def _select_trajectory(
        self, scenario_result: Optional[ScenarioDetectionResult]
    ) -> AbstractTrajectory:
        """Select trajectory based on scenario"""
        if self.current_pdm_trajectory is None:
            raise RuntimeError("No PDM trajectory!")
        
        if self.current_diffusion_trajectory is None:
            return self.current_pdm_trajectory.trajectory
        
        # Extract scores
        pdm_score = self.current_pdm_trajectory.score or 0.0
        diffusion_score = self.current_diffusion_trajectory.score or 0.0
        pdm_progress = self.current_pdm_trajectory.progress
        
        # Prepare scenario info
        scenario_type = None
        scenario_confidence = None
        scenario_features = None
        
        if scenario_result is not None:
            scenario_type = scenario_result.scenario_type
            scenario_confidence = scenario_result.confidence
            scenario_features = scenario_result.features
        
        # Use focused switcher
        selected_planner, metadata = self._switcher.update_and_select(
            pdm_score=pdm_score,
            diffusion_score=diffusion_score,
            pdm_progress=pdm_progress,
            safety_vetoed=False,
            scenario_type=scenario_type,
            scenario_confidence=scenario_confidence,
            scenario_features=scenario_features
        )
        
        # Log decision every second
        if self._iteration % 10 == 0:
            expected = metadata.get('expected_planner', 'unknown')
            match = "✓" if selected_planner.value == expected else "✗"
            
            logger.info(
                f"DECISION: {selected_planner.value} {match} | "
                f"Expected: {expected} | "
                f"PDM={pdm_score:.3f} Diff={diffusion_score:.3f} | "
                f"P={metadata.get('P', 0):.3f} | "
                f"Switches={metadata.get('switch_count', 0)}"
            )
        
        # Return selected trajectory
        if selected_planner == PlannerType.PDM:
            return self.current_pdm_trajectory.trajectory
        else:
            return self.current_diffusion_trajectory.trajectory
    
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
            
            # Extract metrics
            total_score = float(scores[0])
            progress_normalized = float(self._scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 0])
            
            return {
                'total_score': total_score,
                'progress': progress_normalized,
            }
            
        except Exception as e:
            logger.error(f"Error scoring {planner_name}: {e}")
            return {
                'total_score': 0.0,
                'progress': 0.0,
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
    
    def _initialize_route_info(self, route_roadblock_ids: List[str]) -> None:
        """Initialize route information"""
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
        
        if centerline_discrete_path:
            states_se2 = [StateSE2(p.x, p.y, p.heading) for p in centerline_discrete_path]
            self._centerline = PDMPath(states_se2)
        else:
            # Dummy centerline
            states = [StateSE2(0, 0, 0), StateSE2(10, 0, 0), StateSE2(20, 0, 0)]
            self._centerline = PDMPath(states)
    
    def name(self) -> str:
        return "HybridPlanner_Focused"
    
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks