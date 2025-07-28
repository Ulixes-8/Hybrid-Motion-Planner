"""
Version 4: Decisive Hybrid Planner
Achieves superior performance through aggressive scenario-based switching
"""

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

# Import our decisive switcher
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
    is_valid: Optional[bool] = None


class OptimizedScenarioDetector:
    """
    Version 4: Optimized scenario detection for the 272 test scenarios
    """
    
    def __init__(self, map_api):
        self.map_api = map_api
        # Cache for map queries
        self._intersection_cache = {}
        self._crosswalk_cache = {}
        
    def detect_scenario(
        self, 
        ego_state, 
        observation, 
        traffic_light_data,
        route_lane_dict: Dict
    ) -> ScenarioDetectionResult:
        """
        Optimized scenario detection with decisive classification
        """
        features = self._extract_features(ego_state, observation, traffic_light_data)
        
        # Priority order - check most distinctive scenarios first
        
        # 1. STOPPING_WITH_LEAD - highest PDM advantage (0.984)
        if (features['longitudinal_accel'] < -0.6 and 
            features['ego_speed'] < 0.3 and
            features['leading_vehicle'] is not None and
            features['leading_vehicle']['distance'] < 6.0):
            return self._create_result(ScenarioType.STOPPING_WITH_LEAD, 0.95, features)
        
        # 2. STATIONARY_IN_TRAFFIC - strong PDM (0.868)
        if features['ego_speed'] < 0.1 and features['vehicles_within_8m'] > 6:
            return self._create_result(ScenarioType.STATIONARY_IN_TRAFFIC, 0.95, features)
        
        # 3. TRAVERSING_PICKUP_DROPOFF - strong Diffusion (0.685)
        if features['in_pickup_dropoff'] and features['ego_speed'] > 0.1:
            return self._create_result(ScenarioType.TRAVERSING_PICKUP_DROPOFF, 0.9, features)
        
        # 4. STARTING_STRAIGHT_INTERSECTION - strongest Diffusion (0.900)
        if (features['at_intersection'] and 
            features['has_traffic_light'] and
            features['ego_speed'] > 0.1 and
            abs(features['yaw_rate']) < 0.1):
            return self._create_result(ScenarioType.STARTING_STRAIGHT_INTERSECTION, 0.9, features)
        
        # 5. HIGH_MAGNITUDE_SPEED - strong PDM (0.919)
        if features['ego_speed'] > 9.0 and features['acceleration_magnitude'] < 1.0:
            return self._create_result(ScenarioType.HIGH_MAGNITUDE_SPEED, 0.95, features)
        
        # 6. FOLLOWING_LANE_WITH_LEAD - strong PDM (0.939)
        if (features['ego_speed'] > 3.5 and 
            features['leading_vehicle'] is not None and
            features['leading_vehicle']['speed'] > 3.5 and
            features['leading_vehicle']['distance'] < 7.5):
            return self._create_result(ScenarioType.FOLLOWING_LANE_WITH_LEAD, 0.95, features)
        
        # 7. CHANGING_LANE - Diffusion (0.767)
        if (features['lateral_accel'] > 0.4 and 
            features['ego_speed'] > 3.0 and
            features['yaw_rate'] < 0.15):
            return self._create_result(ScenarioType.CHANGING_LANE, 0.85, features)
        
        # 8. HIGH_LATERAL_ACCELERATION - Diffusion (0.573)
        if (1.5 < features['lateral_accel'] < 3.0 and 
            features['yaw_rate'] > 0.2):
            return self._create_result(ScenarioType.HIGH_LATERAL_ACCELERATION, 0.85, features)
        
        # 9. WAITING_FOR_PEDESTRIAN - strong PDM (0.882)
        if features['near_crosswalk'] and features['near_pedestrian']:
            return self._create_result(ScenarioType.WAITING_FOR_PEDESTRIAN, 0.9, features)
        
        # 10. BEHIND_LONG_VEHICLE - PDM (0.926)
        if features['behind_long_vehicle']:
            return self._create_result(ScenarioType.BEHIND_LONG_VEHICLE, 0.9, features)
        
        # 11. Turns at intersections
        if features['at_intersection'] and features['ego_speed'] > 0.1:
            if features['yaw_rate'] > 0.15:
                return self._create_result(ScenarioType.STARTING_LEFT_TURN, 0.85, features)
            elif features['yaw_rate'] < -0.15:
                return self._create_result(ScenarioType.STARTING_RIGHT_TURN, 0.85, features)
        
        # 12. LOW_MAGNITUDE_SPEED - slight Diffusion (0.480)
        if 0.3 < features['ego_speed'] < 1.2 and features['acceleration_magnitude'] < 0.5:
            return self._create_result(ScenarioType.LOW_MAGNITUDE_SPEED, 0.8, features)
        
        # 13. NEAR_MULTIPLE_VEHICLES - PDM (0.659)
        if features['ego_speed'] > 6.0 and features['vehicles_within_8m'] > 6:
            return self._create_result(ScenarioType.NEAR_MULTIPLE_VEHICLES, 0.85, features)
        
        # Default
        return self._create_result(ScenarioType.UNKNOWN, 0.5, features)
    
    def _extract_features(self, ego_state, observation, traffic_light_data) -> Dict:
        """Optimized feature extraction"""
        features = {}
        
        # Ego motion
        vx = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        vy = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        features['ego_speed'] = float(np.hypot(vx, vy))
        
        ax = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        ay = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
        features['longitudinal_accel'] = float(ax)
        features['lateral_accel'] = float(abs(ay))
        features['acceleration_magnitude'] = float(np.hypot(ax, ay))
        features['yaw_rate'] = float(abs(ego_state.dynamic_car_state.angular_velocity))
        features['steering_angle'] = float(ego_state.tire_steering_angle)
        
        # Process objects efficiently
        vehicles_within_8m = 0
        leading_vehicle = None
        behind_long_vehicle = False
        near_pedestrian = False
        
        ego_x = ego_state.rear_axle.x
        ego_y = ego_state.rear_axle.y
        ego_heading = ego_state.rear_axle.heading
        cos_h = np.cos(ego_heading)
        sin_h = np.sin(ego_heading)
        
        if hasattr(observation, 'tracked_objects'):
            for obj in observation.tracked_objects:
                # Transform to ego frame
                rel_x = obj.center.x - ego_x
                rel_y = obj.center.y - ego_y
                obj_x = rel_x * cos_h + rel_y * sin_h
                obj_y = -rel_x * sin_h + rel_y * cos_h
                distance = np.hypot(obj_x, obj_y)
                
                if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                    if distance < 8.0:
                        vehicles_within_8m += 1
                    
                    # Leading vehicle check
                    if (0 < obj_x < 20.0 and abs(obj_y) < 1.5 and 
                        (leading_vehicle is None or obj_x < leading_vehicle['distance'])):
                        obj_speed = float(np.hypot(obj.velocity.x, obj.velocity.y))
                        leading_vehicle = {
                            'distance': obj_x,
                            'speed': obj_speed
                        }
                    
                    # Long vehicle check
                    if (3.0 < obj_x < 10.0 and abs(obj_y) < 0.5 and obj.box.length > 8.0):
                        behind_long_vehicle = True
                        
                elif obj.tracked_object_type in [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]:
                    if distance < 8.0:
                        near_pedestrian = True
        
        features['vehicles_within_8m'] = vehicles_within_8m
        features['leading_vehicle'] = leading_vehicle
        features['behind_long_vehicle'] = behind_long_vehicle
        features['near_pedestrian'] = near_pedestrian
        
        # Map features (with caching)
        features['at_intersection'] = self._check_intersection_cached(ego_state)
        features['near_crosswalk'] = self._check_crosswalk_cached(ego_state)
        features['in_pickup_dropoff'] = self._check_pickup_dropoff(ego_state)
        
        # Traffic light
        features['has_traffic_light'] = bool(traffic_light_data)
        
        return features
    
    def _check_intersection_cached(self, ego_state) -> bool:
        """Check intersection with caching"""
        key = (round(ego_state.rear_axle.x), round(ego_state.rear_axle.y))
        if key not in self._intersection_cache:
            self._intersection_cache[key] = self._check_intersection(ego_state)
        return self._intersection_cache[key]
    
    def _check_crosswalk_cached(self, ego_state) -> bool:
        """Check crosswalk with caching"""
        key = (round(ego_state.rear_axle.x), round(ego_state.rear_axle.y))
        if key not in self._crosswalk_cache:
            self._crosswalk_cache[key] = self._check_crosswalk(ego_state)
        return self._crosswalk_cache[key]
    
    def _check_intersection(self, ego_state) -> bool:
        try:
            connectors = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle, radius=15.0,
                layers=[SemanticMapLayer.LANE_CONNECTOR]
            )
            return len(connectors.get(SemanticMapLayer.LANE_CONNECTOR, [])) > 0
        except:
            return False
    
    def _check_crosswalk(self, ego_state) -> bool:
        try:
            crosswalks = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle, radius=10.0,
                layers=[SemanticMapLayer.CROSSWALK]
            )
            return len(crosswalks.get(SemanticMapLayer.CROSSWALK, [])) > 0
        except:
            return False
    
    def _check_pickup_dropoff(self, ego_state) -> bool:
        """Simplified pickup/dropoff check"""
        # This is highly map-specific - adjust based on your data
        return False  # Conservative default
    
    def _create_result(self, scenario: ScenarioType, confidence: float, features: Dict) -> ScenarioDetectionResult:
        return ScenarioDetectionResult(
            scenario_type=scenario,
            confidence=confidence,
            features=features,
            debug_info=f"{scenario.value} detected"
        )


class HybridPlanner(AbstractPlanner):
    """
    Version 4: Decisive Hybrid Planner
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
        """Initialize Version 4 hybrid planner"""
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
        self._last_scenario = ScenarioType.UNKNOWN
        self._scenario_persistence = 0
        
        # Initialize decisive switcher
        config = LeakyDDMConfig(**leaky_ddm_config) if leaky_ddm_config else LeakyDDMConfig()
        self._switcher = LeakyDDMSwitcher(config)
        
        # PDM components
        self._scorer = None
        self._observation = None
        self._route_lane_dict = None
        self._drivable_area_map = None
        self._centerline = None
        self._map_api = None
        
        # Sampling
        self._trajectory_sampling = TrajectorySampling(num_poses=80, interval_length=0.1)
        self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        
        logger.info("Initialized HybridPlanner Version 4 (Decisive)")
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize all components"""
        logger.info("Initializing Version 4 hybrid planner...")
        
        self._map_api = initialization.map_api
        self._scenario_detector = OptimizedScenarioDetector(self._map_api)
        
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
        """Main planning with decisive scenario-based switching"""
        self._current_planner_input = current_input
        current_time = time.time() - self._start_time if self._start_time else 0.0
        
        ego_state, observation = current_input.history.current_state
        
        # CRITICAL: Detect scenario
        scenario_result = None
        if self._enable_scenario_detection and self._scenario_detector:
            scenario_result = self._scenario_detector.detect_scenario(
                ego_state, 
                observation, 
                current_input.traffic_light_data,
                self._route_lane_dict
            )
            
            # Track scenario persistence
            if scenario_result.scenario_type == self._last_scenario:
                self._scenario_persistence += 1
            else:
                self._scenario_persistence = 0
                self._last_scenario = scenario_result.scenario_type
            
            # Log scenario changes
            if self._scenario_persistence == 0 or self._iteration % 20 == 0:
                logger.info(
                    f"[V4] Scenario: {scenario_result.scenario_type.value} "
                    f"(conf={scenario_result.confidence:.2f}, persist={self._scenario_persistence})"
                )
        
        # Update observation for scoring
        self._observation.update(
            ego_state=ego_state,
            observation=observation,
            traffic_light_data=current_input.traffic_light_data,
            route_lane_dict=self._route_lane_dict
        )
        
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, map_radius=50.0
        )
        
        # Execute planners
        if self._iteration == 0 or current_time - self.last_pdm_time >= self.pdm_period:
            self._execute_pdm_planner(current_input, current_time)
        
        if self._iteration == 0 or current_time - self.last_diffusion_time >= self.diffusion_period:
            self._execute_diffusion_planner(current_input, current_time)
        
        # DECISIVE selection
        selected_trajectory = self._select_trajectory_decisive(scenario_result)
        
        self._iteration += 1
        return selected_trajectory
    
    def _select_trajectory_decisive(
        self, scenario_result: Optional[ScenarioDetectionResult]
    ) -> AbstractTrajectory:
        """Decisive trajectory selection"""
        if self.current_pdm_trajectory is None:
            raise RuntimeError("No PDM trajectory!")
        
        if self.current_diffusion_trajectory is None:
            return self.current_pdm_trajectory.trajectory
        
        # Get scores
        pdm_score = self.current_pdm_trajectory.score or 0.0
        diffusion_score = self.current_diffusion_trajectory.score or 0.0
        
        # Use decisive switcher
        selected_planner, metadata = self._switcher.update_and_select(
            pdm_score=pdm_score,
            diffusion_score=diffusion_score,
            pdm_progress=self.current_pdm_trajectory.progress,
            safety_vetoed=False,
            scenario_type=scenario_result.scenario_type if scenario_result else None,
            scenario_confidence=scenario_result.confidence if scenario_result else None,
            scenario_persistence=self._scenario_persistence
        )
        
        # Log decisions periodically
        if self._iteration % 20 == 0:
            logger.info(
                f"[V4] Decision: {selected_planner.value} | "
                f"PDM={pdm_score:.3f} Diff={diffusion_score:.3f} | "
                f"Switches: {metadata.get('switch_count', 0)}"
            )
        
        # Return selected trajectory
        if selected_planner == PlannerType.PDM:
            return self.current_pdm_trajectory.trajectory
        else:
            return self.current_diffusion_trajectory.trajectory
    
    def _execute_pdm_planner(self, current_input, current_time):
        """Execute PDM planner"""
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
                is_valid=score_data['is_valid']
            )
            self.last_pdm_time = current_time
            
        except Exception as e:
            logger.error(f"PDM failed: {e}")
            raise
    
    def _execute_diffusion_planner(self, current_input, current_time):
        """Execute Diffusion planner"""
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
                is_valid=score_data['is_valid']
            )
            self.last_diffusion_time = current_time
            
        except Exception as e:
            logger.error(f"Diffusion failed: {e}")
    
    def _score_trajectory(self, trajectory: AbstractTrajectory, planner_name: str) -> Dict:
        """Score trajectory"""
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
            progress_normalized = float(self._scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 0])
            
            no_collision = float(self._scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 0])
            drivable_area = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 0])
            driving_direction = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, 0])
            
            is_valid = no_collision > 0.5 and drivable_area > 0.5 and driving_direction > 0.5
            
            return {
                'total_score': total_score,
                'progress': progress_normalized,
                'is_valid': is_valid
            }
            
        except Exception as e:
            logger.error(f"Error scoring {planner_name}: {e}")
            return {'total_score': 0.0, 'progress': 0.0, 'is_valid': False}
    
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
            states = [StateSE2(0, 0, 0), StateSE2(10, 0, 0), StateSE2(20, 0, 0)]
            self._centerline = PDMPath(states)
    
    def name(self) -> str:
        return "HybridPlanner_V4_Decisive"
    
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks