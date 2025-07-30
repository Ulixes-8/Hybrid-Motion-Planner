"""
Hybrid Planner Version 6 - Enhanced Scoring System
Built on Version 5 Conservative Oracle with added novel metrics.
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

# Import our enhanced components
from chimera.planner.leaky_ddm_switcher import (
    EnhancedSwitcher, EnhancedSwitcherConfig, PlannerType, ScenarioType
)
from chimera.planner.enhanced_scoring_metrics import EnhancedScoringMetrics

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
    is_valid: bool = True
    safety_score: Optional[float] = None
    progress_score: Optional[float] = None
    enhanced_metrics: Optional[Dict[str, float]] = None


class RobustScenarioDetector:
    """
    Robust scenario detection with validation and fallback.
    """
    
    def __init__(self, map_api):
        self.map_api = map_api
        self.last_scenario = ScenarioType.UNKNOWN
        self.scenario_persistence = 0
        
    def detect_scenario(
        self, 
        ego_state, 
        observation, 
        traffic_light_data,
        route_lane_dict: Dict
    ) -> ScenarioDetectionResult:
        """
        Detect scenario with validation and confidence estimation.
        """
        try:
            # Extract features safely
            features = self._extract_features_safe(ego_state, observation, traffic_light_data)
            
            # Detect scenario with validation
            result = self._detect_scenario_validated(features)
            
            # Track persistence for confidence
            if result.scenario_type == self.last_scenario:
                self.scenario_persistence += 1
                # Boost confidence for persistent scenarios
                result.confidence = min(1.0, result.confidence + 0.05 * self.scenario_persistence)
            else:
                self.scenario_persistence = 0
                self.last_scenario = result.scenario_type
            
            return result
            
        except Exception as e:
            logger.error(f"Scenario detection error: {e}")
            return ScenarioDetectionResult(
                scenario_type=ScenarioType.UNKNOWN,
                confidence=0.0,
                features={},
                debug_info=f"Detection error: {str(e)}"
            )
    
    def _extract_features_safe(self, ego_state, observation, traffic_light_data) -> Dict:
        """Extract features with error handling"""
        features = {
            # Default values
            'ego_speed': 0.0,
            'longitudinal_accel': 0.0,
            'lateral_accel': 0.0,
            'acceleration_magnitude': 0.0,
            'yaw_rate': 0.0,
            'steering_angle': 0.0,
            'has_lead_vehicle': False,
            'lead_distance': float('inf'),
            'lead_speed': 0.0,
            'lead_same_lane': False,
            'vehicles_within_8m': 0,
            'moving_vehicles_within_8m': 0,
            'behind_long_vehicle': False,
            'has_pedestrian_nearby': False,
            'min_pedestrian_tti': float('inf'),
            'at_intersection': False,
            'near_crosswalk': False,
            'in_pickup_dropoff': False,
            'has_traffic_light': False,
            'is_changing_lane': False,
            'is_turning_left': False,
            'is_turning_right': False,
        }
        
        try:
            # Ego motion
            vx = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
            vy = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
            features['ego_speed'] = float(np.hypot(vx, vy))
            
            features['longitudinal_accel'] = float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x)
            features['lateral_accel'] = float(abs(ego_state.dynamic_car_state.rear_axle_acceleration_2d.y))
            features['acceleration_magnitude'] = float(np.hypot(
                ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
                ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
            ))
            features['yaw_rate'] = float(abs(ego_state.dynamic_car_state.angular_velocity))
            features['steering_angle'] = float(ego_state.tire_steering_angle)
            
        except Exception as e:
            logger.warning(f"Error extracting ego features: {e}")
        
        # Process tracked objects
        try:
            if hasattr(observation, 'tracked_objects'):
                self._process_tracked_objects(ego_state, observation, features)
        except Exception as e:
            logger.warning(f"Error processing tracked objects: {e}")
        
        # Map features
        try:
            features['at_intersection'] = self._check_intersection(ego_state)
            features['near_crosswalk'] = self._check_crosswalk(ego_state)
            features['in_pickup_dropoff'] = self._check_pickup_dropoff(ego_state)
        except Exception as e:
            logger.warning(f"Error checking map features: {e}")
        
        # Traffic light
        features['has_traffic_light'] = bool(traffic_light_data) if traffic_light_data else False
        
        # Derived features
        features['is_changing_lane'] = (
            features['lateral_accel'] > 0.5 and 
            features['ego_speed'] > 3.0 and
            features['yaw_rate'] < 0.15 and
            not features['at_intersection']
        )
        
        features['is_turning_left'] = features['yaw_rate'] > 0.15 or features['steering_angle'] > 0.1
        features['is_turning_right'] = features['yaw_rate'] < -0.15 or features['steering_angle'] < -0.1
        
        return features
    
    def _process_tracked_objects(self, ego_state, observation, features):
        """Process tracked objects safely"""
        for obj in observation.tracked_objects:
            try:
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
                    
                    if distance < 8.0:
                        features['vehicles_within_8m'] += 1
                        if obj_speed > 6.0:
                            features['moving_vehicles_within_8m'] += 1
                    
                    # Leading vehicle
                    if ego_x > 0 and ego_x < 20.0 and abs(ego_y) < 1.0:
                        if ego_x < features['lead_distance']:
                            features['has_lead_vehicle'] = True
                            features['lead_distance'] = ego_x
                            features['lead_speed'] = obj_speed
                            features['lead_same_lane'] = abs(ego_y) < 0.5
                    
                    # Long vehicle check
                    if (obj.box.length > 8.0 and 
                        3.0 < ego_x < 10.0 and 
                        abs(ego_y) < 0.5):
                        features['behind_long_vehicle'] = True
                
                elif obj.tracked_object_type in [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]:
                    if distance < 8.0:
                        features['has_pedestrian_nearby'] = True
                        if features['ego_speed'] > 0.1:
                            tti = distance / features['ego_speed']
                            features['min_pedestrian_tti'] = min(features['min_pedestrian_tti'], tti)
                            
            except Exception as e:
                logger.debug(f"Error processing object: {e}")
                continue
    
    def _detect_scenario_validated(self, features: Dict) -> ScenarioDetectionResult:
        """Detect scenario with confidence based on feature quality"""
        
        # STOPPING_WITH_LEAD
        if (features['longitudinal_accel'] < -0.6 and 
            features['ego_speed'] < 0.3 and
            features['has_lead_vehicle'] and
            features['lead_distance'] < 6.0):
            return self._result(ScenarioType.STOPPING_WITH_LEAD, 0.9, features, "Stopping with lead")
        
        # STATIONARY_IN_TRAFFIC
        if features['ego_speed'] < 0.1 and features['vehicles_within_8m'] > 6:
            return self._result(ScenarioType.STATIONARY_IN_TRAFFIC, 0.9, features, "Stationary in traffic")
        
        # WAITING_FOR_PEDESTRIAN
        if (features['ego_speed'] > 0.1 and 
            features['near_crosswalk'] and 
            features['has_pedestrian_nearby'] and
            features['min_pedestrian_tti'] < 1.5):
            return self._result(ScenarioType.WAITING_FOR_PEDESTRIAN, 0.85, features, "Waiting for pedestrian")
        
        # TRAVERSING_PICKUP_DROPOFF
        if features['in_pickup_dropoff'] and features['ego_speed'] > 0.1:
            return self._result(ScenarioType.TRAVERSING_PICKUP_DROPOFF, 0.85, features, "Pickup/dropoff area")
        
        # BEHIND_LONG_VEHICLE
        if features['behind_long_vehicle']:
            return self._result(ScenarioType.BEHIND_LONG_VEHICLE, 0.85, features, "Behind long vehicle")
        
        # FOLLOWING_LANE_WITH_LEAD
        if (features['ego_speed'] > 3.5 and 
            features['has_lead_vehicle'] and
            features['lead_speed'] > 3.5 and
            features['lead_distance'] < 7.5 and
            features['lead_same_lane']):
            return self._result(ScenarioType.FOLLOWING_LANE_WITH_LEAD, 0.9, features, "Following lead")
        
        # HIGH_MAGNITUDE_SPEED
        if features['ego_speed'] > 9.0 and features['acceleration_magnitude'] < 1.0:
            return self._result(ScenarioType.HIGH_MAGNITUDE_SPEED, 0.9, features, "High speed cruise")
        
        # LOW_MAGNITUDE_SPEED
        if (0.3 < features['ego_speed'] < 1.2 and 
            features['acceleration_magnitude'] < 1.0 and
            not features['in_pickup_dropoff']):
            return self._result(ScenarioType.LOW_MAGNITUDE_SPEED, 0.8, features, "Low speed maneuver")
        
        # HIGH_LATERAL_ACCELERATION
        if (1.5 < features['lateral_accel'] < 3.0 and 
            features['yaw_rate'] > 0.2 and
            not features['at_intersection']):
            return self._result(ScenarioType.HIGH_LATERAL_ACCELERATION, 0.8, features, "High lateral accel")
        
        # CHANGING_LANE
        if features['is_changing_lane']:
            return self._result(ScenarioType.CHANGING_LANE, 0.8, features, "Changing lane")
        
        # NEAR_MULTIPLE_VEHICLES
        if features['ego_speed'] > 6.0 and features['moving_vehicles_within_8m'] > 6:
            return self._result(ScenarioType.NEAR_MULTIPLE_VEHICLES, 0.8, features, "Near multiple vehicles")
        
        # INTERSECTION SCENARIOS
        if features['at_intersection'] and features['ego_speed'] > 0.1:
            if features['is_turning_left']:
                return self._result(ScenarioType.STARTING_LEFT_TURN, 0.8, features, "Starting left turn")
            elif features['is_turning_right']:
                return self._result(ScenarioType.STARTING_RIGHT_TURN, 0.8, features, "Starting right turn")
            elif features['has_traffic_light']:
                return self._result(ScenarioType.STARTING_STRAIGHT_INTERSECTION, 0.8, features, "Straight intersection")
        
        # Default - lower confidence
        return self._result(ScenarioType.UNKNOWN, 0.3, features, "No specific scenario")
    
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
                for attr in ['lane_type', 'lane_type_fid', 'id']:
                    if hasattr(lane, attr):
                        value = str(getattr(lane, attr)).lower()
                        if any(kw in value for kw in ['pickup', 'dropoff', 'loading', 'passenger']):
                            return True
            return False
        except:
            return False
    
    def _result(self, scenario_type: ScenarioType, confidence: float, 
                features: Dict, debug_info: str) -> ScenarioDetectionResult:
        """Create result"""
        return ScenarioDetectionResult(
            scenario_type=scenario_type,
            confidence=confidence,
            features=features,
            debug_info=debug_info
        )


class HybridPlanner(AbstractPlanner):
    """
    Version 6 Hybrid Planner - V5 Conservative Oracle + Enhanced Scoring
    Adds novel metrics to V5's robust conservative switching.
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
        """Initialize enhanced hybrid planner"""
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
        self.last_selected_trajectory: Optional[AbstractTrajectory] = None
        self.last_selected_planner: Optional[PlannerType] = None
        
        # State
        self._iteration = 0
        self._start_time = None
        self._enable_scenario_detection = enable_scenario_detection
        self._scenario_detector = None
        
        # Initialize enhanced switcher (V5 conservative + novel metrics)
        config = EnhancedSwitcherConfig(**leaky_ddm_config) if leaky_ddm_config else EnhancedSwitcherConfig()
        self._switcher = EnhancedSwitcher(config)
        
        # PDM components for scoring
        self._scorer = None
        self._observation = None
        self._route_lane_dict = None
        self._drivable_area_map = None
        self._centerline = None
        self._map_api = None
        
        # Enhanced scoring
        self._enhanced_scorer = EnhancedScoringMetrics()
        
        # Sampling
        self._trajectory_sampling = TrajectorySampling(num_poses=80, interval_length=0.1)
        self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        
        logger.info("Initialized HybridPlanner v6 (V5 Conservative + Enhanced Metrics)")
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize all components"""
        logger.info("Initializing HybridPlanner v6 (Conservative + Enhanced Metrics)...")
        
        # Store map API
        self._map_api = initialization.map_api
        
        # Initialize robust scenario detector
        self._scenario_detector = RobustScenarioDetector(self._map_api)
        
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
        """Main planning function with enhanced scoring"""
        self._current_planner_input = current_input
        current_time = time.time() - self._start_time if self._start_time else 0.0
        
        # Extract current state
        ego_state, observation = current_input.history.current_state
        
        # Robust scenario detection
        scenario_result = None
        if self._enable_scenario_detection and self._scenario_detector:
            scenario_result = self._scenario_detector.detect_scenario(
                ego_state, 
                observation, 
                current_input.traffic_light_data,
                self._route_lane_dict
            )
            
            # Log significant scenarios
            if (scenario_result.confidence > 0.7 or 
                self._iteration % 20 == 0):
                logger.info(
                    f"Scenario: {scenario_result.scenario_type.value} "
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
        # ALWAYS execute PDM on first iteration to ensure we have a trajectory
        if self._iteration == 0:
            self._execute_pdm_planner(current_input, current_time, scenario_result)
            self._execute_diffusion_planner(current_input, current_time, scenario_result)
        else:
            # Normal frequency-based execution
            if current_time - self.last_pdm_time >= self.pdm_period:
                self._execute_pdm_planner(current_input, current_time, scenario_result)
            
            if current_time - self.last_diffusion_time >= self.diffusion_period:
                self._execute_diffusion_planner(current_input, current_time, scenario_result)
        
        # Enhanced selection with novel metrics
        selected_trajectory = self._select_trajectory_enhanced(scenario_result)
        
        self._iteration += 1
        return selected_trajectory
    
    def _execute_pdm_planner(self, current_input, current_time, scenario_result):
        """Execute PDM planner with enhanced scoring"""
        start = time.time()
        try:
            trajectory = self.pdm_planner.compute_planner_trajectory(current_input)
            computation_time = time.time() - start
            
            # Score trajectory with enhanced metrics
            score_data = self._score_trajectory_enhanced(
                trajectory, 'pdm', scenario_result
            )
            
            self.current_pdm_trajectory = TrajectoryWithMetadata(
                trajectory=trajectory,
                planner_name='pdm',
                computation_time=computation_time,
                timestamp=current_time,
                score=score_data['total_score'],
                is_valid=True,
                safety_score=score_data['safety_score'],
                progress_score=score_data.get('progress', 0.0),
                enhanced_metrics=score_data.get('enhanced_metrics', {})
            )
            self.last_pdm_time = current_time
            
        except Exception as e:
            logger.error(f"PDM execution failed: {e}")
    
    def _execute_diffusion_planner(self, current_input, current_time, scenario_result):
        """Execute Diffusion planner with enhanced scoring"""
        start = time.time()
        try:
            trajectory = self.diffusion_planner.compute_planner_trajectory(current_input)
            computation_time = time.time() - start
            
            # Score trajectory with enhanced metrics
            score_data = self._score_trajectory_enhanced(
                trajectory, 'diffusion', scenario_result
            )
            
            self.current_diffusion_trajectory = TrajectoryWithMetadata(
                trajectory=trajectory,
                planner_name='diffusion',
                computation_time=computation_time,
                timestamp=current_time,
                score=score_data['total_score'],
                is_valid=True,
                safety_score=score_data['safety_score'],
                progress_score=score_data.get('progress', 0.0),
                enhanced_metrics=score_data.get('enhanced_metrics', {})
            )
            self.last_diffusion_time = current_time
            
        except Exception as e:
            logger.error(f"Diffusion execution failed: {e}")
    
    def _score_trajectory_enhanced(
        self, 
        trajectory: AbstractTrajectory, 
        planner_name: str,
        scenario_result: Optional[ScenarioDetectionResult]
    ) -> Dict:
        """Score trajectory with standard and enhanced metrics"""
        # First get standard PDM scores
        score_data = self._score_trajectory_safe(trajectory, planner_name)
        
        try:
            # Calculate enhanced metrics
            states = self._trajectory_to_states(trajectory)
            ego_state = self._current_planner_input.history.current_state[0]
            
            scenario_type = scenario_result.scenario_type.value if scenario_result else None
            
            enhanced_metrics = self._enhanced_scorer.calculate_all_metrics(
                states=states,
                initial_ego_state=ego_state,
                observation=self._observation,
                centerline=self._centerline,
                route_lane_dict=self._route_lane_dict,
                drivable_area_map=self._drivable_area_map,
                map_api=self._map_api,
                scenario_type=scenario_type
            )
            
            score_data['enhanced_metrics'] = enhanced_metrics
            
            # Log enhanced metrics for analysis
            if self._iteration % 20 == 0:
                logger.debug(
                    f"{planner_name} enhanced metrics: "
                    f"efficiency={enhanced_metrics['trajectory_efficiency']:.3f}, "
                    f"social={enhanced_metrics['social_compliance']:.3f}, "
                    f"spatial={enhanced_metrics['spatial_utilization']:.3f}"
                )
            
        except Exception as e:
            logger.warning(f"Error calculating enhanced metrics for {planner_name}: {e}")
            score_data['enhanced_metrics'] = {
                'trajectory_efficiency': 0.5,
                'social_compliance': 0.5,
                'spatial_utilization': 0.5,
                'trajectory_consistency': 0.5,
                'maneuver_preparedness': 0.5,
                'weighted_total': 0.5
            }
        
        return score_data
    
    def _select_trajectory_enhanced(
        self, scenario_result: Optional[ScenarioDetectionResult]
    ) -> AbstractTrajectory:
        """Select trajectory using enhanced metrics"""
        
        # Always ensure we have at least PDM trajectory
        if self.current_pdm_trajectory is None:
            logger.error("No PDM trajectory available!")
            if self.last_selected_trajectory is not None:
                logger.warning("Using last selected trajectory as emergency fallback")
                return self.last_selected_trajectory
            else:
                raise RuntimeError("No trajectories available at all!")
        
        # If no diffusion trajectory, use PDM
        if self.current_diffusion_trajectory is None:
            logger.debug("No diffusion trajectory, using PDM")
            self.last_selected_trajectory = self.current_pdm_trajectory.trajectory
            self.last_selected_planner = PlannerType.PDM
            return self.current_pdm_trajectory.trajectory
        
        # Extract all scoring data
        pdm_score = self.current_pdm_trajectory.score or 0.0
        diffusion_score = self.current_diffusion_trajectory.score or 0.0
        pdm_safety = self.current_pdm_trajectory.safety_score or 0.0
        diffusion_safety = self.current_diffusion_trajectory.safety_score or 0.0
        
        pdm_enhanced = self.current_pdm_trajectory.enhanced_metrics or {}
        diffusion_enhanced = self.current_diffusion_trajectory.enhanced_metrics or {}
        
        # Use enhanced switcher
        selected_planner, metadata = self._switcher.update_and_select(
            pdm_score=pdm_score,
            diffusion_score=diffusion_score,
            pdm_safety=pdm_safety,
            diffusion_safety=diffusion_safety,
            pdm_enhanced_metrics=pdm_enhanced,
            diffusion_enhanced_metrics=diffusion_enhanced,
            scenario_result=scenario_result,
            last_planner=self.last_selected_planner
        )
        
        # Log decisions for significant scenarios
        if (scenario_result and 
            scenario_result.confidence > 0.7 and
            self._iteration % 10 == 0):
            
            logger.info(
                f"Selection: {selected_planner.value} | "
                f"Scores: PDM={pdm_score:.3f} Diff={diffusion_score:.3f} | "
                f"Enhanced: PDM={pdm_enhanced.get('weighted_total', 0):.3f} "
                f"Diff={diffusion_enhanced.get('weighted_total', 0):.3f} | "
                f"Expected: {metadata.get('expected_planner', 'unknown')}"
            )
        
        # Select and cache trajectory
        if selected_planner == PlannerType.PDM:
            selected_trajectory = self.current_pdm_trajectory.trajectory
        else:
            selected_trajectory = self.current_diffusion_trajectory.trajectory
        
        self.last_selected_trajectory = selected_trajectory
        self.last_selected_planner = selected_planner
        
        return selected_trajectory
    
    def _score_trajectory_safe(self, trajectory: AbstractTrajectory, planner_name: str) -> Dict:
        """Score trajectory safely (standard PDM scoring)"""
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
            
            # Safety metrics
            no_collision = float(self._scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 0])
            drivable_area = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 0])
            driving_direction = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, 0])
            
            # Safety score (multiplicative metrics)
            safety_score = no_collision * drivable_area * driving_direction
            
            # Progress
            progress = float(self._scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 0])
            
            return {
                'total_score': total_score,
                'is_valid': True,
                'safety_score': safety_score,
                'progress': progress,
                'no_collision': no_collision,
                'drivable_area': drivable_area,
                'driving_direction': driving_direction,
            }
            
        except Exception as e:
            logger.error(f"Error scoring {planner_name}: {e}")
            return {
                'total_score': 0.0,
                'is_valid': True,
                'safety_score': 0.0,
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
        return "HybridPlanner_v6_Conservative_Enhanced"
    
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks