"""
Hybrid Planner Version 4: Scenario-First Decision Making
Prioritizes empirical scenario performance with improved detection and switching.
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

# Import our V4 switcher
from chimera.planner.leaky_ddm_switcher import (
    LeakyDDMSwitcherV4, LeakyDDMConfigV4, PlannerType, ScenarioType
)

logger = logging.getLogger(__name__)


@dataclass
class ScenarioDetectionResult:
    """Enhanced scenario detection result with reliability tracking"""
    scenario_type: ScenarioType
    confidence: float
    features: Dict[str, float]
    is_transition: bool = False  # True if scenario just changed
    persistence: int = 0  # How many cycles in this scenario
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
    is_valid: bool = True
    safety_metrics: Optional[Dict] = None


class ImprovedScenarioDetector:
    """
    Version 4 Scenario Detector with improved accuracy and robustness.
    """
    
    def __init__(self, map_api):
        self.map_api = map_api
        self.last_scenario = ScenarioType.UNKNOWN
        self.scenario_persistence = 0
        self.scenario_history = []  # Track last N scenarios for stability
        self.history_size = 5
        
    def detect_scenario(
        self, 
        ego_state, 
        observation, 
        traffic_light_data,
        route_lane_dict: Dict
    ) -> ScenarioDetectionResult:
        """
        Improved scenario detection with exact nuPlan matching.
        """
        # Extract comprehensive features
        features = self._extract_features(ego_state, observation, traffic_light_data)
        
        # Detect scenario with priority order (safety-critical first)
        scenario_type, confidence, debug_info = self._classify_scenario(features)
        
        # Track persistence and transitions
        is_transition = (scenario_type != self.last_scenario)
        if scenario_type == self.last_scenario:
            self.scenario_persistence += 1
        else:
            self.scenario_persistence = 1
            self.last_scenario = scenario_type
        
        # Add to history for stability check
        self.scenario_history.append(scenario_type)
        if len(self.scenario_history) > self.history_size:
            self.scenario_history.pop(0)
        
        # Adjust confidence based on stability
        if self.scenario_persistence > 3:
            confidence = min(0.99, confidence * 1.1)  # Boost stable scenarios
        elif self._is_flickering():
            confidence *= 0.7  # Reduce confidence if flickering
        
        return ScenarioDetectionResult(
            scenario_type=scenario_type,
            confidence=confidence,
            features=features,
            is_transition=is_transition,
            persistence=self.scenario_persistence,
            debug_info=debug_info
        )
    
    def _extract_features(self, ego_state, observation, traffic_light_data) -> Dict:
        """Extract all features needed for accurate scenario detection"""
        features = {}
        
        # Basic ego motion
        vx = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        vy = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        features['ego_speed'] = float(np.hypot(vx, vy))
        features['ego_velocity_x'] = float(vx)
        features['ego_velocity_y'] = float(vy)
        
        # Accelerations
        ax = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        ay = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
        features['longitudinal_accel'] = float(ax)
        features['lateral_accel'] = float(abs(ay))
        features['acceleration_magnitude'] = float(np.hypot(ax, ay))
        
        # Angular motion
        features['yaw_rate'] = float(abs(ego_state.dynamic_car_state.angular_velocity))
        features['steering_angle'] = float(ego_state.tire_steering_angle)
        
        # Process all tracked objects with improved classification
        vehicles, pedestrians, long_vehicles = self._process_tracked_objects(
            ego_state, observation
        )
        
        features['vehicles'] = vehicles
        features['pedestrians'] = pedestrians
        features['long_vehicles'] = long_vehicles
        
        # Count vehicles in different zones
        features['vehicles_within_6m'] = sum(1 for v in vehicles if v['distance'] < 6.0)
        features['vehicles_within_8m'] = sum(1 for v in vehicles if v['distance'] < 8.0)
        features['moving_vehicles_8m'] = sum(1 for v in vehicles 
                                           if v['distance'] < 8.0 and v['speed'] > 6.0)
        
        # Find leading vehicle with better lane detection
        features['leading_vehicle'] = self._find_leading_vehicle(ego_state, vehicles)
        
        # Pedestrian analysis
        features['nearest_pedestrian'] = self._find_nearest_pedestrian(
            ego_state, pedestrians, features['ego_speed']
        )
        
        # Map features with error handling
        features['at_intersection'] = self._check_intersection(ego_state)
        features['near_crosswalk'] = self._check_crosswalk(ego_state)
        features['in_pickup_dropoff'] = self._check_pickup_dropoff(ego_state)
        features['lane_type'] = self._get_lane_type(ego_state)
        
        # Traffic light analysis
        features['has_traffic_light'] = bool(traffic_light_data)
        features['red_light'] = self._check_red_light(traffic_light_data)
        
        # Lane change detection
        features['lane_change_indicators'] = self._detect_lane_change_indicators(features)
        
        return features
    
    def _classify_scenario(self, features: Dict) -> Tuple[ScenarioType, float, str]:
        """
        Classify scenario with improved logic and priority ordering.
        Returns (scenario_type, confidence, debug_info)
        """
        # SAFETY-CRITICAL SCENARIOS (Highest Priority)
        
        # 1. WAITING_FOR_PEDESTRIAN_TO_CROSS
        # "Distance < 8m, time to intersection < 1.5s, ego not stopped, near crosswalk"
        if features['ego_speed'] > 0.1:  # Not stopped
            ped_result = self._check_pedestrian_scenario(features)
            if ped_result[0]:
                return ScenarioType.WAITING_FOR_PEDESTRIAN, ped_result[1], ped_result[2]
        
        # 2. STOPPING_WITH_LEAD
        # "Decel < -0.6 m/s², velocity < 0.3 m/s, lead < 6m"
        if (features['longitudinal_accel'] < -0.6 and 
            features['ego_speed'] < 0.3 and
            features['leading_vehicle'] is not None and
            features['leading_vehicle']['distance'] < 6.0):
            return (ScenarioType.STOPPING_WITH_LEAD, 0.95, 
                   f"Stopping with lead at {features['leading_vehicle']['distance']:.1f}m")
        
        # STATIC/LOW-SPEED SCENARIOS
        
        # 3. STATIONARY_IN_TRAFFIC
        # "Stationary with >6 vehicles within 8m"
        if features['ego_speed'] < 0.1 and features['vehicles_within_8m'] > 6:
            return (ScenarioType.STATIONARY_IN_TRAFFIC, 0.95,
                   f"Stationary with {features['vehicles_within_8m']} vehicles nearby")
        
        # 4. LOW_MAGNITUDE_SPEED
        # "0.3 < speed < 1.2 m/s with low acceleration, not in pickup/dropoff"
        if (0.3 < features['ego_speed'] < 1.2 and 
            features['acceleration_magnitude'] < 1.0 and
            not features['in_pickup_dropoff']):
            return (ScenarioType.LOW_MAGNITUDE_SPEED, 0.85,
                   "Low speed maneuver")
        
        # FOLLOWING SCENARIOS
        
        # 5. FOLLOWING_LANE_WITH_LEAD
        # "Speed > 3.5 m/s, lead speed > 3.5 m/s, distance < 7.5m, same lane"
        if (features['ego_speed'] > 3.5 and 
            features['leading_vehicle'] is not None):
            lead = features['leading_vehicle']
            if (lead['speed'] > 3.5 and 
                lead['distance'] < 7.5 and
                lead['same_lane_confidence'] > 0.8):
                return (ScenarioType.FOLLOWING_LANE_WITH_LEAD, 0.95,
                       f"Following at {lead['distance']:.1f}m")
        
        # 6. BEHIND_LONG_VEHICLE
        # "3m < longitudinal distance < 10m, length > 8m, same lane"
        for vehicle in features['long_vehicles']:
            if (3.0 < vehicle['longitudinal_distance'] < 10.0 and
                abs(vehicle['lateral_offset']) < 0.5 and
                vehicle['same_lane_confidence'] > 0.8):
                return (ScenarioType.BEHIND_LONG_VEHICLE, 0.9,
                       f"Behind {vehicle['length']:.1f}m vehicle")
        
        # INTERSECTION SCENARIOS
        
        # 7-9. Intersection maneuvers (check these before high-speed)
        if features['at_intersection'] and features['ego_speed'] > 0.1:
            turn_result = self._classify_turn(features)
            if turn_result[0] != ScenarioType.UNKNOWN:
                return turn_result
        
        # SPECIAL AREAS
        
        # 10. TRAVERSING_PICKUP_DROPOFF
        # "In pickup/dropoff area while not stopped"
        if features['in_pickup_dropoff'] and features['ego_speed'] > 0.1:
            return (ScenarioType.TRAVERSING_PICKUP_DROPOFF, 0.9,
                   "Traversing pickup/dropoff area")
        
        # HIGH-SPEED SCENARIOS
        
        # 11. HIGH_MAGNITUDE_SPEED
        # "Speed > 9 m/s with low acceleration"
        if features['ego_speed'] > 9.0 and features['acceleration_magnitude'] < 1.0:
            return (ScenarioType.HIGH_MAGNITUDE_SPEED, 0.95,
                   f"High speed: {features['ego_speed']:.1f} m/s")
        
        # DYNAMIC MANEUVERS
        
        # 12. HIGH_LATERAL_ACCELERATION
        # "1.5 < lateral_accel < 3 m/s² with high yaw rate, not turning"
        if (1.5 < features['lateral_accel'] < 3.0 and 
            features['yaw_rate'] > 0.2 and
            not features['at_intersection']):
            return (ScenarioType.HIGH_LATERAL_ACCELERATION, 0.85,
                   f"High lateral: {features['lateral_accel']:.2f} m/s²")
        
        # 13. CHANGING_LANE
        # Check lane change indicators
        if features['lane_change_indicators']['is_changing']:
            return (ScenarioType.CHANGING_LANE, 
                   features['lane_change_indicators']['confidence'],
                   "Lane change detected")
        
        # 14. NEAR_MULTIPLE_VEHICLES
        # "Nearby (distance < 8m) >6 moving vehicles, ego speed > 6 m/s"
        if features['ego_speed'] > 6.0 and features['moving_vehicles_8m'] > 6:
            return (ScenarioType.NEAR_MULTIPLE_VEHICLES, 0.85,
                   f"{features['moving_vehicles_8m']} moving vehicles nearby")
        
        # DEFAULT
        return (ScenarioType.UNKNOWN, 0.5, "No specific scenario detected")
    
    def _check_pedestrian_scenario(self, features: Dict) -> Tuple[bool, float, str]:
        """Enhanced pedestrian detection with better TTC calculation"""
        if not features['nearest_pedestrian']:
            return False, 0.0, ""
        
        ped = features['nearest_pedestrian']
        
        # Multiple criteria for pedestrian scenario
        criteria_met = 0
        total_criteria = 4
        
        # 1. Distance < 8m
        if ped['distance'] < 8.0:
            criteria_met += 1
        
        # 2. Time to collision < 1.5s (if paths intersect)
        if ped['ttc'] < 1.5:
            criteria_met += 1
        
        # 3. Near crosswalk
        if features['near_crosswalk']:
            criteria_met += 1
        
        # 4. Pedestrian is moving or in path
        if ped['is_crossing'] or ped['in_path']:
            criteria_met += 1
        
        # Need at least 2 criteria
        if criteria_met >= 2:
            confidence = 0.7 + (criteria_met - 2) * 0.1
            debug = (f"Pedestrian at {ped['distance']:.1f}m, "
                    f"TTC={ped['ttc']:.1f}s, "
                    f"criteria={criteria_met}/{total_criteria}")
            return True, confidence, debug
        
        return False, 0.0, ""
    
    def _process_tracked_objects(self, ego_state, observation):
        """Process tracked objects with improved classification"""
        vehicles = []
        pedestrians = []
        long_vehicles = []
        
        if not hasattr(observation, 'tracked_objects'):
            return vehicles, pedestrians, long_vehicles
        
        ego_heading = ego_state.rear_axle.heading
        
        for obj in observation.tracked_objects:
            # Transform to ego frame
            rel_x = obj.center.x - ego_state.rear_axle.x
            rel_y = obj.center.y - ego_state.rear_axle.y
            
            cos_h = np.cos(ego_heading)
            sin_h = np.sin(ego_heading)
            ego_x = rel_x * cos_h + rel_y * sin_h
            ego_y = -rel_x * sin_h + rel_y * cos_h
            
            distance = float(np.hypot(ego_x, ego_y))
            
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                obj_speed = float(np.hypot(obj.velocity.x, obj.velocity.y))
                
                # Calculate relative heading
                obj_heading = np.arctan2(obj.velocity.y, obj.velocity.x)
                heading_diff = abs(self._normalize_angle(obj_heading - ego_heading))
                same_direction = heading_diff < np.pi / 4  # Within 45 degrees
                
                vehicle_info = {
                    'distance': distance,
                    'longitudinal_distance': ego_x,
                    'lateral_offset': ego_y,
                    'speed': obj_speed,
                    'length': float(obj.box.length),
                    'width': float(obj.box.width),
                    'same_lane_confidence': self._calculate_same_lane_confidence(
                        ego_x, ego_y, same_direction
                    ),
                    'heading_diff': heading_diff
                }
                vehicles.append(vehicle_info)
                
                # Check for long vehicle
                if obj.box.length > 8.0:
                    long_vehicles.append(vehicle_info)
                    
            elif obj.tracked_object_type in [TrackedObjectType.PEDESTRIAN, 
                                            TrackedObjectType.BICYCLE]:
                # Enhanced pedestrian info
                pedestrians.append({
                    'distance': distance,
                    'position': {'x': ego_x, 'y': ego_y},
                    'velocity': {
                        'x': float(obj.velocity.x), 
                        'y': float(obj.velocity.y)
                    },
                    'speed': float(np.hypot(obj.velocity.x, obj.velocity.y))
                })
        
        return vehicles, pedestrians, long_vehicles
    
    def _find_leading_vehicle(self, ego_state, vehicles):
        """Find leading vehicle with improved same-lane detection"""
        leading = None
        min_distance = float('inf')
        
        for vehicle in vehicles:
            # Must be ahead
            if vehicle['longitudinal_distance'] <= 0:
                continue
            
            # Must be reasonably in same lane
            if vehicle['same_lane_confidence'] < 0.7:
                continue
            
            # Check if closer than current leading
            if vehicle['longitudinal_distance'] < min_distance:
                min_distance = vehicle['longitudinal_distance']
                leading = vehicle
        
        return leading
    
    def _find_nearest_pedestrian(self, ego_state, pedestrians, ego_speed):
        """Find nearest pedestrian with crossing analysis"""
        if not pedestrians:
            return None
        
        nearest = None
        min_ttc = float('inf')
        
        for ped in pedestrians:
            # Calculate time to collision
            ttc = self._calculate_pedestrian_ttc(ego_state, ped, ego_speed)
            
            # Check if pedestrian is crossing or in path
            is_crossing = abs(ped['velocity']['x']) > 0.5  # Moving laterally
            in_path = abs(ped['position']['y']) < 3.0 and ped['position']['x'] > 0
            
            ped_info = {
                'distance': ped['distance'],
                'ttc': ttc,
                'is_crossing': is_crossing,
                'in_path': in_path,
                'lateral_position': ped['position']['y']
            }
            
            if ttc < min_ttc:
                min_ttc = ttc
                nearest = ped_info
        
        return nearest
    
    def _calculate_same_lane_confidence(self, long_dist, lat_offset, same_direction):
        """Calculate confidence that a vehicle is in the same lane"""
        if not same_direction:
            return 0.0
        
        # Lane width typically 3.5m, so lateral offset should be < 1.75m
        lat_confidence = max(0, 1.0 - abs(lat_offset) / 1.75)
        
        # Reduce confidence for very close vehicles (might be in adjacent lane)
        if long_dist < 3.0:
            lat_confidence *= 0.8
        
        return lat_confidence
    
    def _calculate_pedestrian_ttc(self, ego_state, pedestrian, ego_speed):
        """Calculate time to collision with pedestrian"""
        if ego_speed < 0.1:
            return float('inf')
        
        # Simple TTC based on distance and closing speed
        closing_speed = ego_speed
        
        # If pedestrian is moving, account for that
        ped_speed = pedestrian['speed']
        if ped_speed > 0.1:
            # Rough approximation of closing speed
            closing_speed = max(0.1, ego_speed - ped_speed * 0.5)
        
        return pedestrian['distance'] / closing_speed
    
    def _detect_lane_change_indicators(self, features):
        """Detect lane change with multiple indicators"""
        indicators = {
            'is_changing': False,
            'confidence': 0.0
        }
        
        score = 0.0
        
        # Lateral acceleration (0.5-1.5 m/s²)
        if 0.5 < features['lateral_accel'] < 1.5:
            score += 0.3
        
        # Moderate speed (> 3 m/s)
        if features['ego_speed'] > 3.0:
            score += 0.2
        
        # Low yaw rate (not turning)
        if features['yaw_rate'] < 0.15:
            score += 0.2
        
        # Lateral velocity component
        if abs(features['ego_velocity_y']) > 0.3:
            score += 0.3
        
        indicators['confidence'] = min(0.9, score)
        indicators['is_changing'] = score > 0.6
        
        return indicators
    
    def _classify_turn(self, features):
        """Classify intersection turn type"""
        # Strong left turn indicators
        if features['yaw_rate'] > 0.15 or features['steering_angle'] > 0.1:
            return (ScenarioType.STARTING_LEFT_TURN, 0.85, "Starting left turn")
        
        # Strong right turn indicators  
        elif features['yaw_rate'] < -0.15 or features['steering_angle'] < -0.1:
            return (ScenarioType.STARTING_RIGHT_TURN, 0.85, "Starting right turn")
        
        # Straight through intersection with traffic light
        elif features['has_traffic_light']:
            return (ScenarioType.STARTING_STRAIGHT_INTERSECTION, 0.85,
                   "Straight through traffic light intersection")
        
        return (ScenarioType.UNKNOWN, 0.0, "")
    
    def _is_flickering(self):
        """Check if scenario detection is flickering"""
        if len(self.scenario_history) < 3:
            return False
        
        # Check if scenarios are alternating
        unique_recent = len(set(self.scenario_history[-3:]))
        return unique_recent > 2
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    # Map checking methods with better error handling
    def _check_intersection(self, ego_state) -> bool:
        """Check if at intersection with multiple indicators"""
        try:
            # Check lane connectors
            connectors = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=15.0,
                layers=[SemanticMapLayer.LANE_CONNECTOR]
            )
            has_connectors = len(connectors.get(SemanticMapLayer.LANE_CONNECTOR, [])) > 0
            
            # Also check for intersecting lanes
            if not has_connectors:
                lanes = self.map_api.get_proximal_map_objects(
                    ego_state.rear_axle,
                    radius=20.0,
                    layers=[SemanticMapLayer.LANE]
                )
                # Simple heuristic: multiple lanes at different angles = intersection
                lane_objects = lanes.get(SemanticMapLayer.LANE, [])
                if len(lane_objects) > 3:
                    has_connectors = True
            
            return has_connectors
        except Exception as e:
            logger.debug(f"Error checking intersection: {e}")
            return False
    
    def _check_crosswalk(self, ego_state) -> bool:
        """Check if near crosswalk"""
        try:
            crosswalks = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=12.0,  # Slightly larger radius
                layers=[SemanticMapLayer.CROSSWALK]
            )
            return len(crosswalks.get(SemanticMapLayer.CROSSWALK, [])) > 0
        except Exception as e:
            logger.debug(f"Error checking crosswalk: {e}")
            return False
    
    def _check_pickup_dropoff(self, ego_state) -> bool:
        """Check if in pickup/dropoff area with multiple methods"""
        try:
            # Method 1: Check lane properties
            lanes = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=5.0,
                layers=[SemanticMapLayer.LANE]
            )
            
            for lane in lanes.get(SemanticMapLayer.LANE, []):
                # Check various attributes that might indicate pickup/dropoff
                for attr in ['lane_type', 'lane_type_fid', 'type', 'name']:
                    if hasattr(lane, attr):
                        value = str(getattr(lane, attr)).lower()
                        if any(keyword in value for keyword in 
                              ['pickup', 'dropoff', 'loading', 'passenger', 
                               'taxi', 'rideshare', 'pull']):
                            return True
                
                # Check lane ID
                if hasattr(lane, 'id'):
                    lane_id = str(lane.id).lower()
                    if any(keyword in lane_id for keyword in 
                          ['pickup', 'dropoff', 'loading']):
                        return True
            
            # Method 2: Check for special map regions
            # This is map-specific but worth trying
            
            return False
        except Exception as e:
            logger.debug(f"Error checking pickup/dropoff: {e}")
            return False
    
    def _get_lane_type(self, ego_state) -> str:
        """Get current lane type"""
        try:
            lanes = self.map_api.get_proximal_map_objects(
                ego_state.rear_axle,
                radius=2.0,
                layers=[SemanticMapLayer.LANE]
            )
            
            for lane in lanes.get(SemanticMapLayer.LANE, []):
                if hasattr(lane, 'lane_type'):
                    return str(lane.lane_type)
            
            return "unknown"
        except:
            return "unknown"
    
    def _check_red_light(self, traffic_light_data) -> bool:
        """Check for red/yellow light"""
        if not traffic_light_data:
            return False
        
        for light in traffic_light_data:
            if hasattr(light, 'status'):
                status_name = getattr(light.status, 'name', str(light.status))
                if status_name in ['RED', 'YELLOW']:
                    return True
        
        return False


class HybridPlanner(AbstractPlanner):
    """
    Version 4 Hybrid Planner: Scenario-First Decision Making
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
        self._last_selected_planner = PlannerType.PDM
        
        # Initialize V4 switcher
        config = LeakyDDMConfigV4(**leaky_ddm_config) if leaky_ddm_config else LeakyDDMConfigV4()
        self._switcher = LeakyDDMSwitcherV4(config)
        
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
        
        logger.info("Initialized HybridPlanner V4 with scenario-first decision making")
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize all components"""
        logger.info("Initializing hybrid planner V4...")
        
        # Store map API
        self._map_api = initialization.map_api
        
        # Initialize improved scenario detector
        self._scenario_detector = ImprovedScenarioDetector(self._map_api)
        
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
        """Main planning function with scenario-first logic"""
        self._current_planner_input = current_input
        current_time = time.time() - self._start_time if self._start_time else 0.0
        
        # Extract current state
        ego_state, observation = current_input.history.current_state
        
        # CRITICAL: Scenario detection
        scenario_result = None
        if self._enable_scenario_detection and self._scenario_detector:
            scenario_result = self._scenario_detector.detect_scenario(
                ego_state, 
                observation, 
                current_input.traffic_light_data,
                self._route_lane_dict
            )
            
            # Log scenario changes and high-confidence detections
            if scenario_result.is_transition or scenario_result.confidence > 0.85:
                logger.info(
                    f"SCENARIO: {scenario_result.scenario_type.value} "
                    f"(conf={scenario_result.confidence:.2f}, "
                    f"persist={scenario_result.persistence}) - "
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
        
        # SCENARIO-FIRST SELECTION
        selected_trajectory = self._select_trajectory_v4(scenario_result)
        
        self._iteration += 1
        return selected_trajectory
    
    def _execute_pdm_planner(self, current_input, current_time):
        """Execute PDM and score with safety metrics"""
        start = time.time()
        try:
            trajectory = self.pdm_planner.compute_planner_trajectory(current_input)
            computation_time = time.time() - start
            
            # Score trajectory with enhanced metrics
            score_data = self._score_trajectory_enhanced(trajectory, 'pdm')
            
            self.current_pdm_trajectory = TrajectoryWithMetadata(
                trajectory=trajectory,
                planner_name='pdm',
                computation_time=computation_time,
                timestamp=current_time,
                score=score_data['total_score'],
                progress=score_data['progress'],
                is_valid=score_data['is_valid'],
                safety_metrics=score_data['safety_metrics']
            )
            self.last_pdm_time = current_time
            
        except Exception as e:
            logger.error(f"PDM failed: {e}")
            traceback.print_exc()
            # Don't raise - continue with cached trajectory
    
    def _execute_diffusion_planner(self, current_input, current_time):
        """Execute Diffusion and score with safety metrics"""
        start = time.time()
        try:
            trajectory = self.diffusion_planner.compute_planner_trajectory(current_input)
            computation_time = time.time() - start
            
            # Score trajectory
            score_data = self._score_trajectory_enhanced(trajectory, 'diffusion')
            
            self.current_diffusion_trajectory = TrajectoryWithMetadata(
                trajectory=trajectory,
                planner_name='diffusion',
                computation_time=computation_time,
                timestamp=current_time,
                score=score_data['total_score'],
                progress=score_data['progress'],
                is_valid=score_data['is_valid'],
                safety_metrics=score_data['safety_metrics']
            )
            self.last_diffusion_time = current_time
            
        except Exception as e:
            logger.error(f"Diffusion failed: {e}")
            # Continue with PDM on failure
    
    def _select_trajectory_v4(
        self, scenario_result: Optional[ScenarioDetectionResult]
    ) -> AbstractTrajectory:
        """Version 4 selection: Scenario-first with safety override"""
        if self.current_pdm_trajectory is None:
            raise RuntimeError("No PDM trajectory!")
        
        if self.current_diffusion_trajectory is None:
            logger.debug("No diffusion trajectory, using PDM")
            return self.current_pdm_trajectory.trajectory
        
        # Extract scores and safety metrics
        pdm_score = self.current_pdm_trajectory.score or 0.0
        diffusion_score = self.current_diffusion_trajectory.score or 0.0
        pdm_progress = self.current_pdm_trajectory.progress
        
        pdm_safety = self.current_pdm_trajectory.safety_metrics
        diffusion_safety = self.current_diffusion_trajectory.safety_metrics
        
        # Safety veto check
        safety_veto_diffusion = (
            not self.current_diffusion_trajectory.is_valid or
            diffusion_safety['no_collision'] < 0.5 or
            diffusion_safety['drivable_area'] < 0.5
        )
        
        # Use V4 switcher with all information
        selected_planner, metadata = self._switcher.update_and_select(
            pdm_score=pdm_score,
            diffusion_score=diffusion_score,
            pdm_progress=pdm_progress,
            safety_veto_diffusion=safety_veto_diffusion,
            scenario_result=scenario_result
        )
        
        # Log important decisions
        if selected_planner != self._last_selected_planner:
            logger.info(
                f"SWITCH: {self._last_selected_planner.value} → {selected_planner.value} | "
                f"Reason: {metadata.get('decision_reason', 'score-based')} | "
                f"PDM={pdm_score:.3f} Diff={diffusion_score:.3f}"
            )
            self._last_selected_planner = selected_planner
        
        # Log periodic status
        if self._iteration % 20 == 0:
            scenario_name = scenario_result.scenario_type.value if scenario_result else "unknown"
            logger.info(
                f"Status: Planner={selected_planner.value}, "
                f"Scenario={scenario_name}, "
                f"Switches={metadata.get('switch_count', 0)}"
            )
        
        # Return selected trajectory
        if selected_planner == PlannerType.PDM:
            return self.current_pdm_trajectory.trajectory
        else:
            return self.current_diffusion_trajectory.trajectory
    
    def _score_trajectory_enhanced(self, trajectory: AbstractTrajectory, planner_name: str) -> Dict:
        """Enhanced scoring with safety metrics"""
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
            
            # Extract comprehensive metrics
            total_score = float(scores[0])
            progress_normalized = float(self._scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, 0])
            
            # Safety metrics
            no_collision = float(self._scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 0])
            drivable_area = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 0])
            driving_direction = float(self._scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, 0])
            
            # Additional weighted metrics
            ttc = float(self._scorer._weighted_metrics[WeightedMetricIndex.TTC, 0])
            comfort = float(self._scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, 0])
            
            # Validity check
            is_valid = (no_collision > 0.5 and 
                       drivable_area > 0.5 and 
                       driving_direction > 0.5)
            
            return {
                'total_score': total_score,
                'progress': progress_normalized,
                'is_valid': is_valid,
                'safety_metrics': {
                    'no_collision': no_collision,
                    'drivable_area': drivable_area,
                    'driving_direction': driving_direction,
                    'ttc': ttc,
                    'comfort': comfort
                }
            }
            
        except Exception as e:
            logger.error(f"Error scoring {planner_name}: {e}")
            return {
                'total_score': 0.0,
                'progress': 0.0,
                'is_valid': False,
                'safety_metrics': {
                    'no_collision': 0.0,
                    'drivable_area': 0.0,
                    'driving_direction': 0.0,
                    'ttc': 0.0,
                    'comfort': 0.0
                }
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
        return "HybridPlanner_V4_ScenarioFirst"
    
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks