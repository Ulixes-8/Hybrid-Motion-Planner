"""
Enhanced Scoring Metrics for Hybrid Planner Version 5.3
Novel metrics to better distinguish between PDM and Diffusion planner strengths.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Point, LineString
from shapely import Polygon

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap


class EnhancedScoringMetrics:
    """
    Calculates novel metrics for trajectory evaluation that capture
    aspects not covered by standard PDM scoring.
    """
    
    def __init__(self):
        """Initialize the enhanced metrics calculator."""
        # Scenario-specific metric weights
        self.scenario_metric_weights = {
            'changing_lane': {
                'trajectory_efficiency': 1.2,
                'social_compliance': 1.0,
                'spatial_utilization': 1.1,
                'trajectory_consistency': 0.8,
                'maneuver_preparedness': 1.3
            },
            'high_lateral_acceleration': {
                'trajectory_efficiency': 1.1,
                'social_compliance': 0.9,
                'spatial_utilization': 1.2,
                'trajectory_consistency': 0.7,
                'maneuver_preparedness': 1.0
            },
            'low_magnitude_speed': {
                'trajectory_efficiency': 0.8,
                'social_compliance': 0.9,
                'spatial_utilization': 1.3,
                'trajectory_consistency': 1.1,
                'maneuver_preparedness': 0.9
            },
            'starting_left_turn': {
                'trajectory_efficiency': 1.0,
                'social_compliance': 1.1,
                'spatial_utilization': 1.2,
                'trajectory_consistency': 0.9,
                'maneuver_preparedness': 1.2
            },
            'starting_straight_traffic_light_intersection_traversal': {
                'trajectory_efficiency': 1.1,
                'social_compliance': 1.2,
                'spatial_utilization': 1.0,
                'trajectory_consistency': 1.0,
                'maneuver_preparedness': 1.1
            },
            'traversing_pickup_dropoff': {
                'trajectory_efficiency': 0.7,
                'social_compliance': 1.0,
                'spatial_utilization': 1.4,
                'trajectory_consistency': 1.2,
                'maneuver_preparedness': 0.8
            },
            'stopping_with_lead': {
                'trajectory_efficiency': 0.9,
                'social_compliance': 1.3,
                'spatial_utilization': 1.0,
                'trajectory_consistency': 1.2,
                'maneuver_preparedness': 0.9
            },
            'following_lane_with_lead': {
                'trajectory_efficiency': 1.0,
                'social_compliance': 1.4,
                'spatial_utilization': 0.9,
                'trajectory_consistency': 1.1,
                'maneuver_preparedness': 1.0
            },
            'high_magnitude_speed': {
                'trajectory_efficiency': 1.3,
                'social_compliance': 1.1,
                'spatial_utilization': 0.8,
                'trajectory_consistency': 1.2,
                'maneuver_preparedness': 1.1
            },
            'default': {
                'trajectory_efficiency': 1.0,
                'social_compliance': 1.0,
                'spatial_utilization': 1.0,
                'trajectory_consistency': 1.0,
                'maneuver_preparedness': 1.0
            }
        }
    
    def calculate_all_metrics(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
        map_api: AbstractMap,
        scenario_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate all enhanced metrics for the given trajectory.
        
        Returns dict with keys:
        - trajectory_efficiency
        - social_compliance
        - spatial_utilization
        - trajectory_consistency
        - maneuver_preparedness
        - weighted_total (scenario-adjusted combined score)
        """
        # Default scores in case of errors
        default_scores = {
            'trajectory_efficiency': 0.5,
            'social_compliance': 0.5,
            'spatial_utilization': 0.5,
            'trajectory_consistency': 0.5,
            'maneuver_preparedness': 0.5,
            'weighted_total': 0.5
        }
        
        try:
            # Calculate individual metrics with error handling for each
            try:
                efficiency = self._calculate_trajectory_efficiency(states, centerline)
            except Exception as e:
                efficiency = 0.5
                
            try:
                social = self._calculate_social_compliance(states, observation)
            except Exception as e:
                social = 0.5
                
            try:
                spatial = self._calculate_spatial_utilization(
                    states, initial_ego_state, drivable_area_map, observation, map_api
                )
            except Exception as e:
                spatial = 0.5
                
            try:
                consistency = self._calculate_trajectory_consistency(states)
            except Exception as e:
                consistency = 0.5
                
            try:
                preparedness = self._calculate_maneuver_preparedness(
                    states, route_lane_dict, map_api
                )
            except Exception as e:
                preparedness = 0.5
            
            # Get scenario-specific weights
            weights = self.scenario_metric_weights.get(
                scenario_type, 
                self.scenario_metric_weights['default']
            )
            
            # Calculate weighted total
            weighted_total = (
                weights['trajectory_efficiency'] * efficiency +
                weights['social_compliance'] * social +
                weights['spatial_utilization'] * spatial +
                weights['trajectory_consistency'] * consistency +
                weights['maneuver_preparedness'] * preparedness
            ) / 5.0
            
            return {
                'trajectory_efficiency': efficiency,
                'social_compliance': social,
                'spatial_utilization': spatial,
                'trajectory_consistency': consistency,
                'maneuver_preparedness': preparedness,
                'weighted_total': weighted_total
            }
            
        except Exception as e:
            # If anything fails, return default scores
            return default_scores
    
    def _calculate_trajectory_efficiency(
        self,
        states: npt.NDArray[np.float64],
        centerline: PDMPath
    ) -> float:
        """
        Measures path efficiency as the ratio of progress to path length.
        Higher scores indicate more direct/efficient paths.
        """
        # Handle batch dimension - take first trajectory
        if states.ndim == 3:
            states = states[0]
        
        # Calculate actual path length
        positions = states[:, StateIndex.X:StateIndex.Y+1]
        path_segments = np.diff(positions, axis=0)
        path_length = np.sum(np.linalg.norm(path_segments, axis=1))
        
        # Calculate progress along centerline
        start_point = Point(positions[0])
        end_point = Point(positions[-1])
        progress = centerline.project([start_point, end_point])
        centerline_progress = max(0.0, progress[1] - progress[0])
        
        # Calculate straight-line distance
        straight_line = np.linalg.norm(positions[-1] - positions[0])
        
        # Efficiency considers both progress and path directness
        if path_length > 0:
            # Progress efficiency: how much centerline progress per unit path
            progress_efficiency = min(1.0, centerline_progress / path_length)
            
            # Path directness: how straight the path is
            path_directness = min(1.0, straight_line / path_length)
            
            # Combined efficiency (weighted average)
            efficiency = 0.6 * progress_efficiency + 0.4 * path_directness
        else:
            efficiency = 0.0
        
        return float(np.clip(efficiency, 0.0, 1.0))
    
    def _calculate_social_compliance(
        self,
        states: npt.NDArray[np.float64],
        observation: PDMObservation
    ) -> float:
        """
        Evaluates how well ego maintains socially acceptable behavior
        relative to surrounding traffic.
        """
        if states.ndim == 3:
            states = states[0]
        
        social_scores = []
        
        # Use the number of time steps from states, not observation
        for t in range(states.shape[0]):
            ego_state = states[t]
            ego_speed = np.hypot(
                ego_state[StateIndex.VELOCITY_X],
                ego_state[StateIndex.VELOCITY_Y]
            )
            ego_pos = ego_state[StateIndex.X:StateIndex.Y+1]
            
            # Find nearby vehicles from observation.unique_objects
            nearby_vehicles = []
            try:
                for token, obj in observation.unique_objects.items():
                    if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                        obj_pos = np.array([obj.box.center.x, obj.box.center.y])
                        distance = np.linalg.norm(obj_pos - ego_pos)
                        if distance < 30.0:  # 30m radius
                            obj_speed = np.hypot(obj.velocity.x, obj.velocity.y)
                            nearby_vehicles.append({
                                'distance': distance,
                                'speed': obj_speed,
                                'position': obj_pos,
                                'heading': obj.box.center.heading
                            })
            except Exception as e:
                # If we can't access unique_objects, skip this timestep
                continue
            
            if nearby_vehicles:
                # Speed harmonization
                traffic_speeds = [v['speed'] for v in nearby_vehicles]
                mean_traffic_speed = np.mean(traffic_speeds)
                
                # Score based on speed difference
                speed_diff = abs(ego_speed - mean_traffic_speed)
                speed_harmony_score = np.exp(-speed_diff / 5.0)
                
                # Gap maintenance for vehicles ahead
                lead_vehicles = [
                    v for v in nearby_vehicles
                    if self._is_vehicle_ahead(ego_state, v)
                ]
                
                if lead_vehicles:
                    # Find closest lead vehicle
                    closest_lead = min(lead_vehicles, key=lambda v: v['distance'])
                    gap = closest_lead['distance']
                    
                    # Time headway calculation
                    if ego_speed > 0.1:
                        time_headway = gap / ego_speed
                        # Ideal headway is 1.5-2.5 seconds
                        headway_score = np.exp(-abs(time_headway - 2.0) / 1.0)
                    else:
                        # Stationary - check minimum gap
                        headway_score = 1.0 if gap > 3.0 else gap / 3.0
                    
                    social_score = 0.6 * speed_harmony_score + 0.4 * headway_score
                else:
                    social_score = speed_harmony_score
                
                social_scores.append(social_score)
            else:
                # No nearby vehicles - neutral score
                social_scores.append(0.8)
        
        return float(np.mean(social_scores)) if social_scores else 0.8
    
    def _calculate_spatial_utilization(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
        drivable_area_map: PDMOccupancyMap,
        observation: PDMObservation,
        map_api: AbstractMap
    ) -> float:
        """
        Measures how efficiently ego uses available drivable space.
        Important for tight maneuvers and parking scenarios.
        """
        if states.ndim == 3:
            states = states[0]
        
        utilization_scores = []
        vehicle_params = initial_ego_state.car_footprint.vehicle_parameters
        
        # Process each timestep
        for t in range(states.shape[0]):
            ego_state = states[t]
            ego_center = Point(ego_state[StateIndex.X], ego_state[StateIndex.Y])
            
            # Get current lane if available
            try:
                ego_se2 = StateSE2(
                    ego_state[StateIndex.X],
                    ego_state[StateIndex.Y],
                    ego_state[StateIndex.HEADING]
                )
                lanes = map_api.get_proximal_map_objects(
                    ego_se2,
                    radius=5.0,
                    layers=[SemanticMapLayer.LANE]
                )
                current_lanes = lanes.get(SemanticMapLayer.LANE, [])
            except:
                current_lanes = []
            
            # Calculate lane centering score
            if current_lanes:
                # Find closest lane
                min_distance = float('inf')
                for lane in current_lanes:
                    if hasattr(lane, 'baseline_path'):
                        lane_center = LineString([
                            (p.x, p.y) for p in lane.baseline_path.discrete_path
                        ])
                        distance = ego_center.distance(lane_center)
                        min_distance = min(min_distance, distance)
                
                # Score based on distance from lane center
                if min_distance < float('inf'):
                    centering_score = np.exp(-min_distance / 1.5)
                else:
                    centering_score = 0.5
            else:
                centering_score = 0.5
            
            # Calculate clearance to obstacles
            min_clearance = float('inf')
            try:
                # Access observation at current time if possible
                if hasattr(observation, '__getitem__'):
                    obs_at_time = observation[t]
                    for token, polygon in obs_at_time.items():
                        if 'red_light' not in token:
                            distance = ego_center.distance(polygon)
                            min_clearance = min(min_clearance, distance)
                else:
                    # Fallback: use unique_objects
                    for token, obj in observation.unique_objects.items():
                        if 'red_light' not in token:
                            obj_center = Point(obj.box.center.x, obj.box.center.y)
                            distance = ego_center.distance(obj_center)
                            min_clearance = min(min_clearance, distance)
            except:
                # If we can't access obstacles, use default clearance
                min_clearance = 10.0
            
            # In tight spaces, reward balanced clearances
            if min_clearance < 5.0:
                # Optimal clearance is around 1.5-2.0m
                clearance_score = np.exp(-abs(min_clearance - 1.75) / 1.0)
                utilization_score = 0.3 * centering_score + 0.7 * clearance_score
            else:
                # In open spaces, lane centering is more important
                utilization_score = 0.8 * centering_score + 0.2
            
            utilization_scores.append(utilization_score)
        
        return float(np.mean(utilization_scores)) if utilization_scores else 0.5
    
    def _calculate_trajectory_consistency(
        self,
        states: npt.NDArray[np.float64]
    ) -> float:
        """
        Evaluates the smoothness and predictability of the trajectory.
        Penalizes erratic changes in acceleration and steering.
        """
        if states.ndim == 3:
            states = states[0]
        
        # Extract relevant signals
        accel_x = states[:, StateIndex.ACCELERATION_X]
        accel_y = states[:, StateIndex.ACCELERATION_Y]
        angular_vel = states[:, StateIndex.ANGULAR_VELOCITY]
        
        # Calculate jerk (rate of change of acceleration)
        if len(accel_x) > 1:
            jerk_x = np.diff(accel_x)
            jerk_y = np.diff(accel_y)
            jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2)
            
            # RMS jerk as consistency measure
            rms_jerk = np.sqrt(np.mean(jerk_magnitude**2))
            jerk_score = np.exp(-rms_jerk / 2.0)
        else:
            jerk_score = 1.0
        
        # Angular acceleration consistency
        if len(angular_vel) > 1:
            angular_accel = np.diff(angular_vel)
            rms_angular_accel = np.sqrt(np.mean(angular_accel**2))
            angular_score = np.exp(-rms_angular_accel / 1.0)
        else:
            angular_score = 1.0
        
        # Combined consistency score
        consistency = 0.7 * jerk_score + 0.3 * angular_score
        
        return float(np.clip(consistency, 0.0, 1.0))
    
    def _calculate_maneuver_preparedness(
        self,
        states: npt.NDArray[np.float64],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        map_api: AbstractMap
    ) -> float:
        """
        Evaluates how well positioned the vehicle is for upcoming maneuvers.
        Considers lane position, speed, and heading alignment.
        """
        if states.ndim == 3:
            states = states[0]
        
        if not route_lane_dict:
            return 0.8  # No route info - neutral score
        
        preparedness_scores = []
        
        # Sample at key points along trajectory
        sample_indices = np.linspace(0, len(states)-1, min(10, len(states)), dtype=int)
        
        for idx in sample_indices:
            ego_state = states[idx]
            ego_pos = Point(ego_state[StateIndex.X], ego_state[StateIndex.Y])
            ego_heading = ego_state[StateIndex.HEADING]
            ego_speed = np.hypot(
                ego_state[StateIndex.VELOCITY_X],
                ego_state[StateIndex.VELOCITY_Y]
            )
            
            # Check position relative to route
            on_route_score = 0.0
            heading_alignment_score = 0.0
            
            try:
                for lane_id, lane in route_lane_dict.items():
                    if hasattr(lane, 'baseline_path') and hasattr(lane.baseline_path, 'discrete_path'):
                        discrete_path = lane.baseline_path.discrete_path
                        if len(discrete_path) > 1:
                            lane_line = LineString([
                                (p.x, p.y) for p in discrete_path
                            ])
                            distance = ego_pos.distance(lane_line)
                            
                            # Score for being on route
                            if distance < 5.0:
                                on_route_score = max(on_route_score, np.exp(-distance / 2.0))
                                
                                # Check heading alignment with lane direction
                                # Find closest point on lane
                                closest_point = lane_line.interpolate(lane_line.project(ego_pos))
                                
                                # Get lane direction at closest point
                                if lane_line.length > 0:
                                    # Sample points around closest point
                                    proj_dist = lane_line.project(ego_pos)
                                    pt1 = lane_line.interpolate(max(0, proj_dist - 1.0))
                                    pt2 = lane_line.interpolate(min(lane_line.length, proj_dist + 1.0))
                                    
                                    lane_heading = np.arctan2(
                                        pt2.y - pt1.y,
                                        pt2.x - pt1.x
                                    )
                                    
                                    # Calculate heading difference
                                    heading_diff = abs(self._normalize_angle(ego_heading - lane_heading))
                                    heading_alignment_score = max(
                                        heading_alignment_score,
                                        np.exp(-heading_diff / 0.5)
                                    )
            except Exception as e:
                # If route processing fails, use default scores
                on_route_score = 0.5
                heading_alignment_score = 0.5
            
            # Speed appropriateness (lower speeds for complex areas)
            try:
                # Check if in intersection - use StateSE2 for map_api
                ego_se2 = StateSE2(
                    ego_state[StateIndex.X],
                    ego_state[StateIndex.Y],
                    ego_state[StateIndex.HEADING]
                )
                if map_api.is_in_layer(ego_se2, layer=SemanticMapLayer.INTERSECTION):
                    speed_score = np.exp(-max(0, ego_speed - 5.0) / 3.0)
                else:
                    # Normal road - moderate speeds preferred
                    speed_score = np.exp(-abs(ego_speed - 10.0) / 5.0)
            except:
                speed_score = 0.8
            
            # Combined preparedness
            preparedness = (
                0.4 * on_route_score +
                0.4 * heading_alignment_score +
                0.2 * speed_score
            )
            preparedness_scores.append(preparedness)
        
        return float(np.mean(preparedness_scores)) if preparedness_scores else 0.5
    
    def _is_vehicle_ahead(
        self,
        ego_state: npt.NDArray[np.float64],
        vehicle: Dict
    ) -> bool:
        """Check if a vehicle is ahead of ego."""
        ego_pos = ego_state[StateIndex.X:StateIndex.Y+1]
        ego_heading = ego_state[StateIndex.HEADING]
        
        # Vector to other vehicle
        rel_pos = vehicle['position'] - ego_pos
        
        # Check if in front (dot product with heading vector)
        heading_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        longitudinal_dist = np.dot(rel_pos, heading_vec)
        
        return longitudinal_dist > 0
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle