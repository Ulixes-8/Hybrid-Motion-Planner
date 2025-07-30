"""
Enhanced PDM Scorer for Hybrid Planner Version 6
Implements graduated scoring, scenario-aware metrics, and improved differentiation
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Optional
from enum import IntEnum
import logging

from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    StateIndex, MultiMetricIndex, WeightedMetricIndex, EgoAreaIndex, BBCoordsIndex
)
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from shapely import Point

logger = logging.getLogger(__name__)


class ExtendedWeightedMetricIndex(IntEnum):
    """Extended metrics for enhanced scoring"""
    PROGRESS = 0
    TTC = 1
    COMFORTABLE = 2
    LANE_CENTERING = 3
    TRAJECTORY_SMOOTHNESS = 4
    NEAR_MISS = 5
    
    @staticmethod
    def size() -> int:
        return len(ExtendedWeightedMetricIndex)


class ScenarioType:
    """Scenario type enumeration matching the system"""
    # PDM WINS
    BEHIND_LONG_VEHICLE = "behind_long_vehicle"
    FOLLOWING_LANE_WITH_LEAD = "following_lane_with_lead"
    HIGH_MAGNITUDE_SPEED = "high_magnitude_speed"
    NEAR_MULTIPLE_VEHICLES = "near_multiple_vehicles"
    STARTING_RIGHT_TURN = "starting_right_turn"
    STATIONARY_IN_TRAFFIC = "stationary_in_traffic"
    STOPPING_WITH_LEAD = "stopping_with_lead"
    WAITING_FOR_PEDESTRIAN = "waiting_for_pedestrian_to_cross"
    
    # DIFFUSION WINS
    CHANGING_LANE = "changing_lane"
    HIGH_LATERAL_ACCELERATION = "high_lateral_acceleration"
    LOW_MAGNITUDE_SPEED = "low_magnitude_speed"
    STARTING_LEFT_TURN = "starting_left_turn"
    STARTING_STRAIGHT_INTERSECTION = "starting_straight_traffic_light_intersection_traversal"
    TRAVERSING_PICKUP_DROPOFF = "traversing_pickup_dropoff"
    
    UNKNOWN = "unknown"


class EnhancedPDMScorer(PDMScorer):
    """
    Enhanced PDM Scorer with graduated scoring and scenario awareness
    """
    
    def __init__(self, proposal_sampling, enable_graduated_scoring=True):
        """Initialize enhanced scorer"""
        super().__init__(proposal_sampling)
        self._enable_graduated_scoring = enable_graduated_scoring
        self._scenario_type = None
        
        # Extended metrics storage
        self._extended_weighted_metrics = None
        self._graduated_multi_metrics = None
        
        # Scenario-specific weight configurations
        self._scenario_weights = self._initialize_scenario_weights()
        
    def _initialize_scenario_weights(self) -> Dict:
        """Initialize scenario-specific metric weights"""
        base_weights = np.zeros(ExtendedWeightedMetricIndex.size(), dtype=np.float64)
        base_weights[ExtendedWeightedMetricIndex.PROGRESS] = 5.0
        base_weights[ExtendedWeightedMetricIndex.TTC] = 5.0
        base_weights[ExtendedWeightedMetricIndex.COMFORTABLE] = 2.0
        base_weights[ExtendedWeightedMetricIndex.LANE_CENTERING] = 1.5
        base_weights[ExtendedWeightedMetricIndex.TRAJECTORY_SMOOTHNESS] = 1.0
        base_weights[ExtendedWeightedMetricIndex.NEAR_MISS] = 0.0  # Disabled due to compatibility
        
        scenario_adjustments = {
            ScenarioType.WAITING_FOR_PEDESTRIAN: {
                ExtendedWeightedMetricIndex.TTC: 8.0,
                ExtendedWeightedMetricIndex.PROGRESS: 2.0,
                ExtendedWeightedMetricIndex.COMFORTABLE: 3.0,  # Increase comfort instead of near_miss
            },
            ScenarioType.TRAVERSING_PICKUP_DROPOFF: {
                ExtendedWeightedMetricIndex.COMFORTABLE: 5.0,
                ExtendedWeightedMetricIndex.PROGRESS: 3.0,
                ExtendedWeightedMetricIndex.TRAJECTORY_SMOOTHNESS: 3.0,
            },
            ScenarioType.HIGH_MAGNITUDE_SPEED: {
                ExtendedWeightedMetricIndex.COMFORTABLE: 4.0,
                ExtendedWeightedMetricIndex.TTC: 7.0,
                ExtendedWeightedMetricIndex.LANE_CENTERING: 3.0,
            },
            ScenarioType.CHANGING_LANE: {
                ExtendedWeightedMetricIndex.COMFORTABLE: 3.0,
                ExtendedWeightedMetricIndex.LANE_CENTERING: 4.0,
                ExtendedWeightedMetricIndex.TRAJECTORY_SMOOTHNESS: 4.0,
            },
            ScenarioType.LOW_MAGNITUDE_SPEED: {
                ExtendedWeightedMetricIndex.PROGRESS: 2.0,
                ExtendedWeightedMetricIndex.COMFORTABLE: 4.0,
                ExtendedWeightedMetricIndex.TRAJECTORY_SMOOTHNESS: 3.0,
            },
            ScenarioType.FOLLOWING_LANE_WITH_LEAD: {
                ExtendedWeightedMetricIndex.TTC: 6.0,
                ExtendedWeightedMetricIndex.COMFORTABLE: 3.0,
                ExtendedWeightedMetricIndex.LANE_CENTERING: 2.0,
            },
            ScenarioType.STATIONARY_IN_TRAFFIC: {
                ExtendedWeightedMetricIndex.PROGRESS: 1.0,
                ExtendedWeightedMetricIndex.COMFORTABLE: 1.0,
                ExtendedWeightedMetricIndex.TTC: 6.0,  # Increase TTC instead of near_miss
            },
        }
        
        return {'base': base_weights, 'adjustments': scenario_adjustments}
    
    def score_proposals_enhanced(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        drivable_area_map: PDMOccupancyMap,
        map_api: AbstractMap,
        scenario_type: Optional[str] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Enhanced scoring with scenario awareness and graduated metrics
        """
        self._scenario_type = scenario_type or ScenarioType.UNKNOWN
        
        # Call parent initialization
        self._reset(
            states, initial_ego_state, observation, centerline,
            route_lane_dict, drivable_area_map, map_api
        )
        
        # Initialize extended metrics
        self._extended_weighted_metrics = np.zeros(
            (ExtendedWeightedMetricIndex.size(), self._num_proposals),
            dtype=np.float64
        )
        self._graduated_multi_metrics = np.ones(
            (len(MultiMetricIndex), self._num_proposals),
            dtype=np.float64
        )
        
        # Calculate ego areas (used by multiple metrics)
        self._calculate_ego_area()
        
        # Calculate graduated multiplicative metrics
        if self._enable_graduated_scoring:
            self._calculate_graduated_no_at_fault_collision()
            self._calculate_graduated_drivable_area_compliance()
            self._calculate_graduated_driving_direction_compliance()
        else:
            # Use original binary metrics
            self._calculate_no_at_fault_collision()
            self._calculate_driving_direction_compliance()
            self._calculate_drivable_area_compliance()
            self._graduated_multi_metrics = self._multi_metrics.copy()
        
        # Calculate weighted metrics (original and new)
        self._calculate_scenario_aware_progress()
        self._calculate_graduated_ttc()
        self._calculate_graduated_comfort()
        self._calculate_lane_centering_score()
        self._calculate_trajectory_smoothness()
        # Temporarily disable near miss penalty due to compatibility issues
        # self._calculate_near_miss_penalty()
        # Set default near miss scores
        self._extended_weighted_metrics[ExtendedWeightedMetricIndex.NEAR_MISS] = \
            np.ones(self._num_proposals, dtype=np.float64)
        
        # Aggregate with scenario-specific weights
        return self._aggregate_scores_enhanced()
    
    def _calculate_graduated_no_at_fault_collision(self):
        """Graduated collision scoring based on severity"""
        # Start with the parent's collision detection
        self._calculate_no_at_fault_collision()
        
        # Convert binary to graduated based on collision type
        for proposal_idx in range(self._num_proposals):
            if self._multi_metrics[MultiMetricIndex.NO_COLLISION, proposal_idx] < 1.0:
                # Apply graduated penalty based on what was hit
                base_score = self._multi_metrics[MultiMetricIndex.NO_COLLISION, proposal_idx]
                
                # More lenient for objects, strict for VRUs
                if base_score == 0.0:  # VRU collision
                    self._graduated_multi_metrics[MultiMetricIndex.NO_COLLISION, proposal_idx] = 0.0
                elif base_score == 0.5:  # Object collision
                    self._graduated_multi_metrics[MultiMetricIndex.NO_COLLISION, proposal_idx] = 0.3
            else:
                self._graduated_multi_metrics[MultiMetricIndex.NO_COLLISION, proposal_idx] = 1.0
    
    def _calculate_graduated_drivable_area_compliance(self):
        """Graduated scoring for drivable area violations"""
        # For now, use the parent's binary calculation
        # A full implementation would require analyzing the actual distance outside drivable area
        self._calculate_drivable_area_compliance()
        
        # Copy binary scores to graduated metrics
        for proposal_idx in range(self._num_proposals):
            self._graduated_multi_metrics[MultiMetricIndex.DRIVABLE_AREA, proposal_idx] = \
                self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, proposal_idx]
    
    def _calculate_graduated_driving_direction_compliance(self):
        """Graduated scoring for driving direction compliance"""
        # Use parent's calculation but apply graduated scoring
        self._calculate_driving_direction_compliance()
        
        # Copy and enhance the scores
        for proposal_idx in range(self._num_proposals):
            base_score = self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, proposal_idx]
            
            # Already graduated (0, 0.5, 1), but we can make it smoother
            if base_score == 0.5:
                # Calculate actual distance driven against traffic
                # This would require more detailed calculation, for now keep as is
                self._graduated_multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, proposal_idx] = 0.5
            else:
                self._graduated_multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, proposal_idx] = base_score
    
    def _calculate_scenario_aware_progress(self):
        """Calculate progress with scenario-specific adjustments"""
        # Calculate base progress
        self._calculate_progress()
        
        # Adjust expectations based on scenario
        slow_progress_scenarios = [
            ScenarioType.LOW_MAGNITUDE_SPEED,
            ScenarioType.TRAVERSING_PICKUP_DROPOFF,
            ScenarioType.WAITING_FOR_PEDESTRIAN,
            ScenarioType.STATIONARY_IN_TRAFFIC,
            ScenarioType.STOPPING_WITH_LEAD,
        ]
        
        progress_threshold = 0.05 if self._scenario_type in slow_progress_scenarios else 0.1
        
        # Normalize progress with scenario-aware threshold
        raw_progress = self._progress_raw * self._graduated_multi_metrics[MultiMetricIndex.NO_COLLISION]
        max_raw_progress = np.max(raw_progress)
        
        if max_raw_progress > progress_threshold:
            normalized_progress = raw_progress / max_raw_progress
        else:
            normalized_progress = np.ones(len(raw_progress), dtype=np.float64)
            normalized_progress[self._graduated_multi_metrics[MultiMetricIndex.NO_COLLISION] == 0.0] = 0.0
        
        self._extended_weighted_metrics[ExtendedWeightedMetricIndex.PROGRESS] = normalized_progress
    
    def _calculate_graduated_ttc(self):
        """Graduated TTC scoring instead of binary"""
        # Calculate base TTC
        self._calculate_ttc()
        
        # Find actual TTC values for graduated scoring
        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        
        for proposal_idx in range(self._num_proposals):
            if self._ttc_time_idcs[proposal_idx] < np.inf:
                # Calculate actual TTC value
                ttc_seconds = self._ttc_time_idcs[proposal_idx] * self._proposal_sampling.interval_length
                
                if ttc_seconds >= 2.0:
                    ttc_scores[proposal_idx] = 1.0
                elif ttc_seconds >= 0.95:
                    ttc_scores[proposal_idx] = 0.5 + 0.5 * (ttc_seconds - 0.95) / 1.05
                elif ttc_seconds >= 0.5:
                    ttc_scores[proposal_idx] = 0.5 * (ttc_seconds - 0.5) / 0.45
                else:
                    ttc_scores[proposal_idx] = 0.0
            else:
                ttc_scores[proposal_idx] = 1.0
        
        self._extended_weighted_metrics[ExtendedWeightedMetricIndex.TTC] = ttc_scores
    
    def _calculate_graduated_comfort(self):
        """Graduated comfort scoring based on violation magnitude"""
        # Get the comfort metrics from parent
        self._calculate_is_comfortable()
        
        # For now, copy the binary comfort score
        # In a full implementation, we would calculate graduated penalties
        self._extended_weighted_metrics[ExtendedWeightedMetricIndex.COMFORTABLE] = \
            self._weighted_metrics[WeightedMetricIndex.COMFORTABLE]
    
    def _calculate_lane_centering_score(self):
        """Score based on lane centering quality"""
        lane_center_scores = np.ones(self._num_proposals, dtype=np.float64)
        
        for proposal_idx in range(self._num_proposals):
            # Penalize being in multiple lanes
            multiple_lane_count = np.sum(
                self._ego_areas[proposal_idx, :, EgoAreaIndex.MULTIPLE_LANES]
            )
            
            if multiple_lane_count > 0:
                # Graduated penalty based on duration in multiple lanes
                penalty_factor = multiple_lane_count / (self._proposal_sampling.num_poses + 1)
                lane_center_scores[proposal_idx] *= (1.0 - 0.3 * penalty_factor)
            
            # Additional penalty for non-drivable area
            if self._ego_areas[proposal_idx, :, EgoAreaIndex.NON_DRIVABLE_AREA].any():
                lane_center_scores[proposal_idx] *= 0.5
        
        self._extended_weighted_metrics[ExtendedWeightedMetricIndex.LANE_CENTERING] = lane_center_scores
    
    def _calculate_trajectory_smoothness(self):
        """Score trajectory smoothness and consistency"""
        smoothness_scores = np.ones(self._num_proposals, dtype=np.float64)
        
        for proposal_idx in range(self._num_proposals):
            # Calculate heading changes
            headings = self._states[proposal_idx, :, StateIndex.HEADING]
            heading_changes = np.diff(headings)
            
            # Unwrap heading to handle discontinuities
            heading_changes = np.where(
                heading_changes > np.pi, heading_changes - 2*np.pi, heading_changes
            )
            heading_changes = np.where(
                heading_changes < -np.pi, heading_changes + 2*np.pi, heading_changes
            )
            
            # Penalize oscillations
            sign_changes = np.sum(np.diff(np.sign(heading_changes)) != 0)
            oscillation_penalty = sign_changes / len(heading_changes)
            
            # Calculate curvature consistency
            curvatures = heading_changes / self._proposal_sampling.interval_length
            curvature_variance = np.var(curvatures) if len(curvatures) > 0 else 0.0
            
            # Combined smoothness score
            smoothness_scores[proposal_idx] = np.exp(
                -2.0 * oscillation_penalty - 0.1 * curvature_variance
            )
        
        self._extended_weighted_metrics[ExtendedWeightedMetricIndex.TRAJECTORY_SMOOTHNESS] = smoothness_scores
    
    def _calculate_near_miss_penalty(self):
        """Penalize trajectories that come too close to obstacles"""
        near_miss_scores = np.ones(self._num_proposals, dtype=np.float64)
        
        MIN_SAFE_DISTANCE = {
            'VRU': 2.0,      # 2m for pedestrians/cyclists
            'VEHICLE': 1.5,  # 1.5m for vehicles
            'OBJECT': 0.5    # 0.5m for static objects
        }
        
        try:
            for proposal_idx in range(self._num_proposals):
                min_distances = []
                
                for time_idx in range(self._proposal_sampling.num_poses + 1):
                    ego_polygon = self._ego_polygons[proposal_idx, time_idx]
                    
                    # Check if we have access to geometries and tokens
                    if not hasattr(self._observation[time_idx], 'geometries'):
                        continue
                    
                    # Get all geometries and tokens
                    all_geometries = self._observation[time_idx].geometries
                    all_tokens = self._observation[time_idx].tokens if hasattr(self._observation[time_idx], 'tokens') else []
                    
                    for geometry_idx, geometry in enumerate(all_geometries):
                        if geometry_idx < len(all_tokens):
                            token = all_tokens[geometry_idx]
                            
                            # Skip red lights and already collided objects
                            if (self._observation.red_light_token in token or
                                token in self._observation.collided_track_ids):
                                continue
                        else:
                            token = None
                        
                        # Calculate distance
                        distance = ego_polygon.distance(geometry)
                        
                        # Only consider objects within 3 meters
                        if distance < 3.0:
                            # Determine object type
                            object_type = 'OBJECT'  # Default
                            if token and token in self._observation.unique_objects:
                                tracked_object = self._observation.unique_objects[token]
                                if tracked_object.tracked_object_type in AGENT_TYPES:
                                    object_type = 'VRU' if 'PEDESTRIAN' in str(tracked_object.tracked_object_type) else 'VEHICLE'
                            
                            min_safe = MIN_SAFE_DISTANCE.get(object_type, 1.0)
                            
                            if distance < min_safe:
                                min_distances.append(distance / min_safe)
                
                # Calculate penalty based on minimum distances
                if min_distances:
                    avg_proximity = np.mean(min_distances)
                    near_miss_scores[proposal_idx] = avg_proximity
            
        except Exception as e:
            logger.warning(f"Error calculating near miss penalty: {e}")
            # Return default scores if calculation fails
            near_miss_scores = np.ones(self._num_proposals, dtype=np.float64)
        
        self._extended_weighted_metrics[ExtendedWeightedMetricIndex.NEAR_MISS] = near_miss_scores
    
    def _aggregate_scores_enhanced(self) -> npt.NDArray[np.float64]:
        """Aggregate scores with scenario-specific weights"""
        # Get scenario-specific weights
        weights = self._scenario_weights['base'].copy()
        
        if self._scenario_type in self._scenario_weights['adjustments']:
            adjustments = self._scenario_weights['adjustments'][self._scenario_type]
            for idx, value in adjustments.items():
                weights[idx] = value
        
        # Accumulate multiplicative metrics
        multiplicative_scores = self._graduated_multi_metrics.prod(axis=0)
        
        # Ensure extended weighted metrics are initialized
        if self._extended_weighted_metrics is None or self._extended_weighted_metrics.shape[0] == 0:
            # Fallback to parent aggregation if enhanced metrics not available
            return super()._aggregate_scores()
        
        # Calculate weighted sum with scenario-specific weights
        weighted_scores = (self._extended_weighted_metrics * weights[:, None]).sum(axis=0)
        weighted_scores /= weights.sum() if weights.sum() > 0 else 1.0
        
        # Final scores
        final_scores = multiplicative_scores * weighted_scores
        
        return final_scores