import logging
import time
from typing import List, Type, Optional, Dict
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


@dataclass
class TrajectoryWithMetadata:
    """Container for trajectory with additional metadata."""
    trajectory: AbstractTrajectory
    planner_name: str
    computation_time: float
    timestamp: float
    score: Optional[float] = None
    progress_m: Optional[float] = None  # Raw progress in meters
    is_valid: Optional[bool] = None


class HybridPlanner(AbstractPlanner):
    """
    Hybrid planner using Leaky DDM switching between PDM-Closed and Diffusion Planner.
    """
    
    # Inherited property
    requires_scenario: bool = False
    
    def __init__(
        self,
        pdm_planner: AbstractPlanner,
        diffusion_planner: AbstractPlanner,
        pdm_frequency: float = 10.0,  # Hz
        diffusion_frequency: float = 2.0,  # Hz
        leaky_ddm_config: Optional[Dict] = None,
    ):
        """
        Initialize the hybrid planner.
        
        :param pdm_planner: Instance of PDM-Closed planner
        :param diffusion_planner: Instance of Diffusion planner
        :param pdm_frequency: Frequency for PDM planner execution (Hz)
        :param diffusion_frequency: Frequency for Diffusion planner execution (Hz)
        :param leaky_ddm_config: Configuration for Leaky DDM switcher
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

        logger.info(f"Initialized HybridPlanner with Leaky DDM switching")
    
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
        
        # Get ego speed for trapped detection
        ego_speed_mps = float(np.hypot(
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        ))
        
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
                    progress_m=score_data['progress_raw'],  # Use raw progress in meters
                    is_valid=score_data['is_valid']
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
                    progress_m=score_data['progress_raw'],  # Use raw progress in meters
                    is_valid=score_data['is_valid']
                )
                self.last_diffusion_time = current_time
                
                logger.info(f"Diffusion trajectory computed in {computation_time:.3f}s, score: {score_data['total_score']:.3f}")
            except Exception as e:
                logger.error(f"Diffusion planner failed: {e}")
                # Continue with PDM on diffusion failure
                
        # Select trajectory using Leaky DDM
        selected_trajectory = self._select_trajectory_leaky_ddm(ego_speed_mps)
        
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
            progress_raw = float(self._scorer._progress_raw[0])  # Raw progress in meters
            
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
                'progress': progress_normalized,  
                'progress_raw': progress_raw,  # Raw progress in meters for trapped detection
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
        
    def _select_trajectory_leaky_ddm(self, ego_speed_mps: float) -> AbstractTrajectory:
            """Select trajectory using Leaky DDM switcher."""
            if self.current_pdm_trajectory is None:
                raise RuntimeError("No PDM trajectory available!")
            
            # If no diffusion trajectory, use PDM
            if self.current_diffusion_trajectory is None:
                return self.current_pdm_trajectory.trajectory
            
            # Get scores and progress
            pdm_score = self.current_pdm_trajectory.score or 0.0
            diffusion_score = self.current_diffusion_trajectory.score or 0.0
            progress_m = self.current_pdm_trajectory.progress_m  # Use PDM's progress for trapped detection
            
            # Check if diffusion was vetoed by safety (hard constraint violations)
            diffusion_is_valid = self.current_diffusion_trajectory.is_valid
            safety_vetoed = (
                diffusion_is_valid is False and
                self.current_diffusion_trajectory.score < 0.1
            )
            
            # Use Leaky DDM to select planner
            selected_planner, metadata = self._switcher.update_and_select(
                pdm_score=pdm_score,
                diffusion_score=diffusion_score,
                ego_speed_mps=ego_speed_mps,
                progress_m=progress_m,
                diffusion_is_valid=diffusion_is_valid,  # Pass validity flag
                safety_vetoed=safety_vetoed
            )
            
            # Log decision with SAH-style information
            logger.info(
                f"LeakyDDM: {selected_planner.value} | "
                f"P={metadata['P']:.3f} W={metadata['W']:.3f} DV={metadata['decision_value']:.3f} | "
                f"PDM={pdm_score:.3f}({metadata['pdm_bucket']}) "
                f"Diff={diffusion_score:.3f}({metadata['diffusion_bucket']}) | "
                f"ne={metadata['ne']} np={metadata['np']}"
            )
            
            # Return selected trajectory
            if selected_planner == PlannerType.PDM:
                return self.current_pdm_trajectory.trajectory
            else:
                return self.current_diffusion_trajectory.trajectory