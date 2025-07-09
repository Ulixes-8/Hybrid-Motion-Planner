import logging
from typing import Dict, Optional, TYPE_CHECKING, Tuple
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.ego_state import EgoState

# Type checking imports
if TYPE_CHECKING:
    from nuplan.common.maps.abstract_map import AbstractMap
    from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInput

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryScore:
    """Container for trajectory scores"""
    total_score: float
    is_valid: bool
    collision_score: float
    progress_score: float
    comfort_score: float
    ttc_score: float
    drivable_area_score: float
    driving_direction_score: float
    time_to_collision_s: float
    time_to_at_fault_collision_s: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            'total': self.total_score,
            'valid': float(self.is_valid),
            'collision': self.collision_score,
            'progress': self.progress_score,
            'comfort': self.comfort_score,
            'ttc': self.ttc_score,
            'drivable_area': self.drivable_area_score,
            'driving_direction': self.driving_direction_score,
            'ttc_time': self.time_to_collision_s,
            'collision_time': self.time_to_at_fault_collision_s
        }


class PDMBasedTrajectoryScorer:
    """
    Scores trajectories using PDM's exact methodology.
    This properly simulates trajectories before scoring them.
    """
    
    def __init__(self, proposal_sampling: TrajectorySampling, trajectory_sampling: TrajectorySampling):
        """
        Initialize the PDM-based scorer.
        
        :param proposal_sampling: Sampling for proposals (4s @ 10Hz as per PDM)
        :param trajectory_sampling: Sampling for full trajectory (8s @ 10Hz)
        """
        self.proposal_sampling = proposal_sampling
        self.trajectory_sampling = trajectory_sampling
        self._pdm_scorer = None
        self._simulator = None
        self._initialized = False
        
    def initialize(self):
        """Lazy initialization of PDM components"""
        if not self._initialized:
            # Import here to avoid circular dependencies
            from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
            from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
            
            self._pdm_scorer = PDMScorer(proposal_sampling=self.proposal_sampling)
            self._simulator = PDMSimulator(proposal_sampling=self.proposal_sampling)
            self._initialized = True
    
    def score_trajectory(
        self,
        trajectory: AbstractTrajectory,
        current_ego_state: EgoState,
        observation,
        centerline,
        route_lane_dict: Dict[str, any],
        drivable_area_map,
        map_api,
        planner_name: str
    ) -> TrajectoryScore:
        """
        Score a trajectory using PDM's exact methodology.
        
        This includes:
        1. Converting trajectory to states
        2. Simulating the trajectory through motion model
        3. Scoring using PDM's scorer
        
        :param trajectory: Trajectory to score
        :param current_ego_state: Current ego state
        :param observation: PDM observation
        :param centerline: Route centerline
        :param route_lane_dict: Route lanes dictionary
        :param drivable_area_map: Drivable area map
        :param map_api: Map API
        :param planner_name: Name of planner (for logging)
        :return: Trajectory score
        """
        try:
            # Ensure components are initialized
            self.initialize()
            
            # Convert trajectory to state array
            states = self._trajectory_to_state_array(trajectory, current_ego_state)
            
            # CRITICAL: Simulate the trajectory through motion model
            # This is what PDM does before scoring!
            simulated_states = self._simulator.simulate_proposals(states, current_ego_state)
            
            # Score using PDM's scorer
            scores = self._pdm_scorer.score_proposals(
                states=simulated_states,
                initial_ego_state=current_ego_state,
                observation=observation,
                centerline=centerline,
                route_lane_dict=route_lane_dict,
                drivable_area_map=drivable_area_map,
                map_api=map_api
            )
            
            # Extract score (we only have one trajectory)
            total_score = float(scores[0])
            
            # Get time to infractions
            time_to_collision = self._pdm_scorer.time_to_at_fault_collision(0)
            time_to_ttc = self._pdm_scorer.time_to_ttc_infraction(0)
            
            # Extract detailed metrics from PDM scorer
            # MultiMetricIndex: NO_COLLISION=0, DRIVABLE_AREA=1, DRIVING_DIRECTION=2
            collision_score = float(self._pdm_scorer._multi_metrics[0, 0])
            drivable_area_score = float(self._pdm_scorer._multi_metrics[1, 0])
            driving_direction_score = float(self._pdm_scorer._multi_metrics[2, 0])
            
            # WeightedMetricIndex: PROGRESS=0, TTC=1, COMFORTABLE=2
            progress_score = float(self._pdm_scorer._weighted_metrics[0, 0])
            ttc_score = float(self._pdm_scorer._weighted_metrics[1, 0])
            comfort_score = float(self._pdm_scorer._weighted_metrics[2, 0])
            
            # Check validity (all multiplicative metrics must pass)
            is_valid = (
                collision_score > 0.5 and 
                drivable_area_score > 0.5 and 
                driving_direction_score > 0.5
            )
            
            score = TrajectoryScore(
                total_score=total_score,
                is_valid=is_valid,
                collision_score=collision_score,
                progress_score=progress_score,
                comfort_score=comfort_score,
                ttc_score=ttc_score,
                drivable_area_score=drivable_area_score,
                driving_direction_score=driving_direction_score,
                time_to_collision_s=time_to_ttc,
                time_to_at_fault_collision_s=time_to_collision
            )
            
            logger.debug(f"{planner_name} trajectory score: {score.to_dict()}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error scoring {planner_name} trajectory: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a zero score on error
            return TrajectoryScore(
                total_score=0.0,
                is_valid=False,
                collision_score=0.0,
                progress_score=0.0,
                comfort_score=0.0,
                ttc_score=0.0,
                drivable_area_score=0.0,
                driving_direction_score=0.0,
                time_to_collision_s=float('inf'),
                time_to_at_fault_collision_s=float('inf')
            )
    
    def _trajectory_to_state_array(self, trajectory: AbstractTrajectory, initial_ego_state: EgoState) -> npt.NDArray[np.float64]:
        """
        Convert abstract trajectory to state array format expected by PDM.
        
        :param trajectory: Abstract trajectory
        :param initial_ego_state: Initial ego state
        :return: State array of shape (1, num_poses+1, state_dim)
        """
        # Import here to avoid circular dependencies
        from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
        
        # Sample trajectory
        sampled_states = trajectory.get_sampled_trajectory()
        
        # Determine number of states
        num_states = min(len(sampled_states), self.proposal_sampling.num_poses + 1)
        
        # Initialize state array
        states = np.zeros((1, num_states, StateIndex.size()), dtype=np.float64)
        
        # Fill state array
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
                states[0, i, StateIndex.STEERING_RATE] = 0.0  # Not available, set to 0
                
                # Angular motion
                states[0, i, StateIndex.ANGULAR_VELOCITY] = state.dynamic_car_state.angular_velocity
                states[0, i, StateIndex.ANGULAR_ACCELERATION] = state.dynamic_car_state.angular_acceleration
            else:
                # Pad with last state if trajectory is shorter
                states[0, i] = states[0, i-1]
        
        return states