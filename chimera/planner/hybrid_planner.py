import logging
import time
from typing import List, Type, Optional, Dict
from dataclasses import dataclass
import traceback

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, 
    PlannerInitialization, 
    PlannerInput
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# Import our scoring component
from chimera.scoring.unified_trajectory_scorer import UnifiedTrajectoryScorer

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryWithMetadata:
    """Container for trajectory with additional metadata."""
    trajectory: AbstractTrajectory
    planner_name: str
    computation_time: float
    timestamp: float


class HybridPlanner(AbstractPlanner):
    """
    Hybrid planner that intelligently switches between PDM-Closed and Diffusion Planner.
    """
    
    # Inherited property
    requires_scenario: bool = False
    
    def __init__(
        self,
        pdm_planner: AbstractPlanner,
        diffusion_planner: AbstractPlanner,
        pdm_frequency: float = 10.0,  # Hz
        diffusion_frequency: float = 2.0,  # Hz
    ):
        """
        Initialize the hybrid planner.
        
        :param pdm_planner: Instance of PDM-Closed planner
        :param diffusion_planner: Instance of Diffusion planner
        :param pdm_frequency: Frequency for PDM planner execution (Hz)
        :param diffusion_frequency: Frequency for Diffusion planner execution (Hz)
        """
        self.pdm_planner = pdm_planner
        self.diffusion_planner = diffusion_planner  # Use directly, no wrapper
        
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
        self._current_planner = 'pdm'  # Start with PDM
        self._start_time = None
        
        # Scoring components
        self.scorer = None
        
        # PDM observation components (needed for scoring)
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

        logger.info(f"Initialized HybridPlanner with PDM@{pdm_frequency}Hz, Diffusion@{diffusion_frequency}Hz")
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize both planners."""
        logger.info("Initializing hybrid planner components...")
        
        # Store map API
        self._map_api = initialization.map_api
        
        # Initialize scoring with proper PDM parameters
        # PDM uses 4s @ 10Hz for proposals as per the paper
        proposal_sampling = TrajectorySampling(
            num_poses=40,  # 4s @ 10Hz
            interval_length=0.1
        )
        self.scorer = UnifiedTrajectoryScorer(
            proposal_sampling=proposal_sampling,
            trajectory_sampling=self._trajectory_sampling
        )
        
        # Initialize route information
        self._initialize_route_info(initialization.route_roadblock_ids)
        
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
        # Import here to avoid circular dependencies
        from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import StateSE2
        
        self._route_lane_dict = {}
        centerline_discrete_path = []
        
        for id_ in route_roadblock_ids:
            try:
                # Get the roadblock
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
            # Convert to StateSE2 list
            states_se2 = [
                StateSE2(point.x, point.y, point.heading) 
                for point in centerline_discrete_path
            ]
            self._centerline = PDMPath(states_se2)
            logger.info(f"Created centerline with {len(states_se2)} points")
        else:
            logger.warning("No centerline path could be created from route")
            # Create a dummy centerline to avoid None errors
            dummy_states = [
                StateSE2(0, 0, 0),
                StateSE2(10, 0, 0),
                StateSE2(20, 0, 0)
            ]
            self._centerline = PDMPath(dummy_states)
    
    def name(self) -> str:
        """Return planner name."""
        return "HybridPlanner"
    
    def observation_type(self) -> Type[Observation]:
        """Return observation type - using DetectionsTracks for both planners."""
        return DetectionsTracks
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Main planning function with dual-frequency execution.
        PDM runs at high frequency, Diffusion at low frequency.
        """
        # Store current input for scoring
        self._current_planner_input = current_input
        
        # Get current time relative to start
        current_time = time.time() - self._start_time if self._start_time else 0.0
        
        # Update observation and drivable area for scoring
        ego_state, observation = current_input.history.current_state
        
        # Create PDM observation if needed
        if self._observation is None:
            # Import here to avoid circular dependencies
            from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
            
            self._observation = PDMObservation(
                trajectory_sampling=self._trajectory_sampling,
                proposal_sampling=self._proposal_sampling,
                map_radius=50.0
            )
        
        # Update observation
        self._observation.update(
            ego_state=ego_state,
            observation=observation,
            traffic_light_data=current_input.traffic_light_data,
            route_lane_dict=self._route_lane_dict
        )
        
        # Update drivable area map
        from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import get_drivable_area_map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, map_radius=50.0
        )
        
        # Check if we should run PDM planner (always run on first iteration)
        if self._iteration == 0 or current_time - self.last_pdm_time >= self.pdm_period:
            start = time.time()
            try:
                trajectory = self.pdm_planner.compute_planner_trajectory(current_input)
                computation_time = time.time() - start
                
                self.current_pdm_trajectory = TrajectoryWithMetadata(
                    trajectory=trajectory,
                    planner_name='pdm',
                    computation_time=computation_time,
                    timestamp=current_time
                )
                self.last_pdm_time = current_time
                
                logger.debug(f"PDM trajectory computed in {computation_time:.3f}s")
            except Exception as e:
                logger.error(f"PDM planner failed: {e}")
                traceback.print_exc()
                raise
                
        # Check if we should run Diffusion planner
        if self._iteration == 0 or current_time - self.last_diffusion_time >= self.diffusion_period:
            start = time.time()
            try:
                # Debug: Let's see what's in the history buffer
                logger.debug(f"History buffer size: {len(current_input.history.ego_states)}")
                logger.debug(f"History buffer type: {type(current_input.history)}")
                
                # Debug the inputs to diffusion planner
                if hasattr(self.diffusion_planner, 'planner_input_to_model_inputs'):
                    try:
                        # Get the model inputs to see their shapes
                        model_inputs = self.diffusion_planner.planner_input_to_model_inputs(current_input)
                        logger.info("Diffusion planner model inputs:")
                        for key, value in model_inputs.items():
                            if hasattr(value, 'shape'):
                                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                            else:
                                logger.info(f"  {key}: type={type(value)}")
                                
                        # Check specifically the neighbors input
                        if 'neighbors' in model_inputs:
                            neighbors = model_inputs['neighbors']
                            logger.info(f"Neighbors tensor details: shape={neighbors.shape}, min={neighbors.min()}, max={neighbors.max()}")
                            # Check if it's all zeros or has actual data
                            logger.info(f"Neighbors has data: {neighbors.abs().sum() > 0}")

                        if 'neighbor_agents_past' in model_inputs:
                            neighbor_shape = model_inputs['neighbor_agents_past'].shape
                            logger.info(f"Neighbors shape details: {neighbor_shape}")
                            logger.info(f"Expected shape: [batch=1, agents=32, time=21, features=11]")
                            logger.info(f"Actual shape: [batch={neighbor_shape[0]}, agents={neighbor_shape[1]}, time={neighbor_shape[2]}, features={neighbor_shape[3]}]")
                            
                            if neighbor_shape[2] != 21:
                                logger.error(f"Time dimension mismatch! Expected 21, got {neighbor_shape[2]}")
                                logger.error("Diffusion planner needs 2 seconds of history (21 steps at 0.1s)")
                            
                    except Exception as e:
                        logger.error(f"Error getting model inputs: {e}")
                        traceback.print_exc()
                
                # Now try the actual computation
                trajectory = self.diffusion_planner.compute_planner_trajectory(current_input)
                computation_time = time.time() - start
                
                self.current_diffusion_trajectory = TrajectoryWithMetadata(
                    trajectory=trajectory,
                    planner_name='diffusion',
                    computation_time=computation_time,
                    timestamp=current_time
                )
                self.last_diffusion_time = current_time
                
                logger.info(f"Diffusion trajectory computed successfully in {computation_time:.3f}s")
            except Exception as e:
                logger.error(f"Diffusion planner failed with error: {type(e).__name__}: {str(e)}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                # Continue with PDM
                
        # Select best trajectory using scoring
        selected_trajectory = self._select_trajectory()
        
        self._iteration += 1
        logger.debug(f"Iteration {self._iteration}: Selected {self._current_planner} trajectory")
        
        return selected_trajectory
    
    def _select_trajectory(self) -> AbstractTrajectory:
        """
        Select which trajectory to use based on scoring.
        """
        if self.current_pdm_trajectory is None:
            raise RuntimeError("No PDM trajectory available!")
        
        # If no diffusion trajectory or scoring not available, use PDM
        if self.current_diffusion_trajectory is None or self.scorer is None:
            self._current_planner = 'pdm'
            return self.current_pdm_trajectory.trajectory
        
        # Get current ego state for scoring
        ego_state = self._current_planner_input.history.current_state[0]
        
        # Score both trajectories
        pdm_score = self.scorer.score_trajectory(
            trajectory=self.current_pdm_trajectory.trajectory,
            current_ego_state=ego_state,
            observation=self._observation,
            centerline=self._centerline,
            route_lane_dict=self._route_lane_dict,
            drivable_area_map=self._drivable_area_map,
            map_api=self._map_api,
            planner_name='pdm'
        )
        
        diffusion_score = self.scorer.score_trajectory(
            trajectory=self.current_diffusion_trajectory.trajectory,
            current_ego_state=ego_state,
            observation=self._observation,
            centerline=self._centerline,
            route_lane_dict=self._route_lane_dict,
            drivable_area_map=self._drivable_area_map,
            map_api=self._map_api,
            planner_name='diffusion'
        )
        
        # Log scores
        logger.info(
            f"Trajectory scores - PDM: {pdm_score.total_score:.3f} "
            f"(valid: {pdm_score.is_valid}), "
            f"Diffusion: {diffusion_score.total_score:.3f} "
            f"(valid: {diffusion_score.is_valid})"
        )
        
        # Log detailed scores
        logger.debug(f"PDM details: {pdm_score.to_dict()}")
        logger.debug(f"Diffusion details: {diffusion_score.to_dict()}")

                # Log detailed scores
        logger.info(f"PDM score details: {pdm_score.to_dict()}")
        logger.info(f"Diffusion score details: {diffusion_score.to_dict()}")
        
        # Selection logic with hysteresis
        SWITCHING_THRESHOLD = 1.1  # 10% improvement needed to switch
        
        # First check validity
        if not pdm_score.is_valid and not diffusion_score.is_valid:
            logger.warning("Both trajectories invalid! Using PDM as fallback")
            self._current_planner = 'pdm'
            return self.current_pdm_trajectory.trajectory
        
        if not diffusion_score.is_valid:
            self._current_planner = 'pdm'
            return self.current_pdm_trajectory.trajectory
            
        if not pdm_score.is_valid:
            self._current_planner = 'diffusion'
            return self.current_diffusion_trajectory.trajectory
        
        # Both valid - select based on score with hysteresis
        if self._current_planner == 'pdm':
            # Currently using PDM - switch if diffusion is significantly better
            if diffusion_score.total_score > pdm_score.total_score * SWITCHING_THRESHOLD:
                logger.info(f"Switching to Diffusion planner (score improvement: "
                          f"{(diffusion_score.total_score/pdm_score.total_score - 1)*100:.1f}%)")
                self._current_planner = 'diffusion'
                return self.current_diffusion_trajectory.trajectory
        else:
            # Currently using Diffusion - switch if PDM is significantly better
            if pdm_score.total_score > diffusion_score.total_score * SWITCHING_THRESHOLD:
                logger.info(f"Switching to PDM planner (score improvement: "
                          f"{(pdm_score.total_score/diffusion_score.total_score - 1)*100:.1f}%)")
                self._current_planner = 'pdm'
                return self.current_pdm_trajectory.trajectory
        
        # No switch - return current planner's trajectory
        if self._current_planner == 'pdm':
            return self.current_pdm_trajectory.trajectory
        else:
            return self.current_diffusion_trajectory.trajectory