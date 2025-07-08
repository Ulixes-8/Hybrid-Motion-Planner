import logging
import time
from typing import List, Type, Optional
from dataclasses import dataclass
import traceback

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, 
    PlannerInitialization, 
    PlannerInput
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

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
        
        logger.info(f"Initialized HybridPlanner with PDM@{pdm_frequency}Hz, Diffusion@{diffusion_frequency}Hz")
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize both planners."""
        logger.info("Initializing hybrid planner components...")
        
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
        # Get current time relative to start
        current_time = time.time() - self._start_time if self._start_time else 0.0
        
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
                
        # In compute_planner_trajectory method, replace the diffusion planner section:

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

                                                # In hybrid_planner.py, add this after logging the model inputs:
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
                
        # For now, always return PDM trajectory (we'll add selection logic next)
        selected_trajectory = self._select_trajectory()
        
        self._iteration += 1
        logger.debug(f"Iteration {self._iteration}: Selected {self._current_planner} trajectory")
        
        return selected_trajectory
    
    def _select_trajectory(self) -> AbstractTrajectory:
        """
        Select which trajectory to use.
        For now, just returns PDM trajectory. We'll expand this with scoring logic.
        """
        if self.current_pdm_trajectory is None:
            raise RuntimeError("No PDM trajectory available!")
        
        # Log available trajectories
        if self.current_diffusion_trajectory:
            logger.debug(
                f"Available trajectories: PDM (age: {time.time() - self._start_time - self.current_pdm_trajectory.timestamp:.3f}s), "
                f"Diffusion (age: {time.time() - self._start_time - self.current_diffusion_trajectory.timestamp:.3f}s)"
            )
        else:
            logger.debug("Only PDM trajectory available")
        
        # For now, always return PDM
        self._current_planner = 'pdm'
        return self.current_pdm_trajectory.trajectory