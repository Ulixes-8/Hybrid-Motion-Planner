import logging
from typing import List, Type, Optional

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, 
    PlannerInitialization, 
    PlannerInput
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

logger = logging.getLogger(__name__)


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
        self.diffusion_planner = diffusion_planner
        
        # Timing control
        self.pdm_period = 1.0 / pdm_frequency
        self.diffusion_period = 1.0 / diffusion_frequency
        self.last_pdm_time = 0.0
        self.last_diffusion_time = 0.0
        
        # State tracking
        self._iteration = 0
        self._current_planner = 'pdm'  # Start with PDM
        
        logger.info(f"Initialized HybridPlanner with PDM@{pdm_frequency}Hz, Diffusion@{diffusion_frequency}Hz")
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize both planners."""
        logger.info("Initializing hybrid planner components...")
        self.pdm_planner.initialize(initialization)
        self.diffusion_planner.initialize(initialization)
        self._iteration = 0
    
    def name(self) -> str:
        """Return planner name."""
        return "HybridPlanner"
    
    def observation_type(self) -> Type[Observation]:
        """Return observation type - using DetectionsTracks for both planners."""
        return DetectionsTracks
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Main planning function - for now just uses PDM.
        We'll expand this with dual-frequency and switching logic next.
        """
        self._iteration += 1
        
        # For now, just use PDM planner
        trajectory = self.pdm_planner.compute_planner_trajectory(current_input)
        
        logger.debug(f"Iteration {self._iteration}: Using {self._current_planner} planner")
        
        return trajectory