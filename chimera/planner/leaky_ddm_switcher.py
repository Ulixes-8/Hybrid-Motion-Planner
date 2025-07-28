"""
Version 4: Decisive Leaky DDM Switcher
Optimized for superior performance through aggressive scenario-based switching
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PlannerType(Enum):
    """Planner type enumeration"""
    PDM = "pdm"
    DIFFUSION = "diffusion"


class ScenarioType(Enum):
    """Scenario types with empirical performance data"""
    # Strong PDM scenarios (> 0.85 performance)
    STOPPING_WITH_LEAD = "stopping_with_lead"  # PDM: 0.984
    FOLLOWING_LANE_WITH_LEAD = "following_lane_with_lead"  # PDM: 0.939
    BEHIND_LONG_VEHICLE = "behind_long_vehicle"  # PDM: 0.926
    HIGH_MAGNITUDE_SPEED = "high_magnitude_speed"  # PDM: 0.919
    WAITING_FOR_PEDESTRIAN = "waiting_for_pedestrian_to_cross"  # PDM: 0.882
    STATIONARY_IN_TRAFFIC = "stationary_in_traffic"  # PDM: 0.868
    
    # Moderate PDM scenarios
    STARTING_RIGHT_TURN = "starting_right_turn"  # PDM: 0.693
    NEAR_MULTIPLE_VEHICLES = "near_multiple_vehicles"  # PDM: 0.659
    
    # Strong Diffusion scenarios
    STARTING_STRAIGHT_INTERSECTION = "starting_straight_traffic_light_intersection_traversal"  # Diff: 0.900
    CHANGING_LANE = "changing_lane"  # Diff: 0.767
    TRAVERSING_PICKUP_DROPOFF = "traversing_pickup_dropoff"  # Diff: 0.685
    STARTING_LEFT_TURN = "starting_left_turn"  # Diff: 0.670
    HIGH_LATERAL_ACCELERATION = "high_lateral_acceleration"  # Diff: 0.573
    LOW_MAGNITUDE_SPEED = "low_magnitude_speed"  # Diff: 0.480
    
    UNKNOWN = "unknown"


@dataclass
class LeakyDDMConfig:
    """Version 4 configuration for decisive switching"""
    # Core DDM
    alpha: float = 0.7  # Lower memory for faster response
    theta_upper: float = 0.3  # Lower threshold
    theta_lower: float = -0.3  # Symmetric
    
    # Evidence processing
    k_gain: float = 0.05  # Reduced reliance on scores
    score_scale: float = 1.0
    
    # Scenario detection
    poor_threshold: float = 0.4
    excellent_threshold: float = 0.8
    trapped_cycles: int = 2
    routine_cycles: int = 3
    
    # Biases
    trapped_bias: float = 0.3
    routine_bias: float = -0.15
    progress_threshold: float = 0.15
    veto_penalty: float = -0.2
    
    # DECISIVE parameters
    scenario_scale: float = 1.0  # Maximum influence
    scenario_confidence_threshold: float = 0.5  # Lower threshold
    scenario_decay: float = 0.95  # Slow decay
    
    # Version 4 specific
    decisive_mode: bool = True
    scenario_override_threshold: float = 0.7
    min_scenario_persistence: int = 5


class LeakyDDMSwitcher:
    """
    Version 4: Decisive switcher for superior performance
    """
    
    # Scenario to planner mapping
    SCENARIO_OPTIMAL_PLANNER = {
        # Strong PDM
        ScenarioType.STOPPING_WITH_LEAD: PlannerType.PDM,
        ScenarioType.FOLLOWING_LANE_WITH_LEAD: PlannerType.PDM,
        ScenarioType.BEHIND_LONG_VEHICLE: PlannerType.PDM,
        ScenarioType.HIGH_MAGNITUDE_SPEED: PlannerType.PDM,
        ScenarioType.WAITING_FOR_PEDESTRIAN: PlannerType.PDM,
        ScenarioType.STATIONARY_IN_TRAFFIC: PlannerType.PDM,
        ScenarioType.STARTING_RIGHT_TURN: PlannerType.PDM,
        ScenarioType.NEAR_MULTIPLE_VEHICLES: PlannerType.PDM,
        
        # Strong Diffusion
        ScenarioType.STARTING_STRAIGHT_INTERSECTION: PlannerType.DIFFUSION,
        ScenarioType.CHANGING_LANE: PlannerType.DIFFUSION,
        ScenarioType.TRAVERSING_PICKUP_DROPOFF: PlannerType.DIFFUSION,
        ScenarioType.STARTING_LEFT_TURN: PlannerType.DIFFUSION,
        ScenarioType.HIGH_LATERAL_ACCELERATION: PlannerType.DIFFUSION,
        ScenarioType.LOW_MAGNITUDE_SPEED: PlannerType.DIFFUSION,
        
        ScenarioType.UNKNOWN: PlannerType.PDM  # Default to PDM
    }
    
    # Scenario bias strengths (stronger for clearer winners)
    SCENARIO_BIASES = {
        # Very strong PDM (> 0.9 performance)
        ScenarioType.STOPPING_WITH_LEAD: -1.0,
        ScenarioType.FOLLOWING_LANE_WITH_LEAD: -0.9,
        ScenarioType.BEHIND_LONG_VEHICLE: -0.9,
        ScenarioType.HIGH_MAGNITUDE_SPEED: -0.8,
        
        # Strong PDM
        ScenarioType.WAITING_FOR_PEDESTRIAN: -0.7,
        ScenarioType.STATIONARY_IN_TRAFFIC: -0.6,
        
        # Moderate PDM
        ScenarioType.STARTING_RIGHT_TURN: -0.3,
        ScenarioType.NEAR_MULTIPLE_VEHICLES: -0.3,
        
        # Very strong Diffusion
        ScenarioType.STARTING_STRAIGHT_INTERSECTION: 1.0,
        ScenarioType.CHANGING_LANE: 0.7,
        ScenarioType.TRAVERSING_PICKUP_DROPOFF: 0.6,
        ScenarioType.STARTING_LEFT_TURN: 0.5,
        
        # Moderate Diffusion
        ScenarioType.HIGH_LATERAL_ACCELERATION: 0.3,
        ScenarioType.LOW_MAGNITUDE_SPEED: 0.2,
        
        ScenarioType.UNKNOWN: 0.0
    }
    
    def __init__(self, config: LeakyDDMConfig = None):
        """Initialize Version 4 switcher"""
        self.config = config or LeakyDDMConfig()
        
        # Core state
        self.P = 0.0
        self.current_planner = PlannerType.PDM
        
        # Tracking
        self.last_scenario = ScenarioType.UNKNOWN
        self.scenario_planner_cycles = 0
        self.switch_count = 0
        self.cycles_since_switch = 100
        
        # Performance tracking
        self.consecutive_poor_progress = 0
        self.consecutive_excellent_pdm = 0
        
        logger.info(
            f"[V4] Initialized Decisive LeakyDDM: "
            f"α={self.config.alpha}, θ=±{self.config.theta_upper}, "
            f"scenario_scale={self.config.scenario_scale}"
        )
    
    def update_and_select(
        self,
        pdm_score: float,
        diffusion_score: float,
        pdm_progress: Optional[float] = None,
        safety_vetoed: bool = False,
        scenario_type: Optional[ScenarioType] = None,
        scenario_confidence: Optional[float] = None,
        scenario_persistence: Optional[int] = None
    ) -> Tuple[PlannerType, Dict]:
        """
        Version 4: Decisive selection based on scenarios
        """
        self.cycles_since_switch += 1
        
        # Update performance tracking
        self._update_performance_tracking(pdm_score, pdm_progress)
        
        # DECISIVE MODE: Strong scenario override
        if (self.config.decisive_mode and 
            scenario_type is not None and 
            scenario_confidence is not None and
            scenario_confidence >= self.config.scenario_override_threshold):
            
            # Get optimal planner for scenario
            optimal_planner = self.SCENARIO_OPTIMAL_PLANNER[scenario_type]
            
            # If we're not using the optimal planner and scenario is persistent
            if (self.current_planner != optimal_planner and
                (scenario_persistence is None or scenario_persistence >= self.config.min_scenario_persistence)):
                
                # Override and switch immediately
                logger.info(
                    f"[V4] DECISIVE OVERRIDE: {scenario_type.value} "
                    f"requires {optimal_planner.value} (conf={scenario_confidence:.2f})"
                )
                
                self.current_planner = optimal_planner
                self.switch_count += 1
                self.cycles_since_switch = 0
                self.P = 0.3 if optimal_planner == PlannerType.DIFFUSION else -0.3
                
                return optimal_planner, self._create_metadata(
                    pdm_score, diffusion_score, scenario_type, scenario_confidence,
                    decisive_override=True
                )
        
        # Standard DDM update with strong scenario influence
        if scenario_type is not None and scenario_confidence is not None:
            if scenario_confidence >= self.config.scenario_confidence_threshold:
                # Apply scenario bias
                scenario_bias = self.SCENARIO_BIASES[scenario_type]
                scenario_drift = scenario_bias * self.config.scenario_scale * scenario_confidence
                
                # Track scenario persistence
                if scenario_type != self.last_scenario:
                    self.scenario_planner_cycles = 0
                    self.last_scenario = scenario_type
                else:
                    self.scenario_planner_cycles += 1
            else:
                scenario_drift = 0.0
        else:
            scenario_drift = 0.0
        
        # Score-based drift (secondary)
        score_diff = diffusion_score - pdm_score
        score_drift = (1 - self.config.alpha) * self.config.k_gain * score_diff
        
        # Apply trapped/routine biases
        if self.consecutive_poor_progress >= self.config.trapped_cycles:
            self.P += self.config.trapped_bias
            self.consecutive_poor_progress = 0
            logger.debug(f"[V4] Trapped bias applied: +{self.config.trapped_bias}")
        
        if self.consecutive_excellent_pdm >= self.config.routine_cycles:
            self.P += self.config.routine_bias
            self.consecutive_excellent_pdm = 0
            logger.debug(f"[V4] Routine bias applied: {self.config.routine_bias}")
        
        # Update preference accumulator
        self.P = np.clip(
            self.config.alpha * self.P + score_drift + scenario_drift,
            -1.0, 1.0
        )
        
        # Make decision
        new_planner = self._make_decision()
        
        # Handle switching
        if new_planner != self.current_planner:
            self.switch_count += 1
            self.cycles_since_switch = 0
            
            logger.info(
                f"[V4] SWITCH {self.switch_count}: "
                f"{self.current_planner.value} → {new_planner.value} | "
                f"P={self.P:.3f}, Scenario: {scenario_type.value if scenario_type else 'unknown'}"
            )
            
            self.current_planner = new_planner
            self.consecutive_poor_progress = 0
            self.consecutive_excellent_pdm = 0
        
        return new_planner, self._create_metadata(
            pdm_score, diffusion_score, scenario_type, scenario_confidence
        )
    
    def _update_performance_tracking(self, pdm_score: float, pdm_progress: Optional[float]):
        """Update performance counters"""
        # Poor performance detection
        if (pdm_score < self.config.poor_threshold or 
            (pdm_progress is not None and pdm_progress < self.config.progress_threshold)):
            self.consecutive_poor_progress += 1
        else:
            self.consecutive_poor_progress = 0
        
        # Excellent performance detection
        if pdm_score >= self.config.excellent_threshold:
            self.consecutive_excellent_pdm += 1
        else:
            self.consecutive_excellent_pdm = 0
    
    def _make_decision(self) -> PlannerType:
        """Make decision with hysteresis"""
        if self.current_planner == PlannerType.PDM:
            if self.P > self.config.theta_upper:
                return PlannerType.DIFFUSION
        else:
            if self.P < self.config.theta_lower:
                return PlannerType.PDM
        
        return self.current_planner
    
    def _create_metadata(
        self, 
        pdm_score: float,
        diffusion_score: float,
        scenario_type: Optional[ScenarioType],
        scenario_confidence: Optional[float],
        **flags
    ) -> Dict:
        """Create metadata for logging"""
        optimal_planner = None
        if scenario_type:
            optimal_planner = self.SCENARIO_OPTIMAL_PLANNER.get(scenario_type)
        
        return {
            'P': self.P,
            'pdm_score': pdm_score,
            'diffusion_score': diffusion_score,
            'scenario': scenario_type.value if scenario_type else 'unknown',
            'scenario_confidence': scenario_confidence or 0.0,
            'planner': self.current_planner.value,
            'optimal_planner': optimal_planner.value if optimal_planner else 'unknown',
            'switch_count': self.switch_count,
            'cycles_since_switch': self.cycles_since_switch,
            **flags
        }
    
    def reset(self):
        """Reset switcher"""
        self.P = 0.0
        self.current_planner = PlannerType.PDM
        self.last_scenario = ScenarioType.UNKNOWN
        self.scenario_planner_cycles = 0
        self.switch_count = 0
        self.cycles_since_switch = 100
        self.consecutive_poor_progress = 0
        self.consecutive_excellent_pdm = 0
        logger.info("[V4] Decisive LeakyDDM switcher reset")