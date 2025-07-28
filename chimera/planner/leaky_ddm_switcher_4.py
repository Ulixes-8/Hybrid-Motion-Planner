"""
Oracle DDM Switcher - Version 4
Achieves near-optimal switching by strongly prioritizing empirical performance data.
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
    """Scenario types with empirical winners"""
    # PDM WINS (8 types)
    BEHIND_LONG_VEHICLE = "behind_long_vehicle"
    FOLLOWING_LANE_WITH_LEAD = "following_lane_with_lead"
    HIGH_MAGNITUDE_SPEED = "high_magnitude_speed"
    NEAR_MULTIPLE_VEHICLES = "near_multiple_vehicles"
    STARTING_RIGHT_TURN = "starting_right_turn"
    STATIONARY_IN_TRAFFIC = "stationary_in_traffic"
    STOPPING_WITH_LEAD = "stopping_with_lead"
    WAITING_FOR_PEDESTRIAN = "waiting_for_pedestrian_to_cross"
    
    # DIFFUSION WINS (6 types)
    CHANGING_LANE = "changing_lane"
    HIGH_LATERAL_ACCELERATION = "high_lateral_acceleration"
    LOW_MAGNITUDE_SPEED = "low_magnitude_speed"
    STARTING_LEFT_TURN = "starting_left_turn"
    STARTING_STRAIGHT_INTERSECTION = "starting_straight_traffic_light_intersection_traversal"
    TRAVERSING_PICKUP_DROPOFF = "traversing_pickup_dropoff"
    
    UNKNOWN = "unknown"


@dataclass
class OracleDDMConfig:
    """Configuration for Oracle DDM"""
    # Core parameters
    alpha: float = 0.3  # Low memory for fast adaptation
    theta_upper: float = 0.15  # Low threshold for quick switching
    theta_lower: float = -0.15
    
    # Weighting
    scenario_weight: float = 0.85  # Scenario dominates
    score_weight: float = 0.15  # Score is secondary
    
    # Confidence
    min_confidence: float = 0.4
    high_confidence: float = 0.7
    
    # Override for large performance gaps
    override_threshold: float = 0.3  # 30% win rate difference
    
    # Safety
    min_switch_interval: int = 1  # Very low


class OracleDDMSwitcher:
    """
    Oracle-like switcher that uses empirical performance data.
    """
    
    # Empirical win rates from your data
    EMPIRICAL_WIN_RATES = {
        # PDM win rates
        ScenarioType.STOPPING_WITH_LEAD: {'pdm': 0.984, 'diffusion': 0.976},
        ScenarioType.FOLLOWING_LANE_WITH_LEAD: {'pdm': 0.939, 'diffusion': 0.797},
        ScenarioType.BEHIND_LONG_VEHICLE: {'pdm': 0.926, 'diffusion': 0.846},
        ScenarioType.HIGH_MAGNITUDE_SPEED: {'pdm': 0.919, 'diffusion': 0.783},
        ScenarioType.WAITING_FOR_PEDESTRIAN: {'pdm': 0.882, 'diffusion': 0.494},
        ScenarioType.STATIONARY_IN_TRAFFIC: {'pdm': 0.868, 'diffusion': 0.678},
        ScenarioType.STARTING_RIGHT_TURN: {'pdm': 0.693, 'diffusion': 0.577},
        ScenarioType.NEAR_MULTIPLE_VEHICLES: {'pdm': 0.659, 'diffusion': 0.505},
        
        # Diffusion win rates
        ScenarioType.STARTING_STRAIGHT_INTERSECTION: {'pdm': 0.874, 'diffusion': 0.900},
        ScenarioType.CHANGING_LANE: {'pdm': 0.730, 'diffusion': 0.767},
        ScenarioType.TRAVERSING_PICKUP_DROPOFF: {'pdm': 0.505, 'diffusion': 0.685},
        ScenarioType.STARTING_LEFT_TURN: {'pdm': 0.640, 'diffusion': 0.670},
        ScenarioType.HIGH_LATERAL_ACCELERATION: {'pdm': 0.547, 'diffusion': 0.573},
        ScenarioType.LOW_MAGNITUDE_SPEED: {'pdm': 0.432, 'diffusion': 0.480},
        
        ScenarioType.UNKNOWN: {'pdm': 0.5, 'diffusion': 0.5}
    }
    
    def __init__(self, config: OracleDDMConfig = None):
        """Initialize oracle switcher"""
        self.config = config or OracleDDMConfig()
        
        # State
        self.P = 0.0  # Preference accumulator
        self.current_planner = PlannerType.PDM
        self.cycles_since_switch = 100
        self.total_switches = 0
        
        # Tracking for performance
        self.scenario_history = []
        self.correct_decisions = 0
        self.total_decisions = 0
        
        logger.info(
            f"Initialized Oracle DDM: "
            f"scenario_weight={self.config.scenario_weight}, "
            f"score_weight={self.config.score_weight}"
        )
    
    def update_and_select(
        self,
        pdm_score: float,
        diffusion_score: float,
        scenario_result: Optional['ScenarioDetectionResult'] = None
    ) -> Tuple[PlannerType, Dict]:
        """
        Select planner using oracle-like logic.
        """
        self.cycles_since_switch += 1
        self.total_decisions += 1
        
        # Extract scenario info
        scenario_type = ScenarioType.UNKNOWN
        scenario_confidence = 0.0
        
        if scenario_result is not None:
            scenario_type = scenario_result.scenario_type
            scenario_confidence = scenario_result.confidence
        
        # Get empirical win rates
        win_rates = self.EMPIRICAL_WIN_RATES.get(scenario_type, {'pdm': 0.5, 'diffusion': 0.5})
        pdm_win_rate = win_rates['pdm']
        diffusion_win_rate = win_rates['diffusion']
        win_rate_gap = abs(pdm_win_rate - diffusion_win_rate)
        
        # Determine empirically best planner
        empirical_best = PlannerType.PDM if pdm_win_rate > diffusion_win_rate else PlannerType.DIFFUSION
        
        # SCENARIO-BASED DRIFT (PRIMARY)
        scenario_drift = 0.0
        if scenario_confidence >= self.config.min_confidence:
            # Strong bias towards empirically better planner
            if empirical_best == PlannerType.DIFFUSION:
                scenario_bias = 1.0
            else:
                scenario_bias = -1.0
            
            # Scale by confidence and win rate gap
            scenario_strength = min(1.0, win_rate_gap * 2.0)  # Amplify gaps
            scenario_drift = (scenario_bias * scenario_strength * 
                            self.config.scenario_weight * scenario_confidence)
            
            # IMMEDIATE OVERRIDE for high-confidence, large-gap scenarios
            if (scenario_confidence >= self.config.high_confidence and 
                win_rate_gap >= self.config.override_threshold and
                self.current_planner != empirical_best):
                
                # Force immediate switch
                logger.info(
                    f"OVERRIDE: {scenario_type.value} demands {empirical_best.value} "
                    f"(gap={win_rate_gap:.2f}, conf={scenario_confidence:.2f})"
                )
                
                self.current_planner = empirical_best
                self.cycles_since_switch = 0
                self.total_switches += 1
                self._track_decision(empirical_best, empirical_best)
                
                return empirical_best, self._create_metadata(
                    pdm_score, diffusion_score, scenario_type, scenario_confidence,
                    win_rates, empirical_best, override=True
                )
        
        # SCORE-BASED DRIFT (SECONDARY)
        score_diff = diffusion_score - pdm_score
        score_drift = self.config.score_weight * np.tanh(score_diff * 10.0)  # Stronger response
        
        # Update preference with low memory
        self.P = self.config.alpha * self.P + scenario_drift + score_drift
        self.P = np.clip(self.P, -1.0, 1.0)
        
        # Check if we can switch (minimal lockout)
        if self.cycles_since_switch < self.config.min_switch_interval:
            self._track_decision(self.current_planner, empirical_best)
            return self.current_planner, self._create_metadata(
                pdm_score, diffusion_score, scenario_type, scenario_confidence,
                win_rates, empirical_best, lockout=True
            )
        
        # Make decision
        if self.current_planner == PlannerType.PDM:
            if self.P > self.config.theta_upper:
                new_planner = PlannerType.DIFFUSION
            else:
                new_planner = PlannerType.PDM
        else:
            if self.P < self.config.theta_lower:
                new_planner = PlannerType.PDM
            else:
                new_planner = PlannerType.DIFFUSION
        
        # Switch if needed
        if new_planner != self.current_planner:
            logger.info(
                f"SWITCH: {self.current_planner.value} → {new_planner.value} | "
                f"Scenario: {scenario_type.value} | P={self.P:.3f}"
            )
            self.current_planner = new_planner
            self.cycles_since_switch = 0
            self.total_switches += 1
        
        self._track_decision(self.current_planner, empirical_best)
        
        return self.current_planner, self._create_metadata(
            pdm_score, diffusion_score, scenario_type, scenario_confidence,
            win_rates, empirical_best
        )
    
    def _track_decision(self, selected: PlannerType, empirical_best: PlannerType):
        """Track decision accuracy"""
        if selected == empirical_best:
            self.correct_decisions += 1
    
    def _create_metadata(
        self,
        pdm_score: float,
        diffusion_score: float,
        scenario_type: ScenarioType,
        scenario_confidence: float,
        win_rates: Dict,
        empirical_best: PlannerType,
        **flags
    ) -> Dict:
        """Create metadata"""
        accuracy = (self.correct_decisions / self.total_decisions * 100 
                   if self.total_decisions > 0 else 0)
        
        return {
            'P': self.P,
            'pdm_score': pdm_score,
            'diffusion_score': diffusion_score,
            'scenario': scenario_type.value,
            'scenario_confidence': scenario_confidence,
            'planner': self.current_planner.value,
            'expected_planner': empirical_best.value,
            'pdm_win_rate': win_rates.get('pdm', 0.5),
            'diffusion_win_rate': win_rates.get('diffusion', 0.5),
            'cycles_since_switch': self.cycles_since_switch,
            'total_switches': self.total_switches,
            'decision_accuracy': accuracy,
            **flags
        }
    
    def reset(self):
        """Reset switcher"""
        self.P = 0.0
        self.current_planner = PlannerType.PDM
        self.cycles_since_switch = 100
        self.total_switches = 0
        self.correct_decisions = 0
        self.total_decisions = 0
        logger.info("Oracle DDM reset")