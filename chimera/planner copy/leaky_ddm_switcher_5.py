"""
Conservative Oracle Switcher - Version 5
Robust switching that always allows trajectory execution.
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
class ConservativeOracleConfig:
    """Configuration for Conservative Oracle"""
    # Memory and thresholds
    alpha: float = 0.7  # Higher memory
    theta_upper: float = 0.4
    theta_lower: float = -0.4
    
    # Weights
    scenario_weight: float = 0.6
    score_weight: float = 0.4
    
    # Confidence
    min_confidence: float = 0.6
    high_confidence: float = 0.85
    
    # Override threshold
    override_threshold: float = 0.4
    
    # Safety
    min_switch_interval: int = 3
    min_score_threshold: float = 0.0  # Accept any score
    score_difference_threshold: float = 0.05
    
    # Validation
    require_score_validation: bool = True
    fallback_to_pdm: bool = True


class ConservativeOracleSwitcher:
    """
    Conservative oracle switcher with validation and safety checks.
    """
    
    # Empirical win rates from data
    EMPIRICAL_WIN_RATES = {
        ScenarioType.STOPPING_WITH_LEAD: {'pdm': 0.984, 'diffusion': 0.976},
        ScenarioType.FOLLOWING_LANE_WITH_LEAD: {'pdm': 0.939, 'diffusion': 0.797},
        ScenarioType.BEHIND_LONG_VEHICLE: {'pdm': 0.926, 'diffusion': 0.846},
        ScenarioType.HIGH_MAGNITUDE_SPEED: {'pdm': 0.919, 'diffusion': 0.783},
        ScenarioType.WAITING_FOR_PEDESTRIAN: {'pdm': 0.882, 'diffusion': 0.494},
        ScenarioType.STATIONARY_IN_TRAFFIC: {'pdm': 0.868, 'diffusion': 0.678},
        ScenarioType.STARTING_RIGHT_TURN: {'pdm': 0.693, 'diffusion': 0.577},
        ScenarioType.NEAR_MULTIPLE_VEHICLES: {'pdm': 0.659, 'diffusion': 0.505},
        
        ScenarioType.STARTING_STRAIGHT_INTERSECTION: {'pdm': 0.874, 'diffusion': 0.900},
        ScenarioType.CHANGING_LANE: {'pdm': 0.730, 'diffusion': 0.767},
        ScenarioType.TRAVERSING_PICKUP_DROPOFF: {'pdm': 0.505, 'diffusion': 0.685},
        ScenarioType.STARTING_LEFT_TURN: {'pdm': 0.640, 'diffusion': 0.670},
        ScenarioType.HIGH_LATERAL_ACCELERATION: {'pdm': 0.547, 'diffusion': 0.573},
        ScenarioType.LOW_MAGNITUDE_SPEED: {'pdm': 0.432, 'diffusion': 0.480},
        
        ScenarioType.UNKNOWN: {'pdm': 0.5, 'diffusion': 0.5}
    }
    
    def __init__(self, config: ConservativeOracleConfig = None):
        """Initialize conservative switcher"""
        self.config = config or ConservativeOracleConfig()
        
        # State
        self.P = 0.0  # Neutral start
        self.current_planner = PlannerType.PDM  # Safe default
        self.cycles_since_switch = 100
        self.total_switches = 0
        
        # Performance tracking
        self.consecutive_poor_scores = 0
        self.scenario_consistency = 0
        self.last_scenario = ScenarioType.UNKNOWN
        
        # History
        self.decision_history = []
        
        logger.info(
            f"Initialized Conservative Oracle: "
            f"α={self.config.alpha}, "
            f"θ=±{self.config.theta_upper}"
        )
    
    def update_and_select(
        self,
        pdm_score: float,
        diffusion_score: float,
        pdm_safety: float,
        diffusion_safety: float,
        scenario_result: Optional['ScenarioDetectionResult'] = None,
        last_planner: Optional[PlannerType] = None
    ) -> Tuple[PlannerType, Dict]:
        """
        Select planner conservatively with validation.
        """
        self.cycles_since_switch += 1
        
        # Validate inputs
        pdm_score = max(0.0, min(1.0, pdm_score))
        diffusion_score = max(0.0, min(1.0, diffusion_score))
        pdm_safety = max(0.0, min(1.0, pdm_safety))
        diffusion_safety = max(0.0, min(1.0, diffusion_safety))
        
        # Extract scenario info
        scenario_type = ScenarioType.UNKNOWN
        scenario_confidence = 0.0
        
        if scenario_result is not None:
            scenario_type = scenario_result.scenario_type
            scenario_confidence = scenario_result.confidence
            
            # Track scenario consistency
            if scenario_type == self.last_scenario:
                self.scenario_consistency += 1
            else:
                self.scenario_consistency = 0
                self.last_scenario = scenario_type
        
        # SAFETY CHECK - only reject if BOTH scores are extremely low
        if pdm_score < 0.01 and diffusion_score < 0.01:
            logger.warning(f"Both scores extremely low: PDM={pdm_score:.3f}, Diff={diffusion_score:.3f}")
            self.consecutive_poor_scores += 1
            
            # Always use PDM as fallback
            return PlannerType.PDM, self._create_metadata(
                pdm_score, diffusion_score, pdm_safety, diffusion_safety,
                scenario_type, scenario_confidence, safety_fallback=True
            )
        else:
            self.consecutive_poor_scores = 0
        
        # Get empirical performance
        win_rates = self.EMPIRICAL_WIN_RATES.get(scenario_type, {'pdm': 0.5, 'diffusion': 0.5})
        pdm_win_rate = win_rates['pdm']
        diffusion_win_rate = win_rates['diffusion']
        win_rate_gap = abs(pdm_win_rate - diffusion_win_rate)
        empirical_best = PlannerType.PDM if pdm_win_rate > diffusion_win_rate else PlannerType.DIFFUSION
        
        # SCENARIO-BASED DRIFT
        scenario_drift = 0.0
        if scenario_confidence >= self.config.min_confidence:
            # Calculate scenario bias
            if empirical_best == PlannerType.DIFFUSION:
                scenario_bias = min(1.0, win_rate_gap * 2.0)  # Positive for diffusion
            else:
                scenario_bias = -min(1.0, win_rate_gap * 2.0)  # Negative for PDM
            
            # Apply with confidence and consistency boost
            consistency_boost = min(0.2, self.scenario_consistency * 0.02)
            scenario_drift = scenario_bias * self.config.scenario_weight * (scenario_confidence + consistency_boost)
            
            # CONSERVATIVE OVERRIDE - only for very high confidence and large gaps
            if (scenario_confidence >= self.config.high_confidence and 
                win_rate_gap >= self.config.override_threshold and
                self.current_planner != empirical_best and
                self.cycles_since_switch >= self.config.min_switch_interval and
                self.scenario_consistency >= 3):  # Require consistent detection
                
                # Additional safety check
                if empirical_best == PlannerType.PDM or diffusion_safety > 0.5:
                    logger.info(
                        f"Conservative override to {empirical_best.value} for {scenario_type.value} "
                        f"(gap={win_rate_gap:.2f}, conf={scenario_confidence:.2f}, consistency={self.scenario_consistency})"
                    )
                    
                    self.current_planner = empirical_best
                    self.cycles_since_switch = 0
                    self.total_switches += 1
                    
                    return empirical_best, self._create_metadata(
                        pdm_score, diffusion_score, pdm_safety, diffusion_safety,
                        scenario_type, scenario_confidence, override=True
                    )
        
        # SCORE-BASED DRIFT with validation
        score_diff = diffusion_score - pdm_score
        
        # Consider any score difference
        score_drift = self.config.score_weight * np.tanh(score_diff * 5.0)
        
        # SAFETY PENALTY for low safety scores
        safety_penalty = 0.0
        if diffusion_safety < 0.5 and self.current_planner == PlannerType.DIFFUSION:
            safety_penalty = -0.2  # Push towards PDM
        elif pdm_safety < 0.5 and self.current_planner == PlannerType.PDM:
            safety_penalty = 0.2  # Push towards Diffusion (less likely)
        
        # Update preference with memory
        total_drift = (1 - self.config.alpha) * (scenario_drift + score_drift + safety_penalty)
        self.P = self.config.alpha * self.P + total_drift
        self.P = np.clip(self.P, -1.5, 1.5)
        
        # Check minimum switch interval
        if self.cycles_since_switch < self.config.min_switch_interval:
            return self.current_planner, self._create_metadata(
                pdm_score, diffusion_score, pdm_safety, diffusion_safety,
                scenario_type, scenario_confidence, lockout=True
            )
        
        # Make conservative decision
        new_planner = self._make_conservative_decision()
        
        # Validate switch decision
        if new_planner != self.current_planner:
            # Additional validation before switching
            if self._validate_switch(new_planner, pdm_score, diffusion_score, pdm_safety, diffusion_safety):
                logger.info(
                    f"Conservative switch: {self.current_planner.value} → {new_planner.value} | "
                    f"Scenario: {scenario_type.value} | P={self.P:.3f} | "
                    f"Scores: PDM={pdm_score:.3f} Diff={diffusion_score:.3f}"
                )
                self.current_planner = new_planner
                self.cycles_since_switch = 0
                self.total_switches += 1
            else:
                logger.debug(f"Switch to {new_planner.value} rejected by validation")
        
        return self.current_planner, self._create_metadata(
            pdm_score, diffusion_score, pdm_safety, diffusion_safety,
            scenario_type, scenario_confidence
        )
    
    def _make_conservative_decision(self) -> PlannerType:
        """Make decision with conservative thresholds"""
        if self.current_planner == PlannerType.PDM:
            # Require strong evidence to switch away from PDM
            if self.P > self.config.theta_upper:
                return PlannerType.DIFFUSION
        else:
            # Easier to switch back to PDM (safety)
            if self.P < self.config.theta_lower:
                return PlannerType.PDM
        
        return self.current_planner
    
    def _validate_switch(
        self, 
        new_planner: PlannerType,
        pdm_score: float,
        diffusion_score: float,
        pdm_safety: float,
        diffusion_safety: float
    ) -> bool:
        """Validate that switching is reasonable"""
        if not self.config.require_score_validation:
            return True
        
        # Only prevent switches in extreme cases
        
        # Don't switch to diffusion if it has terrible safety
        if new_planner == PlannerType.DIFFUSION and diffusion_safety < 0.1:
            logger.debug("Rejecting switch to Diffusion due to very low safety")
            return False
        
        # Don't switch if the score difference is extreme
        if new_planner == PlannerType.DIFFUSION and pdm_score - diffusion_score > 0.5:
            logger.debug("Rejecting switch to Diffusion due to large score gap")
            return False
        elif new_planner == PlannerType.PDM and diffusion_score - pdm_score > 0.5:
            logger.debug("Rejecting switch to PDM due to large score gap")
            return False
        
        return True
    
    def _create_metadata(
        self,
        pdm_score: float,
        diffusion_score: float,
        pdm_safety: float,
        diffusion_safety: float,
        scenario_type: ScenarioType,
        scenario_confidence: float,
        **flags
    ) -> Dict:
        """Create metadata"""
        win_rates = self.EMPIRICAL_WIN_RATES.get(scenario_type, {'pdm': 0.5, 'diffusion': 0.5})
        empirical_best = PlannerType.PDM if win_rates['pdm'] > win_rates['diffusion'] else PlannerType.DIFFUSION
        
        return {
            'P': self.P,
            'pdm_score': pdm_score,
            'diffusion_score': diffusion_score,
            'pdm_safety': pdm_safety,
            'diffusion_safety': diffusion_safety,
            'scenario': scenario_type.value,
            'scenario_confidence': scenario_confidence,
            'scenario_consistency': self.scenario_consistency,
            'planner': self.current_planner.value,
            'expected_planner': empirical_best.value,
            'cycles_since_switch': self.cycles_since_switch,
            'total_switches': self.total_switches,
            'consecutive_poor_scores': self.consecutive_poor_scores,
            **flags
        }
    
    def reset(self):
        """Reset switcher"""
        self.P = 0.0
        self.current_planner = PlannerType.PDM
        self.cycles_since_switch = 100
        self.total_switches = 0
        self.consecutive_poor_scores = 0
        self.scenario_consistency = 0
        self.last_scenario = ScenarioType.UNKNOWN
        logger.info("Conservative Oracle reset")