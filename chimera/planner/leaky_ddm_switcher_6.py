"""
Empirical Trust Switcher - Version 6
Trusts empirical data strongly for optimal performance.
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
class EmpiricalTrustConfig:
    """Configuration for Empirical Trust Switcher"""
    # Memory and thresholds
    alpha: float = 0.5  # Lower memory for faster adaptation
    theta_upper: float = 0.25  # Tighter threshold
    theta_lower: float = -0.25
    
    # Weights
    scenario_weight: float = 0.8  # Strong scenario influence
    score_weight: float = 0.2  # Secondary
    
    # Confidence
    min_confidence: float = 0.5  # Lower barrier
    high_confidence: float = 0.75  # Lower for faster override
    
    # Override threshold
    override_threshold: float = 0.25  # 25% gap triggers override
    
    # Safety
    min_switch_interval: int = 1  # Minimal lockout
    min_score_threshold: float = 0.0
    score_difference_threshold: float = 0.1
    
    # Validation
    require_score_validation: bool = False  # Trust scenarios
    fallback_to_pdm: bool = True
    
    # Scenario-specific boosts
    scenario_boosts: Dict[str, float] = None
    scenario_decay: float = 0.8  # Faster decay
    
    def __post_init__(self):
        if self.scenario_boosts is None:
            self.scenario_boosts = {
                'waiting_for_pedestrian_to_cross': -0.3,  # Strong PDM
                'traversing_pickup_dropoff': 0.3,  # Strong Diffusion
                'following_lane_with_lead': -0.2,  # PDM
                'high_magnitude_speed': -0.2,  # PDM
                'low_magnitude_speed': 0.2,  # Diffusion
            }


class EmpiricalTrustSwitcher:
    """
    Version 6 switcher that strongly trusts empirical performance data.
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
    
    def __init__(self, config: EmpiricalTrustConfig = None):
        """Initialize empirical trust switcher"""
        self.config = config or EmpiricalTrustConfig()
        
        # State
        self.P = 0.0  # Neutral start
        self.current_planner = PlannerType.PDM  # Safe default
        self.cycles_since_switch = 100
        self.total_switches = 0
        
        # Performance tracking
        self.consecutive_poor_scores = 0
        self.scenario_consistency = 0
        self.last_scenario = ScenarioType.UNKNOWN
        
        # Scenario influence tracking
        self.scenario_influence = 0.0
        
        logger.info(
            f"Initialized Empirical Trust Switcher V6: "
            f"α={self.config.alpha}, θ=±{self.config.theta_upper}, "
            f"scenario_weight={self.config.scenario_weight}"
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
        Select planner with strong trust in empirical data.
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
        
        # Get empirical performance
        win_rates = self.EMPIRICAL_WIN_RATES.get(scenario_type, {'pdm': 0.5, 'diffusion': 0.5})
        pdm_win_rate = win_rates['pdm']
        diffusion_win_rate = win_rates['diffusion']
        win_rate_gap = abs(pdm_win_rate - diffusion_win_rate)
        empirical_best = PlannerType.PDM if pdm_win_rate > diffusion_win_rate else PlannerType.DIFFUSION
        
        # FAST OVERRIDE for high-confidence scenarios with clear winners
        if (scenario_confidence >= self.config.high_confidence and 
            win_rate_gap >= self.config.override_threshold and
            self.cycles_since_switch >= self.config.min_switch_interval):
            
            if empirical_best != self.current_planner:
                logger.info(
                    f"Fast override to {empirical_best.value} for {scenario_type.value} "
                    f"(gap={win_rate_gap:.2f}, conf={scenario_confidence:.2f})"
                )
                
                self.current_planner = empirical_best
                self.cycles_since_switch = 0
                self.total_switches += 1
                
                return empirical_best, self._create_metadata(
                    pdm_score, diffusion_score, pdm_safety, diffusion_safety,
                    scenario_type, scenario_confidence, fast_override=True
                )
        
        # SCENARIO-BASED DRIFT with boosts
        scenario_drift = 0.0
        if scenario_confidence >= self.config.min_confidence:
            # Base scenario bias
            if empirical_best == PlannerType.DIFFUSION:
                base_bias = min(1.0, win_rate_gap * 3.0)  # Amplified
            else:
                base_bias = -min(1.0, win_rate_gap * 3.0)
            
            # Apply scenario-specific boost if available
            boost = self.config.scenario_boosts.get(scenario_type.value, 0.0)
            total_bias = base_bias + boost
            
            # Apply with confidence
            scenario_drift = total_bias * self.config.scenario_weight * scenario_confidence
            
            # Update scenario influence with decay
            self.scenario_influence = self.config.scenario_decay * self.scenario_influence + (1 - self.config.scenario_decay) * abs(scenario_drift)
        else:
            # Decay scenario influence when no clear scenario
            self.scenario_influence *= self.config.scenario_decay
        
        # SCORE-BASED DRIFT (secondary)
        score_diff = diffusion_score - pdm_score
        
        # Only consider significant score differences
        if abs(score_diff) > self.config.score_difference_threshold:
            score_drift = self.config.score_weight * np.tanh(score_diff * 10.0)
        else:
            score_drift = 0.0
        
        # SAFETY CONSIDERATION (minimal)
        safety_penalty = 0.0
        if diffusion_safety < 0.3 and self.current_planner == PlannerType.DIFFUSION:
            safety_penalty = -0.1  # Small push towards PDM
        
        # Update preference with lower memory
        total_drift = (1 - self.config.alpha) * (scenario_drift + score_drift + safety_penalty)
        self.P = self.config.alpha * self.P + total_drift
        self.P = np.clip(self.P, -2.0, 2.0)  # Allow stronger preferences
        
        # Make decision
        new_planner = self._make_decision()
        
        # Switch if needed (minimal validation)
        if new_planner != self.current_planner and self.cycles_since_switch >= self.config.min_switch_interval:
            logger.info(
                f"Empirical switch: {self.current_planner.value} → {new_planner.value} | "
                f"Scenario: {scenario_type.value} (conf={scenario_confidence:.2f}) | "
                f"P={self.P:.3f} | Expected: {empirical_best.value}"
            )
            self.current_planner = new_planner
            self.cycles_since_switch = 0
            self.total_switches += 1
        
        return self.current_planner, self._create_metadata(
            pdm_score, diffusion_score, pdm_safety, diffusion_safety,
            scenario_type, scenario_confidence
        )
    
    def _make_decision(self) -> PlannerType:
        """Make decision based on preference"""
        if self.P > self.config.theta_upper:
            return PlannerType.DIFFUSION
        elif self.P < self.config.theta_lower:
            return PlannerType.PDM
        else:
            return self.current_planner
    
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
            'scenario_influence': self.scenario_influence,
            'planner': self.current_planner.value,
            'expected_planner': empirical_best.value,
            'empirical_gap': abs(win_rates['pdm'] - win_rates['diffusion']),
            'cycles_since_switch': self.cycles_since_switch,
            'total_switches': self.total_switches,
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
        self.scenario_influence = 0.0
        logger.info("Empirical Trust Switcher reset")