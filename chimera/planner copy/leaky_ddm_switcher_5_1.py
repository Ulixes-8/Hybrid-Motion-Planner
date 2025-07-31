"""
Conservative Oracle Switcher - Version 6
Enhanced with scenario-specific boosts for better performance.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field
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
    """Configuration for Conservative Oracle - Version 6"""
    # Memory and thresholds
    alpha: float = 0.6  # Adaptive memory
    theta_upper: float = 0.3
    theta_lower: float = -0.3
    
    # Weights
    scenario_weight: float = 0.7
    score_weight: float = 0.3
    
    # Confidence
    min_confidence: float = 0.55
    high_confidence: float = 0.8
    
    # Override threshold
    override_threshold: float = 0.3
    
    # Safety
    min_switch_interval: int = 2
    min_score_threshold: float = 0.0
    score_difference_threshold: float = 0.08
    
    # Validation
    require_score_validation: bool = False
    fallback_to_pdm: bool = True
    
    # Version 6: Scenario-specific boosts
    scenario_boosts: Dict[str, float] = field(default_factory=lambda: {
        'waiting_for_pedestrian_to_cross': -0.25,
        'traversing_pickup_dropoff': 0.25,
        'following_lane_with_lead': -0.15,
        'high_magnitude_speed': -0.15,
        'changing_lane': 0.1,
        'low_magnitude_speed': 0.15,
        'starting_left_turn': 0.1,
        'near_multiple_vehicles': -0.05,
        'starting_right_turn': -0.05,
    })
    
    # Scenario decay
    scenario_decay: float = 0.85


class ConservativeOracleSwitcher:
    """
    Version 6 Conservative oracle switcher with scenario boosts.
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
        """Initialize version 6 switcher"""
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
        
        # Scenario influence tracking (Version 6)
        self.scenario_influence = 0.0
        self.last_scenario_boost = 0.0
        
        logger.info(
            f"Initialized Conservative Oracle v6: "
            f"α={self.config.alpha}, "
            f"θ=±{self.config.theta_upper}, "
            f"boosts={len(self.config.scenario_boosts)}"
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
        Select planner with enhanced scoring awareness (v6).
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
        
        # SCENARIO-BASED DRIFT with V6 enhancements
        scenario_drift = 0.0
        scenario_boost = 0.0
        
        if scenario_confidence >= self.config.min_confidence:
            # Calculate base scenario bias
            if empirical_best == PlannerType.DIFFUSION:
                scenario_bias = min(1.0, win_rate_gap * 2.0)  # Positive for diffusion
            else:
                scenario_bias = -min(1.0, win_rate_gap * 2.0)  # Negative for PDM
            
            # Apply with confidence and consistency boost
            consistency_boost = min(0.2, self.scenario_consistency * 0.02)
            scenario_drift = scenario_bias * self.config.scenario_weight * (scenario_confidence + consistency_boost)
            
            # VERSION 6: Apply scenario-specific boosts
            scenario_name = scenario_type.value
            if scenario_name in self.config.scenario_boosts:
                scenario_boost = self.config.scenario_boosts[scenario_name]
                # Apply boost with confidence scaling
                scenario_drift += scenario_boost * scenario_confidence
            
            # Update scenario influence tracking
            self.scenario_influence = scenario_drift
            self.last_scenario_boost = scenario_boost
            
            # ENHANCED OVERRIDE - consider both empirical gap and scenario boosts
            effective_gap = win_rate_gap + abs(scenario_boost) * 0.5
            
            if (scenario_confidence >= self.config.high_confidence and 
                effective_gap >= self.config.override_threshold and
                self.current_planner != empirical_best and
                self.cycles_since_switch >= self.config.min_switch_interval and
                self.scenario_consistency >= 2):  # Slightly reduced for v6
                
                # Additional safety check
                if empirical_best == PlannerType.PDM or diffusion_safety > 0.5:
                    logger.info(
                        f"Enhanced override to {empirical_best.value} for {scenario_type.value} "
                        f"(gap={win_rate_gap:.2f}, boost={scenario_boost:.2f}, "
                        f"conf={scenario_confidence:.2f}, consistency={self.scenario_consistency})"
                    )
                    
                    self.current_planner = empirical_best
                    self.cycles_since_switch = 0
                    self.total_switches += 1
                    
                    return empirical_best, self._create_metadata(
                        pdm_score, diffusion_score, pdm_safety, diffusion_safety,
                        scenario_type, scenario_confidence, override=True,
                        scenario_boost=scenario_boost
                    )
        else:
            # Decay scenario influence when confidence is low
            self.scenario_influence *= self.config.scenario_decay
            scenario_drift = self.scenario_influence
        
        # SCORE-BASED DRIFT with enhanced scoring awareness
        score_diff = diffusion_score - pdm_score
        
        # V6: More nuanced score interpretation
        # Enhanced scores should be more meaningful, so we can trust larger differences
        if abs(score_diff) > 0.2:  # Significant difference
            score_drift = self.config.score_weight * np.tanh(score_diff * 7.0)
        else:  # Small difference
            score_drift = self.config.score_weight * np.tanh(score_diff * 5.0)
        
        # SAFETY PENALTY for low safety scores
        safety_penalty = 0.0
        if diffusion_safety < 0.5 and self.current_planner == PlannerType.DIFFUSION:
            safety_penalty = -0.2  # Push towards PDM
        elif pdm_safety < 0.5 and self.current_planner == PlannerType.PDM:
            safety_penalty = 0.15  # Push towards Diffusion (less aggressive)
        
        # Update preference with memory
        total_drift = (1 - self.config.alpha) * (scenario_drift + score_drift + safety_penalty)
        self.P = self.config.alpha * self.P + total_drift
        self.P = np.clip(self.P, -1.5, 1.5)
        
        # Check minimum switch interval
        if self.cycles_since_switch < self.config.min_switch_interval:
            return self.current_planner, self._create_metadata(
                pdm_score, diffusion_score, pdm_safety, diffusion_safety,
                scenario_type, scenario_confidence, lockout=True,
                scenario_boost=scenario_boost
            )
        
        # Make decision
        new_planner = self._make_decision()
        
        # Validate switch decision
        if new_planner != self.current_planner:
            # Additional validation before switching
            if self._validate_switch_v6(
                new_planner, pdm_score, diffusion_score, 
                pdm_safety, diffusion_safety, scenario_boost
            ):
                logger.info(
                    f"V6 switch: {self.current_planner.value} → {new_planner.value} | "
                    f"Scenario: {scenario_type.value} (boost={scenario_boost:.2f}) | "
                    f"P={self.P:.3f} | Scores: PDM={pdm_score:.3f} Diff={diffusion_score:.3f}"
                )
                self.current_planner = new_planner
                self.cycles_since_switch = 0
                self.total_switches += 1
            else:
                logger.debug(f"Switch to {new_planner.value} rejected by v6 validation")
        
        return self.current_planner, self._create_metadata(
            pdm_score, diffusion_score, pdm_safety, diffusion_safety,
            scenario_type, scenario_confidence, scenario_boost=scenario_boost
        )
    
    def _make_decision(self) -> PlannerType:
        """Make decision based on preference state"""
        if self.current_planner == PlannerType.PDM:
            if self.P > self.config.theta_upper:
                return PlannerType.DIFFUSION
        else:
            if self.P < self.config.theta_lower:
                return PlannerType.PDM
        
        return self.current_planner
    
    def _validate_switch_v6(
        self, 
        new_planner: PlannerType,
        pdm_score: float,
        diffusion_score: float,
        pdm_safety: float,
        diffusion_safety: float,
        scenario_boost: float
    ) -> bool:
        """Enhanced validation for v6 with scenario boost awareness"""
        if not self.config.require_score_validation:
            # Even without validation, apply minimal safety checks
            if new_planner == PlannerType.DIFFUSION and diffusion_safety < 0.1:
                return False
            return True
        
        # V6: Consider scenario boost in validation
        effective_score_diff = abs(pdm_score - diffusion_score)
        
        # If there's a strong scenario boost, be more lenient with score differences
        if abs(scenario_boost) > 0.2:
            allowed_score_gap = 0.6  # More lenient
        else:
            allowed_score_gap = 0.4  # Normal
        
        # Don't switch to diffusion if it has terrible safety
        if new_planner == PlannerType.DIFFUSION and diffusion_safety < 0.1:
            logger.debug("Rejecting switch to Diffusion due to very low safety")
            return False
        
        # Don't switch if the score difference is extreme (unless boosted)
        if new_planner == PlannerType.DIFFUSION and pdm_score - diffusion_score > allowed_score_gap:
            logger.debug(f"Rejecting switch to Diffusion due to large score gap ({allowed_score_gap})")
            return False
        elif new_planner == PlannerType.PDM and diffusion_score - pdm_score > allowed_score_gap:
            logger.debug(f"Rejecting switch to PDM due to large score gap ({allowed_score_gap})")
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
        scenario_boost: float = 0.0,
        **flags
    ) -> Dict:
        """Create metadata with v6 enhancements"""
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
            'scenario_boost': scenario_boost,
            'scenario_influence': self.scenario_influence,
            'planner': self.current_planner.value,
            'expected_planner': empirical_best.value,
            'cycles_since_switch': self.cycles_since_switch,
            'total_switches': self.total_switches,
            'consecutive_poor_scores': self.consecutive_poor_scores,
            'version': 6,
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
        self.last_scenario_boost = 0.0
        logger.info("Conservative Oracle v6 reset")