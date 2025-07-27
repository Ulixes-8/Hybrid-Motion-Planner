"""
Leaky DDM Switcher Version 4: Scenario-First Decision Making
Prioritizes empirical scenario performance with hierarchical decision logic.
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
    # PDM WINS (>80% win rate)
    BEHIND_LONG_VEHICLE = "behind_long_vehicle"  # PDM: 92.6%
    FOLLOWING_LANE_WITH_LEAD = "following_lane_with_lead"  # PDM: 93.9%
    HIGH_MAGNITUDE_SPEED = "high_magnitude_speed"  # PDM: 91.9%
    NEAR_MULTIPLE_VEHICLES = "near_multiple_vehicles"  # PDM: 65.9%
    STARTING_RIGHT_TURN = "starting_right_turn"  # PDM: 69.3%
    STATIONARY_IN_TRAFFIC = "stationary_in_traffic"  # PDM: 86.8%
    STOPPING_WITH_LEAD = "stopping_with_lead"  # PDM: 98.4%
    WAITING_FOR_PEDESTRIAN = "waiting_for_pedestrian_to_cross"  # PDM: 88.2%
    
    # DIFFUSION WINS (>55% win rate)
    CHANGING_LANE = "changing_lane"  # Diffusion: 76.7%
    HIGH_LATERAL_ACCELERATION = "high_lateral_acceleration"  # Diffusion: 57.3%
    LOW_MAGNITUDE_SPEED = "low_magnitude_speed"  # Diffusion: 48.0% (close)
    STARTING_LEFT_TURN = "starting_left_turn"  # Diffusion: 67.0%
    STARTING_STRAIGHT_INTERSECTION = "starting_straight_traffic_light_intersection_traversal"  # Diffusion: 90.0%
    TRAVERSING_PICKUP_DROPOFF = "traversing_pickup_dropoff"  # Diffusion: 68.5%
    
    UNKNOWN = "unknown"


@dataclass
class LeakyDDMConfigV4:
    """Configuration for V4 Switcher"""
    # Core DDM parameters - reduced thresholds for faster switching
    alpha: float = 0.75  # Slightly faster decay for responsiveness
    theta_upper: float = 0.4  # Lower threshold for faster switching
    theta_lower: float = -0.4  # Symmetric
    
    # Score influence - moderate
    score_gain: float = 0.12  # Slightly higher for score influence
    
    # Scenario confidence thresholds
    high_confidence_threshold: float = 0.8  # Direct switching
    moderate_confidence_threshold: float = 0.6  # Strong bias
    low_confidence_threshold: float = 0.4  # Weak bias
    
    # Safety thresholds
    critical_score_threshold: float = 0.1  # Below this is critical failure
    
    # Asymmetric biasing
    pdm_preference_bias: float = -0.05  # Slight PDM preference
    
    # Switch control
    scenario_lockout_cycles: int = 0  # No lockout for scenario changes
    score_lockout_cycles: int = 1  # Minimal lockout for score-based
    
    # Emergency overrides
    pedestrian_force_cycles: int = 5  # Force PDM for N cycles after pedestrian
    safety_force_cycles: int = 3  # Force PDM after safety violation


class LeakyDDMSwitcherV4:
    """
    Version 4 Switcher: Hierarchical decision making
    1. Safety-critical scenarios → Force appropriate planner
    2. High-confidence scenarios → Use empirical winner
    3. Moderate confidence → Strong bias toward winner
    4. Low confidence → Score-based DDM
    """
    
    # Empirical winners
    SCENARIO_WINNERS = {
        # Clear PDM winners
        ScenarioType.STOPPING_WITH_LEAD: PlannerType.PDM,
        ScenarioType.WAITING_FOR_PEDESTRIAN: PlannerType.PDM,
        ScenarioType.FOLLOWING_LANE_WITH_LEAD: PlannerType.PDM,
        ScenarioType.BEHIND_LONG_VEHICLE: PlannerType.PDM,
        ScenarioType.HIGH_MAGNITUDE_SPEED: PlannerType.PDM,
        ScenarioType.STATIONARY_IN_TRAFFIC: PlannerType.PDM,
        ScenarioType.NEAR_MULTIPLE_VEHICLES: PlannerType.PDM,
        ScenarioType.STARTING_RIGHT_TURN: PlannerType.PDM,
        
        # Clear Diffusion winners
        ScenarioType.STARTING_STRAIGHT_INTERSECTION: PlannerType.DIFFUSION,
        ScenarioType.CHANGING_LANE: PlannerType.DIFFUSION,
        ScenarioType.TRAVERSING_PICKUP_DROPOFF: PlannerType.DIFFUSION,
        ScenarioType.STARTING_LEFT_TURN: PlannerType.DIFFUSION,
        ScenarioType.HIGH_LATERAL_ACCELERATION: PlannerType.DIFFUSION,
        ScenarioType.LOW_MAGNITUDE_SPEED: PlannerType.DIFFUSION,
        
        # Unknown - slight PDM preference
        ScenarioType.UNKNOWN: PlannerType.PDM
    }
    
    # Safety-critical scenarios
    SAFETY_CRITICAL = {
        ScenarioType.STOPPING_WITH_LEAD,
        ScenarioType.WAITING_FOR_PEDESTRIAN,
        ScenarioType.STATIONARY_IN_TRAFFIC
    }
    
    def __init__(self, config: LeakyDDMConfigV4 = None):
        """Initialize V4 switcher"""
        self.config = config or LeakyDDMConfigV4()
        
        # Core state
        self.P = self.config.pdm_preference_bias  # Start with slight PDM bias
        self.current_planner = PlannerType.PDM
        
        # Tracking
        self.cycles_since_switch = 0
        self.switch_count = 0
        self.last_scenario = ScenarioType.UNKNOWN
        self.last_decision_reason = "initial"
        
        # Force counters
        self.force_pdm_cycles = 0
        self.force_diffusion_cycles = 0
        
        # Performance tracking
        self.scenario_correct_decisions = 0
        self.scenario_total_decisions = 0
        
        logger.info("Initialized LeakyDDM V4 with scenario-first logic")
    
    def update_and_select(
        self,
        pdm_score: float,
        diffusion_score: float,
        pdm_progress: Optional[float] = None,
        safety_veto_diffusion: bool = False,
        scenario_result = None  # ScenarioDetectionResult
    ) -> Tuple[PlannerType, Dict]:
        """
        Hierarchical decision making:
        1. Handle forced states (safety overrides)
        2. High confidence scenario → direct selection
        3. Moderate confidence → biased DDM
        4. Low/no confidence → score-based DDM
        """
        self.cycles_since_switch += 1
        
        # Update force counters
        if self.force_pdm_cycles > 0:
            self.force_pdm_cycles -= 1
        if self.force_diffusion_cycles > 0:
            self.force_diffusion_cycles -= 1
        
        # Extract scenario info
        scenario_type = ScenarioType.UNKNOWN
        scenario_confidence = 0.0
        scenario_changed = False
        
        if scenario_result is not None:
            scenario_type = scenario_result.scenario_type
            scenario_confidence = scenario_result.confidence
            scenario_changed = scenario_result.is_transition
            
            # Track scenario changes
            if scenario_type != self.last_scenario:
                self.last_scenario = scenario_type
        
        # LEVEL 1: SAFETY OVERRIDES
        if safety_veto_diffusion:
            self.force_pdm_cycles = self.config.safety_force_cycles
            self.P = min(self.P - 0.3, -self.config.theta_upper)  # Strong PDM bias
            return self._make_decision("safety_veto", pdm_score, diffusion_score, 
                                     scenario_type, scenario_confidence)
        
        if scenario_type == ScenarioType.WAITING_FOR_PEDESTRIAN and scenario_confidence > 0.6:
            self.force_pdm_cycles = self.config.pedestrian_force_cycles
            return self._force_planner(PlannerType.PDM, "pedestrian_safety", 
                                     pdm_score, diffusion_score, scenario_type)
        
        # Check other safety-critical scenarios
        if scenario_type in self.SAFETY_CRITICAL and scenario_confidence > 0.7:
            if pdm_score < self.config.critical_score_threshold:
                logger.warning(f"Safety-critical scenario {scenario_type.value} but PDM score low: {pdm_score:.3f}")
            else:
                return self._force_planner(PlannerType.PDM, "safety_critical", 
                                         pdm_score, diffusion_score, scenario_type)
        
        # LEVEL 2: HIGH-CONFIDENCE SCENARIO
        if scenario_confidence >= self.config.high_confidence_threshold:
            best_planner = self.SCENARIO_WINNERS.get(scenario_type, PlannerType.PDM)
            
            # Track correctness
            self.scenario_total_decisions += 1
            if best_planner == self.current_planner:
                self.scenario_correct_decisions += 1
            
            # Allow immediate switch for scenario changes
            if scenario_changed:
                self.cycles_since_switch = 100  # Reset lockout
            
            # Direct selection for very high confidence
            if scenario_confidence >= 0.9:
                return self._force_planner(best_planner, f"scenario_high_conf_{scenario_type.value}", 
                                         pdm_score, diffusion_score, scenario_type)
            
            # Strong bias for high confidence
            target_p = self.config.theta_upper if best_planner == PlannerType.DIFFUSION else -self.config.theta_upper
            self.P = 0.7 * self.P + 0.3 * target_p
            
        # LEVEL 3: MODERATE CONFIDENCE SCENARIO
        elif scenario_confidence >= self.config.moderate_confidence_threshold:
            best_planner = self.SCENARIO_WINNERS.get(scenario_type, PlannerType.PDM)
            
            # Moderate bias
            bias = 0.2 if best_planner == PlannerType.DIFFUSION else -0.2
            self.P = self.config.alpha * self.P + (1 - self.config.alpha) * bias
        
        # LEVEL 4: SCORE-BASED DDM (with slight PDM preference)
        score_diff = diffusion_score - pdm_score
        
        # Apply asymmetric gain (harder to switch to diffusion)
        if score_diff > 0:
            score_drift = self.config.score_gain * 0.8 * score_diff  # Reduced gain for diffusion
        else:
            score_drift = self.config.score_gain * score_diff
        
        # Update preference with decay and drift
        self.P = self.config.alpha * self.P + score_drift
        
        # Add slight PDM preference
        self.P += self.config.pdm_preference_bias * (1 - self.config.alpha)
        
        # Clamp P
        self.P = np.clip(self.P, -1.0, 1.0)
        
        # Make decision
        return self._make_decision("score_based", pdm_score, diffusion_score, 
                                 scenario_type, scenario_confidence)
    
    def _force_planner(
        self, 
        planner: PlannerType, 
        reason: str,
        pdm_score: float,
        diffusion_score: float,
        scenario_type: ScenarioType
    ) -> Tuple[PlannerType, Dict]:
        """Force a specific planner with logging"""
        if planner != self.current_planner:
            self.switch_count += 1
            self.cycles_since_switch = 0
            logger.info(
                f"FORCED SWITCH to {planner.value}: {reason} | "
                f"Scenario: {scenario_type.value}"
            )
        
        self.current_planner = planner
        self.last_decision_reason = reason
        
        # Adjust P to match decision
        if planner == PlannerType.PDM:
            self.P = min(self.P, -0.2)
        else:
            self.P = max(self.P, 0.2)
        
        return planner, self._create_metadata(pdm_score, diffusion_score, 
                                            scenario_type, reason)
    
    def _make_decision(
        self, 
        reason_prefix: str,
        pdm_score: float,
        diffusion_score: float,
        scenario_type: ScenarioType,
        scenario_confidence: float
    ) -> Tuple[PlannerType, Dict]:
        """Make decision with hysteresis and lockout"""
        # Check lockout (but allow override for scenario changes)
        if self.cycles_since_switch < self.config.score_lockout_cycles:
            if not (scenario_type != ScenarioType.UNKNOWN and 
                   scenario_confidence > self.config.moderate_confidence_threshold):
                return self.current_planner, self._create_metadata(
                    pdm_score, diffusion_score, scenario_type, 
                    f"{reason_prefix}_lockout"
                )
        
        # Apply hysteresis
        if self.current_planner == PlannerType.PDM:
            if self.P > self.config.theta_upper:
                new_planner = PlannerType.DIFFUSION
                reason = f"{reason_prefix}_threshold_crossed"
            else:
                new_planner = PlannerType.PDM
                reason = f"{reason_prefix}_no_change"
        else:
            if self.P < self.config.theta_lower:
                new_planner = PlannerType.PDM
                reason = f"{reason_prefix}_threshold_crossed"
            else:
                new_planner = PlannerType.DIFFUSION
                reason = f"{reason_prefix}_no_change"
        
        # Handle switching
        if new_planner != self.current_planner:
            self.switch_count += 1
            self.cycles_since_switch = 0
            
            logger.info(
                f"SWITCH {self.switch_count}: {self.current_planner.value} → {new_planner.value} | "
                f"Reason: {reason} | P={self.P:.3f} | "
                f"PDM={pdm_score:.3f} Diff={diffusion_score:.3f}"
            )
            
            self.current_planner = new_planner
        
        self.last_decision_reason = reason
        return new_planner, self._create_metadata(pdm_score, diffusion_score, 
                                                 scenario_type, reason)
    
    def _create_metadata(
        self, 
        pdm_score: float,
        diffusion_score: float,
        scenario_type: ScenarioType,
        decision_reason: str
    ) -> Dict:
        """Create comprehensive metadata"""
        accuracy = (self.scenario_correct_decisions / 
                   max(1, self.scenario_total_decisions))
        
        return {
            'P': self.P,
            'pdm_score': pdm_score,
            'diffusion_score': diffusion_score,
            'score_gap': diffusion_score - pdm_score,
            'scenario': scenario_type.value,
            'expected_planner': self.SCENARIO_WINNERS.get(scenario_type, PlannerType.PDM).value,
            'current_planner': self.current_planner.value,
            'decision_reason': decision_reason,
            'cycles_since_switch': self.cycles_since_switch,
            'switch_count': self.switch_count,
            'force_pdm_cycles': self.force_pdm_cycles,
            'scenario_accuracy': accuracy
        }
    
    def reset(self):
        """Reset switcher state"""
        self.P = self.config.pdm_preference_bias
        self.current_planner = PlannerType.PDM
        self.cycles_since_switch = 0
        self.switch_count = 0
        self.last_scenario = ScenarioType.UNKNOWN
        self.last_decision_reason = "reset"
        self.force_pdm_cycles = 0
        self.force_diffusion_cycles = 0
        self.scenario_correct_decisions = 0
        self.scenario_total_decisions = 0
        logger.info("LeakyDDM V4 switcher reset")