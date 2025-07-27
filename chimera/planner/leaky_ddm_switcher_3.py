"""
Focused Leaky DDM Switcher - Scenario Recognition Priority
Switches to the empirically better planner based on scenario detection.
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
    """Scenario types with clear empirical winners"""
    # PDM WINS (8 types)
    BEHIND_LONG_VEHICLE = "behind_long_vehicle"  # PDM: 0.926
    FOLLOWING_LANE_WITH_LEAD = "following_lane_with_lead"  # PDM: 0.939
    HIGH_MAGNITUDE_SPEED = "high_magnitude_speed"  # PDM: 0.919
    NEAR_MULTIPLE_VEHICLES = "near_multiple_vehicles"  # PDM: 0.659
    STARTING_RIGHT_TURN = "starting_right_turn"  # PDM: 0.693
    STATIONARY_IN_TRAFFIC = "stationary_in_traffic"  # PDM: 0.868
    STOPPING_WITH_LEAD = "stopping_with_lead"  # PDM: 0.984
    WAITING_FOR_PEDESTRIAN = "waiting_for_pedestrian_to_cross"  # PDM: 0.882
    
    # DIFFUSION WINS (6 types)
    CHANGING_LANE = "changing_lane"  # Diffusion: 0.767
    HIGH_LATERAL_ACCELERATION = "high_lateral_acceleration"  # Diffusion: 0.573
    LOW_MAGNITUDE_SPEED = "low_magnitude_speed"  # Diffusion: 0.480
    STARTING_LEFT_TURN = "starting_left_turn"  # Diffusion: 0.670
    STARTING_STRAIGHT_INTERSECTION = "starting_straight_traffic_light_intersection_traversal"  # Diffusion: 0.900
    TRAVERSING_PICKUP_DROPOFF = "traversing_pickup_dropoff"  # Diffusion: 0.685
    
    UNKNOWN = "unknown"


@dataclass
class LeakyDDMConfig:
    """Focused configuration for scenario-based switching"""
    # Core DDM
    alpha: float = 0.8  # Memory decay
    theta_upper: float = 0.5  # Switch to Diffusion threshold
    theta_lower: float = -0.5  # Switch to PDM threshold
    
    # Evidence processing
    k_gain: float = 0.1  # Score influence
    
    # Scenario influence - STRONG
    scenario_scale: float = 0.6  # Strong scenario influence
    scenario_confidence_threshold: float = 0.6  # Reasonable confidence
    
    # Minimal lockout for responsiveness
    switch_lockout_cycles: int = 2
    
    # Score validation
    min_score_threshold: float = 0.05  # Only reject if extremely low


class LeakyDDMSwitcher:
    """
    Focused switcher that prioritizes scenario recognition.
    """
    
    # Empirical performance data - WHO WINS EACH SCENARIO
    SCENARIO_WINNERS = {
        # PDM WINS
        ScenarioType.BEHIND_LONG_VEHICLE: PlannerType.PDM,
        ScenarioType.FOLLOWING_LANE_WITH_LEAD: PlannerType.PDM,
        ScenarioType.HIGH_MAGNITUDE_SPEED: PlannerType.PDM,
        ScenarioType.NEAR_MULTIPLE_VEHICLES: PlannerType.PDM,
        ScenarioType.STARTING_RIGHT_TURN: PlannerType.PDM,
        ScenarioType.STATIONARY_IN_TRAFFIC: PlannerType.PDM,
        ScenarioType.STOPPING_WITH_LEAD: PlannerType.PDM,
        ScenarioType.WAITING_FOR_PEDESTRIAN: PlannerType.PDM,
        
        # DIFFUSION WINS
        ScenarioType.CHANGING_LANE: PlannerType.DIFFUSION,
        ScenarioType.HIGH_LATERAL_ACCELERATION: PlannerType.DIFFUSION,
        ScenarioType.LOW_MAGNITUDE_SPEED: PlannerType.DIFFUSION,
        ScenarioType.STARTING_LEFT_TURN: PlannerType.DIFFUSION,
        ScenarioType.STARTING_STRAIGHT_INTERSECTION: PlannerType.DIFFUSION,
        ScenarioType.TRAVERSING_PICKUP_DROPOFF: PlannerType.DIFFUSION,
        
        # Unknown - slight PDM preference
        ScenarioType.UNKNOWN: PlannerType.PDM
    }
    
    # Scenario bias strengths based on win margins
    SCENARIO_BIASES = {
        # Strong PDM scenarios (big win margins)
        ScenarioType.STOPPING_WITH_LEAD: -0.9,  # 98.4% vs 1.6%
        ScenarioType.FOLLOWING_LANE_WITH_LEAD: -0.8,  # 93.9% vs 6.7%
        ScenarioType.BEHIND_LONG_VEHICLE: -0.8,  # 92.6% vs 7.4%
        ScenarioType.HIGH_MAGNITUDE_SPEED: -0.7,  # 91.9% vs 8.1%
        ScenarioType.WAITING_FOR_PEDESTRIAN: -0.7,  # 88.2% vs 11.8%
        ScenarioType.STATIONARY_IN_TRAFFIC: -0.6,  # 86.8% vs 13.2%
        ScenarioType.STARTING_RIGHT_TURN: -0.3,  # 69.3% vs 30.7%
        ScenarioType.NEAR_MULTIPLE_VEHICLES: -0.3,  # 65.9% vs 34.1%
        
        # Strong Diffusion scenarios
        ScenarioType.STARTING_STRAIGHT_INTERSECTION: 0.8,  # 10.0% vs 90.0%
        ScenarioType.CHANGING_LANE: 0.5,  # 23.3% vs 76.7%
        ScenarioType.TRAVERSING_PICKUP_DROPOFF: 0.4,  # 31.5% vs 68.5%
        ScenarioType.STARTING_LEFT_TURN: 0.3,  # 33.0% vs 67.0%
        ScenarioType.HIGH_LATERAL_ACCELERATION: 0.2,  # 42.7% vs 57.3%
        ScenarioType.LOW_MAGNITUDE_SPEED: 0.1,  # 43.2% vs 48.0% (close)
        
        ScenarioType.UNKNOWN: 0.0
    }
    
    def __init__(self, config: LeakyDDMConfig = None):
        """Initialize focused switcher"""
        self.config = config or LeakyDDMConfig()
        
        # Core state
        self.P = 0.0  # Start neutral
        self.current_planner = PlannerType.PDM
        
        # Tracking
        self.cycles_since_switch = 100
        self.last_scenario = ScenarioType.UNKNOWN
        self.scenario_persistence = 0
        self.switch_count = 0
        
        logger.info(
            f"Initialized Focused LeakyDDM: "
            f"scenario_scale={self.config.scenario_scale}"
        )
    
    def update_and_select(
        self,
        pdm_score: float,
        diffusion_score: float,
        pdm_progress: Optional[float] = None,
        safety_vetoed: bool = False,
        scenario_type: Optional['ScenarioType'] = None,
        scenario_confidence: Optional[float] = None,
        scenario_features: Optional[Dict] = None
    ) -> Tuple[PlannerType, Dict]:
        """
        Select planner with scenario recognition as top priority.
        """
        self.cycles_since_switch += 1
        
        # Track scenario persistence
        if scenario_type == self.last_scenario:
            self.scenario_persistence += 1
        else:
            self.scenario_persistence = 0
            self.last_scenario = scenario_type
        
        # Validate scores
        if pdm_score < self.config.min_score_threshold and diffusion_score < self.config.min_score_threshold:
            logger.warning(f"Both scores very low: PDM={pdm_score:.3f}, Diff={diffusion_score:.3f}")
            return self.current_planner, self._create_metadata(
                pdm_score, diffusion_score, scenario_type, scenario_confidence,
                low_scores=True
            )
        
        # SCENARIO-BASED DECISION (PRIMARY)
        if scenario_type is not None and scenario_confidence is not None:
            if scenario_confidence >= self.config.scenario_confidence_threshold:
                # Get the empirically better planner for this scenario
                best_planner = self.SCENARIO_WINNERS.get(scenario_type, PlannerType.PDM)
                scenario_bias = self.SCENARIO_BIASES.get(scenario_type, 0.0)
                
                # Apply strong scenario influence
                scenario_drift = scenario_bias * self.config.scenario_scale * scenario_confidence
                
                # If we should be using the best planner but aren't, push harder
                if best_planner != self.current_planner:
                    # Add urgency based on persistence
                    urgency = min(0.2, self.scenario_persistence * 0.02)
                    scenario_drift += np.sign(scenario_bias) * urgency
                    
                    logger.debug(
                        f"Scenario {scenario_type.value}: "
                        f"Best={best_planner.value}, Current={self.current_planner.value}, "
                        f"Drift={scenario_drift:.3f}"
                    )
            else:
                scenario_drift = 0.0
        else:
            scenario_drift = 0.0
        
        # SCORE-BASED DRIFT (SECONDARY)
        score_diff = diffusion_score - pdm_score
        score_drift = (1 - self.config.alpha) * self.config.k_gain * score_diff
        
        # Update preference accumulator
        self.P = np.clip(
            self.config.alpha * self.P + score_drift + scenario_drift,
            -1.5, 1.5  # Reasonable bounds
        )
        
        # Check lockout
        if self.cycles_since_switch < self.config.switch_lockout_cycles:
            # Allow override for strong scenario mismatch
            if scenario_type is not None:
                best_planner = self.SCENARIO_WINNERS.get(scenario_type, PlannerType.PDM)
                if (best_planner != self.current_planner and 
                    abs(self.SCENARIO_BIASES.get(scenario_type, 0)) > 0.5 and
                    self.scenario_persistence >= 3):
                    # Force switch for strong scenarios
                    logger.info(f"Overriding lockout for strong scenario: {scenario_type.value}")
                else:
                    return self.current_planner, self._create_metadata(
                        pdm_score, diffusion_score, scenario_type, scenario_confidence,
                        lockout=True
                    )
            else:
                return self.current_planner, self._create_metadata(
                    pdm_score, diffusion_score, scenario_type, scenario_confidence,
                    lockout=True
                )
        
        # Make decision
        new_planner = self._make_decision()
        
        # Log switches
        if new_planner != self.current_planner:
            self.switch_count += 1
            self.cycles_since_switch = 0
            
            logger.info(
                f"SWITCH {self.switch_count}: {self.current_planner.value} → {new_planner.value} | "
                f"Scenario: {scenario_type.value if scenario_type else 'unknown'} | "
                f"P={self.P:.3f}, PDM={pdm_score:.3f}, Diff={diffusion_score:.3f}"
            )
            
            self.current_planner = new_planner
        
        return new_planner, self._create_metadata(
            pdm_score, diffusion_score, scenario_type, scenario_confidence
        )
    
    def _make_decision(self) -> PlannerType:
        """Simple threshold crossing"""
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
        return {
            'P': self.P,
            'pdm_score': pdm_score,
            'diffusion_score': diffusion_score,
            'score_gap': diffusion_score - pdm_score,
            'scenario': scenario_type.value if scenario_type else 'unknown',
            'scenario_confidence': scenario_confidence or 0.0,
            'scenario_persistence': self.scenario_persistence,
            'cycles_since_switch': self.cycles_since_switch,
            'switch_count': self.switch_count,
            'planner': self.current_planner.value,
            'expected_planner': self.SCENARIO_WINNERS.get(scenario_type, PlannerType.PDM).value if scenario_type else 'unknown',
            **flags
        }
    
    def reset(self):
        """Reset switcher"""
        self.P = 0.0
        self.current_planner = PlannerType.PDM
        self.cycles_since_switch = 100
        self.last_scenario = ScenarioType.UNKNOWN
        self.scenario_persistence = 0
        self.switch_count = 0
        logger.info("Focused LeakyDDM switcher reset")