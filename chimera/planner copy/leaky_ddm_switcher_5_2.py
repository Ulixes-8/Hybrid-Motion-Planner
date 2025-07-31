"""
Enhanced Leaky DDM Switcher - Version 6
Incorporates novel metrics for improved switching decisions.
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
class EnhancedSwitcherConfig:
    """Configuration for Enhanced Switcher with novel metrics - extends V5 Conservative Config"""
    # V5 Conservative Oracle parameters - UNCHANGED
    alpha: float = 0.7  # Higher memory to avoid erratic switches
    theta_upper: float = 0.4   # Higher threshold
    theta_lower: float = -0.4  # Symmetric
    
    # V5 Weights - UNCHANGED
    scenario_weight: float = 0.6   # Scenario important but not overwhelming
    score_weight: float = 0.4      # Score validation important
    
    # V5 Confidence - UNCHANGED
    min_confidence: float = 0.6    # Higher minimum
    high_confidence: float = 0.85  # Very high for override
    
    # V5 Override threshold - UNCHANGED
    override_threshold: float = 0.4  # 40% win rate difference for override
    
    # V5 Safety and timing - UNCHANGED
    min_switch_interval: int = 3   # More cycles between switches
    min_score_threshold: float = 0.0  # Accept any score
    score_difference_threshold: float = 0.05  # Lower threshold for score differences
    
    # V5 Validation flags - UNCHANGED
    require_score_validation: bool = True  # Must have good scores to switch
    fallback_to_pdm: bool = True          # PDM as safe default
    
    # V6 ADDITIONS - NEW PARAMETERS ONLY
    enhanced_metrics_weight: float = 0.6  # NEW: weight for novel metrics
    
    # Enhanced metric preferences by scenario
    metric_preferences: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        """Initialize default metric preferences"""
        if self.metric_preferences is None:
            # Which enhanced metrics matter most for each scenario type
            self.metric_preferences = {
                'changing_lane': {
                    'trajectory_efficiency': 0.3,
                    'maneuver_preparedness': 0.4,
                    'social_compliance': 0.3
                },
                'traversing_pickup_dropoff': {
                    'spatial_utilization': 0.5,
                    'trajectory_consistency': 0.3,
                    'social_compliance': 0.2
                },
                'high_magnitude_speed': {
                    'trajectory_efficiency': 0.4,
                    'trajectory_consistency': 0.4,
                    'social_compliance': 0.2
                },
                'starting_left_turn': {
                    'maneuver_preparedness': 0.4,
                    'spatial_utilization': 0.3,
                    'trajectory_consistency': 0.3
                },
                'following_lane_with_lead': {
                    'social_compliance': 0.5,
                    'trajectory_consistency': 0.3,
                    'trajectory_efficiency': 0.2
                }
            }


class EnhancedSwitcher:
    """
    Enhanced switcher that incorporates novel metrics for better decisions.
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
    
    def __init__(self, config: EnhancedSwitcherConfig = None):
        """Initialize enhanced switcher"""
        self.config = config or EnhancedSwitcherConfig()
        
        # State
        self.P = 0.0  # Preference state
        self.current_planner = PlannerType.PDM
        self.cycles_since_switch = 100
        self.total_switches = 0
        
        # Performance tracking
        self.consecutive_poor_scores = 0
        self.scenario_consistency = 0
        self.last_scenario = ScenarioType.UNKNOWN
        
        # Enhanced metrics tracking
        self.metric_history = []
        self.metric_trends = {}
        
        logger.info(
            f"Initialized Enhanced Switcher v6 (Conservative + Metrics): "
            f"α={self.config.alpha}, "
            f"θ=±{self.config.theta_upper}, "
            f"enhanced_weight={self.config.enhanced_metrics_weight}"
        )
    
    def update_and_select(
        self,
        pdm_score: float,
        diffusion_score: float,
        pdm_safety: float,
        diffusion_safety: float,
        pdm_enhanced_metrics: Dict[str, float],
        diffusion_enhanced_metrics: Dict[str, float],
        scenario_result: Optional['ScenarioDetectionResult'] = None,
        last_planner: Optional[PlannerType] = None
    ) -> Tuple[PlannerType, Dict]:
        """
        Select planner using standard and enhanced metrics.
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
        
        # SAFETY CHECK
        if pdm_score < 0.01 and diffusion_score < 0.01:
            logger.warning(f"Both scores extremely low: PDM={pdm_score:.3f}, Diff={diffusion_score:.3f}")
            self.consecutive_poor_scores += 1
            
            return PlannerType.PDM, self._create_metadata(
                pdm_score, diffusion_score, pdm_safety, diffusion_safety,
                pdm_enhanced_metrics, diffusion_enhanced_metrics,
                scenario_type, scenario_confidence, safety_fallback=True
            )
        else:
            self.consecutive_poor_scores = 0
        
        # Calculate enhanced metric differential
        enhanced_drift = self._calculate_enhanced_metric_drift(
            pdm_enhanced_metrics,
            diffusion_enhanced_metrics,
            scenario_type
        )
        
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
                scenario_bias = min(1.0, win_rate_gap * 2.0)
            else:
                scenario_bias = -min(1.0, win_rate_gap * 2.0)
            
            # Apply with confidence and consistency boost
            consistency_boost = min(0.2, self.scenario_consistency * 0.02)
            scenario_drift = scenario_bias * self.config.scenario_weight * (scenario_confidence + consistency_boost)
            
            # HIGH CONFIDENCE OVERRIDE
            if (scenario_confidence >= self.config.high_confidence and 
                win_rate_gap >= self.config.override_threshold and
                self.current_planner != empirical_best and
                self.cycles_since_switch >= self.config.min_switch_interval and
                self.scenario_consistency >= 3):  # Require consistent detection (V5 setting)
                
                # Check enhanced metrics support the switch
                if self._enhanced_metrics_support_switch(
                    pdm_enhanced_metrics, diffusion_enhanced_metrics, empirical_best
                ):
                    logger.info(
                        f"Enhanced override to {empirical_best.value} for {scenario_type.value} "
                        f"(gap={win_rate_gap:.2f}, conf={scenario_confidence:.2f}, "
                        f"enhanced_support=True)"
                    )
                    
                    self.current_planner = empirical_best
                    self.cycles_since_switch = 0
                    self.total_switches += 1
                    
                    return empirical_best, self._create_metadata(
                        pdm_score, diffusion_score, pdm_safety, diffusion_safety,
                        pdm_enhanced_metrics, diffusion_enhanced_metrics,
                        scenario_type, scenario_confidence, override=True
                    )
        
        # SCORE-BASED DRIFT
        score_diff = diffusion_score - pdm_score
        score_drift = self.config.score_weight * np.tanh(score_diff * 5.0)
        
        # SAFETY PENALTY
        safety_penalty = 0.0
        if diffusion_safety < 0.5 and self.current_planner == PlannerType.DIFFUSION:
            safety_penalty = -0.2
        elif pdm_safety < 0.5 and self.current_planner == PlannerType.PDM:
            safety_penalty = 0.2
        
        # COMBINE ALL EVIDENCE
        total_drift = (1 - self.config.alpha) * (
            scenario_drift + 
            score_drift + 
            safety_penalty + 
            enhanced_drift * self.config.enhanced_metrics_weight
        )
        
        # Update preference with memory
        self.P = self.config.alpha * self.P + total_drift
        self.P = np.clip(self.P, -1.5, 1.5)
        
        # Track metric trends
        self._update_metric_trends(pdm_enhanced_metrics, diffusion_enhanced_metrics)
        
        # Check minimum switch interval
        if self.cycles_since_switch < self.config.min_switch_interval:
            return self.current_planner, self._create_metadata(
                pdm_score, diffusion_score, pdm_safety, diffusion_safety,
                pdm_enhanced_metrics, diffusion_enhanced_metrics,
                scenario_type, scenario_confidence, lockout=True
            )
        
        # Make decision
        new_planner = self._make_decision()
        
        # Validate switch
        if new_planner != self.current_planner:
            if self._validate_switch_enhanced(
                new_planner, pdm_score, diffusion_score, 
                pdm_safety, diffusion_safety,
                pdm_enhanced_metrics, diffusion_enhanced_metrics
            ):
                logger.info(
                    f"Enhanced switch: {self.current_planner.value} → {new_planner.value} | "
                    f"Scenario: {scenario_type.value} | P={self.P:.3f} | "
                    f"Enhanced drift: {enhanced_drift:.3f}"
                )
                self.current_planner = new_planner
                self.cycles_since_switch = 0
                self.total_switches += 1
            else:
                logger.debug(f"Switch to {new_planner.value} rejected by enhanced validation")
        
        return self.current_planner, self._create_metadata(
            pdm_score, diffusion_score, pdm_safety, diffusion_safety,
            pdm_enhanced_metrics, diffusion_enhanced_metrics,
            scenario_type, scenario_confidence
        )
    
    def _calculate_enhanced_metric_drift(
        self,
        pdm_metrics: Dict[str, float],
        diffusion_metrics: Dict[str, float],
        scenario_type: ScenarioType
    ) -> float:
        """
        Calculate drift based on enhanced metrics.
        Positive values favor Diffusion, negative favor PDM.
        """
        if not pdm_metrics or not diffusion_metrics:
            return 0.0
        
        # Get scenario-specific preferences
        preferences = self.config.metric_preferences.get(
            scenario_type.value,
            {
                'trajectory_efficiency': 0.2,
                'social_compliance': 0.2,
                'spatial_utilization': 0.2,
                'trajectory_consistency': 0.2,
                'maneuver_preparedness': 0.2
            }
        )
        
        # Calculate weighted difference for each metric
        drift = 0.0
        for metric, weight in preferences.items():
            pdm_value = pdm_metrics.get(metric, 0.5)
            diffusion_value = diffusion_metrics.get(metric, 0.5)
            
            # Difference favoring diffusion
            metric_diff = diffusion_value - pdm_value
            drift += weight * metric_diff
        
        # Apply scenario-specific interpretation
        if scenario_type in [ScenarioType.CHANGING_LANE, ScenarioType.TRAVERSING_PICKUP_DROPOFF]:
            # For these scenarios, spatial utilization is extra important
            spatial_diff = diffusion_metrics.get('spatial_utilization', 0.5) - pdm_metrics.get('spatial_utilization', 0.5)
            drift += 0.2 * spatial_diff
        
        elif scenario_type in [ScenarioType.FOLLOWING_LANE_WITH_LEAD, ScenarioType.HIGH_MAGNITUDE_SPEED]:
            # For these scenarios, consistency and social compliance matter more
            consistency_diff = pdm_metrics.get('trajectory_consistency', 0.5) - diffusion_metrics.get('trajectory_consistency', 0.5)
            drift -= 0.2 * consistency_diff  # Negative because we favor PDM here
        
        return np.tanh(drift * 3.0)  # Normalize to reasonable range
    
    def _enhanced_metrics_support_switch(
        self,
        pdm_metrics: Dict[str, float],
        diffusion_metrics: Dict[str, float],
        target_planner: PlannerType
    ) -> bool:
        """
        Check if enhanced metrics support switching to target planner.
        """
        if not pdm_metrics or not diffusion_metrics:
            return True  # Don't block switch if metrics unavailable
        
        # Calculate overall enhanced metric advantage
        pdm_total = pdm_metrics.get('weighted_total', 0.5)
        diffusion_total = diffusion_metrics.get('weighted_total', 0.5)
        
        if target_planner == PlannerType.DIFFUSION:
            # Check if diffusion has better enhanced metrics
            return diffusion_total > pdm_total - 0.1  # Small tolerance
        else:
            # Check if PDM has better enhanced metrics
            return pdm_total > diffusion_total - 0.1
    
    def _update_metric_trends(
        self,
        pdm_metrics: Dict[str, float],
        diffusion_metrics: Dict[str, float]
    ):
        """Track metric trends over time for analysis."""
        if pdm_metrics and diffusion_metrics:
            self.metric_history.append({
                'pdm': pdm_metrics.copy(),
                'diffusion': diffusion_metrics.copy(),
                'planner': self.current_planner.value
            })
            
            # Keep only recent history
            if len(self.metric_history) > 50:
                self.metric_history.pop(0)
    
    def _validate_switch_enhanced(
        self,
        new_planner: PlannerType,
        pdm_score: float,
        diffusion_score: float,
        pdm_safety: float,
        diffusion_safety: float,
        pdm_metrics: Dict[str, float],
        diffusion_metrics: Dict[str, float]
    ) -> bool:
        """Enhanced validation including novel metrics."""
        if not self.config.require_score_validation:
            return True
        
        # Safety checks
        if new_planner == PlannerType.DIFFUSION and diffusion_safety < 0.1:
            return False
        
        # Score difference checks
        if new_planner == PlannerType.DIFFUSION and pdm_score - diffusion_score > 0.4:
            # Check if enhanced metrics compensate
            if pdm_metrics and diffusion_metrics:
                diffusion_enhanced = diffusion_metrics.get('weighted_total', 0.5)
                pdm_enhanced = pdm_metrics.get('weighted_total', 0.5)
                if diffusion_enhanced - pdm_enhanced < 0.2:
                    return False
        
        return True
    
    def _make_decision(self) -> PlannerType:
        """Make decision based on preference state."""
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
        pdm_safety: float,
        diffusion_safety: float,
        pdm_enhanced: Dict[str, float],
        diffusion_enhanced: Dict[str, float],
        scenario_type: ScenarioType,
        scenario_confidence: float,
        **flags
    ) -> Dict:
        """Create comprehensive metadata."""
        win_rates = self.EMPIRICAL_WIN_RATES.get(scenario_type, {'pdm': 0.5, 'diffusion': 0.5})
        empirical_best = PlannerType.PDM if win_rates['pdm'] > win_rates['diffusion'] else PlannerType.DIFFUSION
        
        return {
            'P': self.P,
            'pdm_score': pdm_score,
            'diffusion_score': diffusion_score,
            'pdm_safety': pdm_safety,
            'diffusion_safety': diffusion_safety,
            'pdm_enhanced_total': pdm_enhanced.get('weighted_total', 0.5) if pdm_enhanced else 0.5,
            'diffusion_enhanced_total': diffusion_enhanced.get('weighted_total', 0.5) if diffusion_enhanced else 0.5,
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
        """Reset switcher state."""
        self.P = 0.0
        self.current_planner = PlannerType.PDM
        self.cycles_since_switch = 100
        self.total_switches = 0
        self.consecutive_poor_scores = 0
        self.scenario_consistency = 0
        self.last_scenario = ScenarioType.UNKNOWN
        self.metric_history = []
        self.metric_trends = {}
        logger.info("Enhanced Switcher reset")