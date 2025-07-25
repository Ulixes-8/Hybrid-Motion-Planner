"""
Leaky DDM (Drift-Diffusion Model) Switcher for Hybrid Planning
Now with integrated scenario awareness based on empirical performance data.
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


@dataclass
class LeakyDDMConfig:
    """Configuration for Leaky DDM Switcher"""
    # Leak rate (memory decay)
    alpha: float = 0.85  # exp(-Δt/τ), gives ~10 cycle memory
    
    # Decision thresholds
    theta_upper: float = 0.7   # Threshold to switch to diffusion
    theta_lower: float = -0.7  # Threshold to switch to PDM
    
    # Score mapping parameters
    k_gain: float = 0.07  # Gain for tanh mapping
    score_scale: float = 1.0  # Expected score range [0, score_scale]
    
    # Scenario detection thresholds
    poor_threshold: float = 0.4
    excellent_threshold: float = 0.8
    trapped_cycles: int = 3  # Consecutive poor progress cycles
    routine_cycles: int = 5  # Consecutive excellent cycles
    
    # Bias injection magnitudes
    trapped_bias: float = 0.175  # 0.25 * theta (with theta=0.7)
    routine_bias: float = -0.08
    
    # Progress threshold for trapped detection
    progress_threshold: float = 0.2
    
    # Safety feedback
    veto_penalty: float = -0.1  # Penalty when learner is vetoed
    
    # Scenario-aware parameters (new)
    scenario_scale: float = 0.15  # Scale for scenario bias influence
    scenario_confidence_threshold: float = 0.6  # Min confidence to apply scenario bias
    scenario_decay: float = 0.9  # Decay rate for scenario influence


class LeakyDDMSwitcher:
    """
    Unified switching mechanism using a Leaky Drift-Diffusion Model.
    Now enhanced with scenario-aware biasing based on empirical performance data.
    """
    
    def __init__(self, config: LeakyDDMConfig = None):
        """Initialize the Leaky DDM Switcher"""
        self.config = config or LeakyDDMConfig()
        
        # Core state: preference accumulator
        self.P = -0.5 * self.config.theta_upper  # Conservative start
        
        # Scenario bias accumulator (new)
        self.S = 0.0  # Smoothed scenario influence
        
        # Current planner
        self.current_planner = PlannerType.PDM
        
        # Scenario tracking
        self.consecutive_poor_progress = 0
        self.consecutive_excellent_pdm = 0
        
        # History for diagnostics
        self.history = []
        
        logger.info(f"Initialized LeakyDDM with α={self.config.alpha}, Θ=±{self.config.theta_upper}")
    
    def update_and_select(
        self,
        pdm_score: float,
        diffusion_score: float,
        pdm_progress: Optional[float] = None,
        safety_vetoed: bool = False,
        scenario_bias: Optional[float] = None,  # New parameter
        scenario_confidence: Optional[float] = None  # New parameter
    ) -> Tuple[PlannerType, Dict]:
        """
        Main decision function implementing the Leaky DDM with optional scenario awareness.
        
        Args:
            pdm_score: Total score from PDM planner [0, 1]
            diffusion_score: Total score from diffusion planner [0, 1]
            pdm_progress: Progress metric from PDM (for trapped detection)
            safety_vetoed: Whether the diffusion planner was vetoed by safety
            scenario_bias: Bias based on scenario type [-1=PDM, +1=Diffusion] (optional)
            scenario_confidence: Confidence in scenario detection [0, 1] (optional)
            
        Returns:
            Selected planner type and diagnostic metadata
        """
        # 1. Update scenario bias accumulator if provided
        if scenario_bias is not None and scenario_confidence is not None:
            if scenario_confidence >= self.config.scenario_confidence_threshold:
                # Apply scenario bias with smoothing
                self.S = (self.config.scenario_decay * self.S + 
                         (1 - self.config.scenario_decay) * scenario_bias)
            else:
                # Decay scenario influence when confidence is low
                self.S *= self.config.scenario_decay
        else:
            # No scenario info provided, decay existing influence
            self.S *= self.config.scenario_decay
        
        # 2. Update scenario counters
        self._update_scenario_counters(pdm_score, pdm_progress)
        
        # 3. Compute score difference and map through tanh
        score_diff = diffusion_score - pdm_score
        f_score = np.tanh(self.config.k_gain * score_diff / (0.05 * self.config.score_scale))
        
        # 3a. Special case: both planners performing poorly
        if pdm_score < 0.1 and diffusion_score < 0.1:
            f_score += 0.1
            logger.debug(f"Both planners poor (PDM={pdm_score:.3f}, Diff={diffusion_score:.3f}), adding exploration bias")
        
        # 4. Compute total drift including scenario influence
        base_drift = (1 - self.config.alpha) * f_score
        scenario_drift = 0.0
        
        if scenario_bias is not None and scenario_confidence is not None:
            # Add scenario-aware drift
            scenario_drift = ((1 - self.config.alpha) * self.config.scenario_scale * 
                            self.S * scenario_confidence)
        
        # 5. Leaky integration with both drifts
        self.P = np.clip(
            self.config.alpha * self.P + base_drift + scenario_drift,
            -2 * self.config.theta_upper,
            2 * self.config.theta_upper
        )
        
        # 6. Apply discrete scenario biases (trapped/routine)
        # Modulate these based on scenario context
        modulation = 1.0
        if abs(self.S) > 0.5:  # Strong scenario preference
            if self.S > 0 and self.consecutive_excellent_pdm >= self.config.routine_cycles:
                # Scenario favors diffusion but we're in routine mode - reduce routine bias
                modulation = 0.5
            elif self.S < 0 and self.consecutive_poor_progress >= self.config.trapped_cycles:
                # Scenario favors PDM but we're trapped - reduce trapped bias
                modulation = 0.5
        
        self._apply_scenario_biases(modulation)
        
        # 7. Apply safety feedback
        if safety_vetoed and self.current_planner == PlannerType.DIFFUSION:
            self.P += self.config.veto_penalty
            logger.debug(f"Safety veto penalty applied: P += {self.config.veto_penalty}")
        
        # 8. Make decision with hysteresis
        new_planner = self._make_decision()
        
        # 9. Reset counters if switching
        if new_planner != self.current_planner:
            self.consecutive_poor_progress = 0
            self.consecutive_excellent_pdm = 0
            logger.info(f"Switching from {self.current_planner.value} to {new_planner.value} (P={self.P:.3f}, S={self.S:.3f})")
        
        self.current_planner = new_planner
        
        # 10. Prepare metadata
        metadata = {
            'P': self.P,
            'S': self.S,
            'f_score': f_score,
            'score_diff': score_diff,
            'pdm_score': pdm_score,
            'diffusion_score': diffusion_score,
            'consecutive_poor': self.consecutive_poor_progress,
            'consecutive_excellent': self.consecutive_excellent_pdm,
            'planner': new_planner.value,
            'scenario_bias': scenario_bias if scenario_bias is not None else 0.0,
            'scenario_confidence': scenario_confidence if scenario_confidence is not None else 0.0,
            'scenario_drift': scenario_drift
        }
        
        # Store history
        self.history.append(metadata)
        if len(self.history) > 1000:  # Keep last 1000 decisions
            self.history.pop(0)
        
        return new_planner, metadata
    
    def _update_scenario_counters(self, pdm_score: float, pdm_progress: Optional[float]):
        """Update counters for scenario detection"""
        # Check for poor performance (trapped) - use score AND progress
        is_poor_performance = pdm_score < self.config.poor_threshold
        is_low_progress = pdm_progress is not None and pdm_progress < self.config.progress_threshold
        
        if is_poor_performance or is_low_progress:
            self.consecutive_poor_progress += 1
        else:
            self.consecutive_poor_progress = 0
        
        # Check for excellent PDM performance
        if pdm_score >= self.config.excellent_threshold:
            self.consecutive_excellent_pdm += 1
        else:
            self.consecutive_excellent_pdm = 0
    
    def _apply_scenario_biases(self, modulation: float = 1.0):
        """Apply contextual biases based on scenario detection"""
        # Trapped scenario - inject positive bias
        if self.consecutive_poor_progress >= self.config.trapped_cycles:
            bias = self.config.trapped_bias * modulation
            self.P += bias
            logger.debug(f"Trapped bias applied: P += {bias:.3f}")
            # Reset counter to avoid repeated injection
            self.consecutive_poor_progress = 0
        
        # Routine scenario - apply negative drift
        if self.consecutive_excellent_pdm >= self.config.routine_cycles:
            bias = self.config.routine_bias * modulation
            self.P += bias
            logger.debug(f"Routine bias applied: P += {bias:.3f}")
            self.consecutive_excellent_pdm = 0
    
    def _make_decision(self) -> PlannerType:
        """Make planner decision with hysteresis"""
        # Current planner is PDM
        if self.current_planner == PlannerType.PDM:
            # Need to exceed upper threshold to switch to diffusion
            if self.P > self.config.theta_upper:
                return PlannerType.DIFFUSION
        # Current planner is diffusion
        else:
            # Need to drop below lower threshold to switch to PDM
            if self.P < self.config.theta_lower:
                return PlannerType.PDM
        
        # Stay with current planner (within hysteresis zone)
        return self.current_planner
    
    def reset(self):
        """Reset the switcher to initial state"""
        self.P = -0.5 * self.config.theta_upper
        self.S = 0.0
        self.current_planner = PlannerType.PDM
        self.consecutive_poor_progress = 0
        self.consecutive_excellent_pdm = 0
        self.history.clear()
        logger.info("LeakyDDM switcher reset")