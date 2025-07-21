"""
Leaky DDM (Drift-Diffusion Model) Switcher for Hybrid Planning

This implements a unified switching mechanism that replaces SAH-Drive's three separate
components with a single continuous accumulator based on drift-diffusion theory.

We keep SAH's counters but express them as bias pulses so we remain a one-variable DDM.
Evidence pulses move P ≈ 0.30 per red/blue frame; trapped/routine pulses move P ≈ 0.09/0.245;
STDP weight contributes 20% to the decision but decays faster to avoid pinning.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Deque
from dataclasses import dataclass
from enum import Enum
from collections import deque
import pandas as pd

logger = logging.getLogger(__name__)


class PlannerType(Enum):
    """Planner type enumeration"""
    PDM = "pdm"
    DIFFUSION = "diffusion"


@dataclass
class LeakyDDMConfig:
    """Configuration for Leaky DDM Switcher"""
    # Leak rates
    alpha: float = 0.85  # Main accumulator leak rate
    alpha_w: float = 0.95  # Long-term weight leak rate (faster decay)
    
    # Decision thresholds
    theta: float = 0.7   # Decision boundary (symmetric)
    
    # Evidence and bias scales (separated for independent tuning)
    drift_scale: float = 0.21  # 0.30 * theta - increased for more muscle
    scenario_bias_routine: float = 0.09  # Halved from 0.175 for routine
    scenario_bias_trapped: float = 0.245  # 0.35 * theta - boosted for trapped
    
    # Score buckets (fixed thresholds, no auto-tuning for excellent)
    poor_threshold: float = 0.4
    excellent_threshold: float = 0.85  # Locked at 0.85
    auto_threshold_poor_only: bool = True  # Only auto-tune poor threshold
    
    # Scenario detection (SAH-Drive counters)
    ne_threshold: int = 3  # Reduced from 5 - consecutive excellent PDM frames
    np_threshold: int = 2  # Reduced from 3 - consecutive poor Diffusion frames
    
    # Dwell timer
    min_dwell_frames: int = 6  # Reduced from 10
    
    # Long-term weight influence
    lambda_w: float = 0.2  # Weight of W in decision
    w_learn_rate: float = 0.05  # Increased from 0.02
    w_max: float = 0.5  # Clip W to ±0.5*theta to prevent pinning
    
    # Progress thresholds for trapped detection
    min_progress_m: float = 0.25  # meters
    min_speed_mps: float = 0.1   # m/s
    
    # Clamping
    p_clamp_factor: float = 2.0  # Clamp P to ±(factor * theta)
    
    # Startup
    p_initial_factor: float = -0.2  # Start at -0.2*theta (was -0.5)


class LeakyDDMSwitcher:
    """
    Unified switching mechanism using a Leaky Drift-Diffusion Model.
    
    This version closely matches SAH-Drive's behavior:
    - Buckets scores into poor/ordinary/excellent
    - Uses discrete evidence pulses
    - Maintains scenario counters (ne, np)
    - Includes long-term weight W (STDP-like)
    - Enforces dwell time after switches
    """
    
    def __init__(self, config: LeakyDDMConfig = None):
        """Initialize the Leaky DDM Switcher"""
        self.config = config or LeakyDDMConfig()
        
        # Core accumulators
        self.P = self.config.p_initial_factor * self.config.theta  # Less conservative start
        self.W = 0.0  # Long-term weight (neutral start)
        
        # Current planner
        self.current_planner = PlannerType.PDM
        
        # SAH-style counters
        self.ne = 0  # Consecutive excellent PDM
        self.np = 0  # Consecutive poor Diffusion
        
        # Dwell timer (guard against zero)
        self.frames_since_switch = max(1, self.config.min_dwell_frames)
        
        # History for diagnostics (using deque for O(1) operations)
        self.history: Deque[Dict] = deque(maxlen=1000)
        
        # Score tracking for auto-thresholding
        self.score_history: Deque[Tuple[float, float]] = deque(maxlen=500)
        self.frames_since_threshold_update = 0
        
        logger.info(f"Initialized LeakyDDM with θ={self.config.theta}, α={self.config.alpha}, α_w={self.config.alpha_w}")
        logger.info(f"Evidence scale: {self.config.drift_scale:.3f}, Routine bias: {self.config.scenario_bias_routine:.3f}, "
                   f"Trapped bias: {self.config.scenario_bias_trapped:.3f}")
    
    def update_and_select(
        self,
        pdm_score: float,
        diffusion_score: float,
        ego_speed_mps: float = 1.0,
        progress_m: Optional[float] = None,
        diffusion_is_valid: bool = True,
        safety_vetoed: bool = False
    ) -> Tuple[PlannerType, Dict]:
        """
        Main decision function implementing the Leaky DDM with SAH-Drive behavior.
        
        Args:
            pdm_score: Total score from PDM planner [0, 1]
            diffusion_score: Total score from diffusion planner [0, 1]
            ego_speed_mps: Current ego speed in m/s
            progress_m: Progress in meters (for trapped detection)
            diffusion_is_valid: Whether diffusion trajectory passes validity checks
            safety_vetoed: Whether the diffusion planner was vetoed by safety
            
        Returns:
            Selected planner type and diagnostic metadata
        """
        # 0. Update score history and auto-threshold if enabled
        self.score_history.append((pdm_score, diffusion_score))
        if self.config.auto_threshold_poor_only:
            self._update_poor_threshold()
        
        # 1. Bucket the scores (SAH-style)
        pdm_bucket = self._bucket_score(pdm_score)
        diffusion_bucket = self._bucket_score(diffusion_score)
        
        # 2. Update scenario counters
        self._update_counters(pdm_bucket, diffusion_bucket, ego_speed_mps, progress_m, diffusion_is_valid)
        
        # 3. Compute discrete evidence pulse
        if pdm_bucket == 'poor' and diffusion_bucket != 'poor':
            evidence = 1.0  # Strong evidence for diffusion
        elif diffusion_bucket == 'poor' and pdm_bucket != 'poor':
            evidence = -1.0  # Strong evidence for PDM
        elif pdm_bucket == 'excellent' and diffusion_bucket == 'ordinary':
            evidence = -0.5  # Moderate evidence for PDM
        elif diffusion_bucket == 'excellent' and pdm_bucket == 'ordinary':
            evidence = 0.5  # Moderate evidence for diffusion
        else:
            evidence = 0.0  # No clear winner
        
        # 4. Update main accumulator P with leaky integration
        drift = self.config.drift_scale * evidence  # Use increased drift scale
        self.P = self.config.alpha * self.P + (1 - self.config.alpha) * drift
        
        # 5. Apply scenario biases (after leaky integration)
        if self.ne >= self.config.ne_threshold:
            # Routine scenario - push toward PDM (reduced bias)
            self.P -= self.config.scenario_bias_routine
            self.ne = 0  # Reset counter
            logger.debug(f"Routine bias applied: P -= {self.config.scenario_bias_routine}")
            
        if self.np >= self.config.np_threshold:
            # Trapped scenario - push toward diffusion (boosted bias)
            self.P += self.config.scenario_bias_trapped
            self.np = 0  # Reset counter
            logger.debug(f"Trapped bias applied: P += {self.config.scenario_bias_trapped}")
        
        # 6. Update long-term weight W (bidirectional learning with faster decay)
        # W learns toward winner and away from loser
        if self.current_planner == PlannerType.DIFFUSION:
            w_target = self.config.theta if diffusion_score > pdm_score else -self.config.theta
        else:
            w_target = -self.config.theta if pdm_score > diffusion_score else self.config.theta
        
        self.W += self.config.w_learn_rate * w_target
        self.W *= self.config.alpha_w  # Faster decay
        # Clip W to prevent pinning
        w_max = self.config.w_max * self.config.theta
        self.W = np.clip(self.W, -w_max, w_max)
        
        # 7. Safety veto - reduced penalty (evidence already handles poor score)
        if safety_vetoed and self.current_planner == PlannerType.DIFFUSION:
            self.P -= 0.5 * self.config.scenario_bias_routine  # Very small penalty
            logger.debug(f"Safety veto penalty applied")
        
        # 8. Clamp P after all updates
        p_max = self.config.p_clamp_factor * self.config.theta
        self.P = np.clip(self.P, -p_max, p_max)
        
        # 9. Make decision with dwell timer
        self.frames_since_switch += 1
        
        if self.frames_since_switch >= max(1, self.config.min_dwell_frames):
            # Combine P and W for decision
            decision_value = self.P + self.config.lambda_w * self.W
            
            new_planner = self._make_decision(decision_value)
            
            if new_planner != self.current_planner:
                logger.info(f"Switching from {self.current_planner.value} to {new_planner.value} "
                          f"(P={self.P:.3f}, W={self.W:.3f}, combined={decision_value:.3f})")
                self.current_planner = new_planner
                self.frames_since_switch = 0
                # Reset counters on switch
                self.ne = 0
                self.np = 0
        else:
            new_planner = self.current_planner
        
        # 10. Prepare metadata
        metadata = {
            'P': self.P,
            'W': self.W,
            'ne': self.ne,
            'np': self.np,
            'evidence': evidence,
            'drift': drift,
            'pdm_bucket': pdm_bucket,
            'diffusion_bucket': diffusion_bucket,
            'pdm_score': pdm_score,
            'diffusion_score': diffusion_score,
            'frames_since_switch': self.frames_since_switch,
            'planner': new_planner.value,
            'poor_threshold': self.config.poor_threshold,
            'excellent_threshold': self.config.excellent_threshold,
            'decision_value': self.P + self.config.lambda_w * self.W
        }
        
        # Store history
        self.history.append(metadata)
        
        return new_planner, metadata
    
    def _bucket_score(self, score: float) -> str:
        """Bucket score into poor/ordinary/excellent (SAH-style)"""
        if score < self.config.poor_threshold:
            return 'poor'
        elif score >= self.config.excellent_threshold:
            return 'excellent'
        else:
            return 'ordinary'
    
    def _update_counters(self, pdm_bucket: str, diffusion_bucket: str, 
                        ego_speed_mps: float, progress_m: Optional[float],
                        diffusion_is_valid: bool):
        """Update SAH-style scenario counters"""
        # Update ne (consecutive excellent PDM)
        if pdm_bucket == 'excellent':
            self.ne += 1
        else:
            self.ne = 0
        
        # Update np (consecutive poor Diffusion OR trapped)
        # Only count poor diffusion when trajectory is valid
        diffusion_is_poor = diffusion_bucket == 'poor' and diffusion_is_valid
        
        # Require BOTH low speed AND low progress for trapped
        is_trapped = (
            ego_speed_mps < self.config.min_speed_mps and
            (progress_m is not None and progress_m < self.config.min_progress_m)
        )
        
        if diffusion_is_poor or is_trapped:
            self.np += 1
        else:
            self.np = 0
    
    def _update_poor_threshold(self):
        """Update only poor threshold based on rolling percentiles"""
        self.frames_since_threshold_update += 1
        
        # Update every 50 frames (5 seconds at 10Hz)
        if self.frames_since_threshold_update >= 50 and len(self.score_history) >= 100:
            pdm_scores = [s[0] for s in self.score_history]
            diff_scores = [s[1] for s in self.score_history]
            all_scores = pdm_scores + diff_scores
            
            # Only update poor threshold (30th percentile)
            self.config.poor_threshold = float(np.percentile(all_scores, 30))
            # Keep excellent threshold fixed at 0.85
            
            self.frames_since_threshold_update = 0
            logger.debug(f"Updated poor threshold: {self.config.poor_threshold:.3f}, "
                        f"excellent threshold fixed at: {self.config.excellent_threshold:.3f}")
    
    def _make_decision(self, decision_value: float) -> PlannerType:
        """Make planner decision with hysteresis"""
        if self.current_planner == PlannerType.PDM:
            # Need to exceed +theta to switch to diffusion
            if decision_value > self.config.theta:
                return PlannerType.DIFFUSION
        else:
            # Need to drop below -theta to switch to PDM
            if decision_value < -self.config.theta:
                return PlannerType.PDM
        
        return self.current_planner
    
    def debug_dump(self, n: int = 100) -> pd.DataFrame:
        """Export last N history entries as DataFrame for debugging"""
        if not self.history:
            return pd.DataFrame()
        
        # Get last n entries
        entries = list(self.history)[-n:]
        return pd.DataFrame(entries)
    
    def reset(self):
        """Reset the switcher to initial state"""
        self.P = self.config.p_initial_factor * self.config.theta
        self.W = 0.0
        self.current_planner = PlannerType.PDM
        self.ne = 0
        self.np = 0
        self.frames_since_switch = max(1, self.config.min_dwell_frames)
        self.history.clear()
        self.score_history.clear()
        self.frames_since_threshold_update = 0
        logger.info("LeakyDDM switcher reset")