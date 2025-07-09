# Re-export the PDM-based scorer as the unified scorer
from chimera.scoring.pdm_based_trajectory_scorer import (
    PDMBasedTrajectoryScorer as UnifiedTrajectoryScorer,
    TrajectoryScore
)

__all__ = ['UnifiedTrajectoryScorer', 'TrajectoryScore']