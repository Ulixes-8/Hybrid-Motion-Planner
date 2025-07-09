#!/usr/bin/env python3
"""
Test scoring system in isolation
"""
import sys
import os

sys.path.append(os.path.join(os.environ['HOME'], 'chimera'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/tuplan_garage'))

import numpy as np
import logging
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scoring():
    try:
        from chimera.scoring.unified_trajectory_scorer import UnifiedTrajectoryScorer
        
        # Create sampling parameters like PDM uses
        proposal_sampling = TrajectorySampling(
            num_poses=40,  # 4s @ 10Hz
            interval_length=0.1
        )
        trajectory_sampling = TrajectorySampling(
            num_poses=80,  # 8s @ 10Hz
            interval_length=0.1
        )
        
        # Instantiate scorer
        scorer = UnifiedTrajectoryScorer(
            proposal_sampling=proposal_sampling,
            trajectory_sampling=trajectory_sampling
        )
        
        logger.info("✓ Scoring system instantiated successfully")
        logger.info(f"  - Proposal sampling: {proposal_sampling.num_poses} poses @ {proposal_sampling.interval_length}s")
        logger.info(f"  - Trajectory sampling: {trajectory_sampling.num_poses} poses @ {trajectory_sampling.interval_length}s")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_scoring()