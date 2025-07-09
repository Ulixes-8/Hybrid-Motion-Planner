#!/usr/bin/env python3
"""
Debug script to test hybrid planner instantiation and identify issues
"""
import sys
import os
import traceback

# Add paths for all our modules
sys.path.append(os.path.join(os.environ['HOME'], 'chimera'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/nuplan-devkit'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/tuplan_garage'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/diffusion_planner'))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports needed by hybrid planner"""
    try:
        # Test chimera scoring imports
        from chimera.scoring.unified_trajectory_scorer import UnifiedTrajectoryScorer
        logger.info("✓ Chimera scoring imports successful")
        
        # Test PDM imports
        from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
        logger.info("✓ PDM planner imports successful")
        
        # Test Diffusion planner imports
        from diffusion_planner.planner.planner import DiffusionPlanner
        logger.info("✓ Diffusion planner imports successful")
        
        # Test hybrid planner import
        from chimera.planner.hybrid_planner import HybridPlanner
        logger.info("✓ Hybrid planner import successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test config loading"""
    try:
        config_dir = os.path.join(os.environ['HOME'], 'chimera/chimera/config')
        with initialize_config_dir(config_dir=config_dir):
            cfg = compose(config_name="planner/hybrid_planner")
            logger.info("✓ Config loading successful")
            return cfg
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        traceback.print_exc()
        return None

def test_planner_instantiation(cfg):
    """Test planner instantiation"""
    try:
        # The config is already at the top level, no nesting needed
        logger.info(f"Config keys: {list(cfg.keys())}")
        
        planner = instantiate(cfg)
        logger.info(f"✓ Planner instantiation successful: {planner.name()}")
        logger.info(f"  - PDM planner: {type(planner.pdm_planner).__name__}")
        logger.info(f"  - Diffusion planner: {type(planner.diffusion_planner).__name__}")
        return True
    except Exception as e:
        logger.error(f"✗ Planner instantiation failed: {e}")
        traceback.print_exc()
        return False

def main():
    logger.info("Starting Hybrid Planner Debug Test...")
    
    # Test 1: Imports
    if not test_imports():
        logger.error("Import test failed - fix imports first")
        return
    
    # Test 2: Config loading
    cfg = test_config_loading()
    if cfg is None:
        logger.error("Config loading failed - fix config paths")
        return
    
    # Test 3: Planner instantiation
    if not test_planner_instantiation(cfg):
        logger.error("Planner instantiation failed - check component configs")
        return
    
    logger.info("🎉 All tests passed! Hybrid planner is ready for simulation.")

if __name__ == "__main__":
    main()