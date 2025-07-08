#!/usr/bin/env python3
import sys
import os

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

def test_hybrid_planner_instantiation():
    """Test if we can instantiate the hybrid planner."""
    
    # Initialize Hydra with our config directory
    config_dir = os.path.join(os.environ['HOME'], 'chimera/chimera/config')
    
    # Use initialize_config_dir without version_base
    with initialize_config_dir(config_dir=config_dir):
        # Load the planner config
        cfg = compose(config_name="planner/hybrid_planner")
        
        logger.info("Config loaded successfully")
        logger.info(f"Config structure: {cfg}")
        
        try:
            # The config has a 'planner' key at the top level
            planner_cfg = cfg.planner if 'planner' in cfg else cfg
            
            # Try to instantiate the hybrid planner
            planner = instantiate(planner_cfg)
            logger.info(f"✓ Successfully instantiated {planner.name()}")
            
            # Test that both sub-planners exist
            assert hasattr(planner, 'pdm_planner'), "PDM planner not found"
            assert hasattr(planner, 'diffusion_planner'), "Diffusion planner not found"
            logger.info("✓ Both sub-planners found")
            
            return True
        except Exception as e:
            logger.error(f"✗ Failed to instantiate planner: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_hybrid_planner_instantiation()
    sys.exit(0 if success else 1)