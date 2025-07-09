#!/usr/bin/env python3
import sys
import os
import time

# Add paths for all our modules
sys.path.append(os.path.join(os.environ['HOME'], 'chimera'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/nuplan-devkit'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/tuplan_garage'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/diffusion_planner'))

# Set environment variables
os.environ['NUPLAN_DATA_ROOT'] = os.path.join(os.environ['HOME'], 'nuplan/dataset')
os.environ['NUPLAN_MAPS_ROOT'] = os.path.join(os.environ['HOME'], 'nuplan/dataset/maps')
os.environ['NUPLAN_EXP_ROOT'] = os.path.join(os.environ['HOME'], 'nuplan/exp')

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import logging

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

# Configure logging to see scoring details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_hybrid_planner_scoring():
    """Test hybrid planner with scoring enabled."""
    
    # Initialize Hydra
    config_dir = os.path.join(os.environ['HOME'], 'chimera/chimera/config')
    
    with initialize_config_dir(config_dir=config_dir):
        # Load planner config
        cfg = compose(config_name="planner/hybrid_planner")
        planner_cfg = cfg.planner if 'planner' in cfg else cfg
        
        # Find database files
        import glob
        db_pattern = os.path.join(os.environ['NUPLAN_DATA_ROOT'], 'nuplan-v1.1/mini/*.db')
        db_files = glob.glob(db_pattern)
        
        if not db_files:
            db_pattern = os.path.join(os.environ['NUPLAN_DATA_ROOT'], 'nuplan-v1.1/val/*.db')
            db_files = glob.glob(db_pattern)
            
        if not db_files:
            logger.error(f"No .db files found in {db_pattern}")
            return False
            
        logger.info(f"Found {len(db_files)} database files")
        logger.info(f"Using: {db_files[0]}")
        
        # Create scenario builder
        scenario_builder = NuPlanScenarioBuilder(
            data_root=os.environ['NUPLAN_DATA_ROOT'],
            map_root=os.environ['NUPLAN_MAPS_ROOT'],
            sensor_root=None,
            db_files=[db_files[0]],
            map_version="nuplan-maps-v1.0",
            max_workers=1
        )
        
        # Create scenario filter - get a scenario with good diversity
        scenario_filter = ScenarioFilter(
            scenario_types=['lane_following_with_lead', 'following_lane_with_slow_lead'],
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=1,
            limit_total_scenarios=1,
            expand_scenarios=False,
            remove_invalid_goals=True,
            shuffle=False,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=100.0,  # Get scenario with some movement
            ego_start_speed_threshold=5.0,
            ego_stop_speed_threshold=None,
            speed_noise_tolerance=None,
        )
        
        # Get scenarios
        logger.info("Loading scenarios from nuPlan dataset...")
        worker = SingleMachineParallelExecutor(use_process_pool=False)
        scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
        
        if not scenarios:
            logger.error("No scenarios found!")
            return False
            
        scenario = scenarios[0]
        logger.info(f"Using scenario: {scenario.token} from log: {scenario.log_name}")
        logger.info(f"Scenario type: {scenario.scenario_type}")
                
        # Instantiate planner
        planner = instantiate(planner_cfg)
        
        # Initialize planner
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        initialization = PlannerInitialization(
            route_roadblock_ids=route_roadblock_ids,
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )
        planner.initialize(initialization)
        
        # Test scoring over multiple iterations
        logger.info("\nTesting hybrid planner with scoring...")
        logger.info("=" * 80)
        
        # Start from later iteration for history
        start_iteration = 30
        num_test_iterations = 20
        
        # Track planner selections
        planner_selections = {'pdm': 0, 'diffusion': 0}
        score_history = []
        
        for i in range(start_iteration, start_iteration + num_test_iterations):
            # Build history buffer
            history_buffer_size = 25
            history_states = []
            history_observations = []
            
            for hist_offset in range(history_buffer_size):
                hist_iteration = i - hist_offset
                if hist_iteration >= 0:
                    history_states.append(scenario.get_ego_state_at_iteration(hist_iteration))
                    history_observations.append(scenario.get_tracked_objects_at_iteration(hist_iteration))
            
            # Reverse to have oldest first
            history_states.reverse()
            history_observations.reverse()
            
            # Create history buffer
            history_buffer = SimulationHistoryBuffer.initialize_from_list(
                buffer_size=history_buffer_size,
                ego_states=history_states,
                observations=history_observations,
                sample_interval=scenario.database_interval
            )
            
            # Create planner input
            ego_state = scenario.get_ego_state_at_iteration(i)
            iteration = SimulationIteration(time_point=ego_state.time_point, index=i)
            
            planner_input = PlannerInput(
                iteration=iteration,
                history=history_buffer,
                traffic_light_data=scenario.get_traffic_light_status_at_iteration(i)
            )
            
            # Compute trajectory
            try:
                start_time = time.time()
                trajectory = planner.compute_planner_trajectory(planner_input)
                computation_time = time.time() - start_time
                
                # Track which planner was selected
                selected_planner = planner._current_planner
                planner_selections[selected_planner] += 1
                
                # Log iteration summary
                logger.info(
                    f"Iteration {i}: Selected {selected_planner.upper()} planner "
                    f"(computation: {computation_time:.3f}s)"
                )
                
                # If both trajectories are available, show score comparison
                if planner.current_pdm_trajectory and planner.current_diffusion_trajectory:
                    logger.info(f"  Both planners have trajectories available for scoring")
                
            except Exception as e:
                logger.error(f"Error at iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Small delay to simulate real-time
            time.sleep(0.05)
        
        # Summary
        logger.info("=" * 80)
        logger.info("Test Summary:")
        logger.info(f"  Total iterations: {num_test_iterations}")
        logger.info(f"  PDM selected: {planner_selections['pdm']} times "
                   f"({planner_selections['pdm']/num_test_iterations*100:.1f}%)")
        logger.info(f"  Diffusion selected: {planner_selections['diffusion']} times "
                   f"({planner_selections['diffusion']/num_test_iterations*100:.1f}%)")
        
        return True

if __name__ == "__main__":
    success = test_hybrid_planner_scoring()
    sys.exit(0 if success else 1)