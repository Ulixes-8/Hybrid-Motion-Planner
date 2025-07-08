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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_planner_with_real_data():
    """Test hybrid planner with real nuPlan data."""
    
    # Initialize Hydra
    config_dir = os.path.join(os.environ['HOME'], 'chimera/chimera/config')
    
    with initialize_config_dir(config_dir=config_dir):
        # Load planner config
        cfg = compose(config_name="planner/hybrid_planner")
        planner_cfg = cfg.planner if 'planner' in cfg else cfg
        
        # Find database files
        import glob
        
        # Try mini split first (smaller, faster)
        db_pattern = os.path.join(os.environ['NUPLAN_DATA_ROOT'], 'nuplan-v1.1/mini/*.db')
        db_files = glob.glob(db_pattern)
        
        if not db_files:
            # Try val split
            db_pattern = os.path.join(os.environ['NUPLAN_DATA_ROOT'], 'nuplan-v1.1/val/*.db')
            db_files = glob.glob(db_pattern)
            
        if not db_files:
            logger.error(f"No .db files found in {db_pattern}")
            return False
            
        logger.info(f"Found {len(db_files)} database files")
        logger.info(f"Using: {db_files[0]}")
        
        # Create scenario builder to load real data
        scenario_builder = NuPlanScenarioBuilder(
            data_root=os.environ['NUPLAN_DATA_ROOT'],
            map_root=os.environ['NUPLAN_MAPS_ROOT'],
            sensor_root=None,
            db_files=[db_files[0]],  # Use just one DB file
            map_version="nuplan-maps-v1.0",
            max_workers=1
        )
        
        # Create a simple scenario filter to get just one scenario
        scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=1,
            limit_total_scenarios=1,
            expand_scenarios=False,
            remove_invalid_goals=True,
            shuffle=False,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            ego_start_speed_threshold=None,
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
                
        # Instantiate planner
        planner = instantiate(planner_cfg)

        # Create planner initialization
        from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

        # Get route roadblock ids from scenario
        route_roadblock_ids = scenario.get_route_roadblock_ids()

        # Create initialization
        initialization = PlannerInitialization(
            route_roadblock_ids=route_roadblock_ids,
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )

        planner.initialize(initialization)

        # Test dual-frequency execution
        logger.info("\nTesting dual-frequency execution...")
        logger.info("PDM @ 10Hz, Diffusion @ 2Hz")
        logger.info("-" * 60)

        pdm_runs = 0
        diffusion_runs = 0

        # Important: Start from a later iteration to ensure we have enough history
        # The Diffusion planner needs 2 seconds of history (21 steps at 0.1s)
        start_iteration = 30  # Start from iteration 30 to have enough history
        
        # Run through some iterations
        for i in range(start_iteration, start_iteration + 10):  # Test 10 iterations
            start_time = time.time()
            
            # Create planner input from scenario
            from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
            from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
            from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
            
            # Get current ego state
            ego_state = scenario.get_ego_state_at_iteration(i)
            
            # Build history - the Diffusion planner's data processor will extract what it needs
            # We need at least 21 past states for the diffusion planner
            history_buffer_size = 25  # A bit more than 21 to be safe
            
            # Collect historical states going back in time
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
            
            # Create iteration
            iteration = SimulationIteration(time_point=ego_state.time_point, index=i)
            
            # Create planner input
            planner_input = PlannerInput(
                iteration=iteration,
                history=history_buffer,
                traffic_light_data=scenario.get_traffic_light_status_at_iteration(i)
            )
            
            # Track execution
            pdm_before = planner.last_pdm_time
            diffusion_before = planner.last_diffusion_time
            
            # Compute trajectory
            try:
                trajectory = planner.compute_planner_trajectory(planner_input)
                
                # Check which planners ran
                pdm_ran = planner.last_pdm_time > pdm_before
                diffusion_ran = planner.last_diffusion_time > diffusion_before
                
                if pdm_ran:
                    pdm_runs += 1
                if diffusion_ran:
                    diffusion_runs += 1
                    
                logger.info(
                    f"Iteration {i}: "
                    f"PDM {'✓' if pdm_ran else ' '} "
                    f"Diffusion {'✓' if diffusion_ran else ' '} "
                    f"| Selected: {planner._current_planner}"
                )
                
            except Exception as e:
                logger.error(f"Error at iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                break  # Stop on error to see what happened
                
            # Small delay to simulate real-time execution
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.1 - elapsed)  # Target 10Hz
            time.sleep(sleep_time)

        logger.info("-" * 60)
        logger.info(f"Summary: PDM ran {pdm_runs} times, Diffusion ran {diffusion_runs} times")

        return True

if __name__ == "__main__":
    success = test_hybrid_planner_with_real_data()
    sys.exit(0 if success else 1)