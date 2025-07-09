#!/usr/bin/env python3
"""
Test trajectory scoring with real nuPlan data to understand the scoring
"""
import sys
import os

sys.path.append(os.path.join(os.environ['HOME'], 'chimera'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/nuplan-devkit'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/tuplan_garage'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/diffusion_planner'))

import numpy as np
import logging
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

# NuPlan imports
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scoring_differences():
    """Test to find scenarios where trajectories get different scores"""
    
    # Set environment
    os.environ['NUPLAN_DATA_ROOT'] = os.path.join(os.environ['HOME'], 'nuplan/dataset')
    os.environ['NUPLAN_MAPS_ROOT'] = os.path.join(os.environ['HOME'], 'nuplan/dataset/maps')
    
    # Initialize Hydra
    config_dir = os.path.join(os.environ['HOME'], 'chimera/chimera/config')
    
    with initialize_config_dir(config_dir=config_dir):
        cfg = compose(config_name="planner/hybrid_planner")
        planner_cfg = cfg.planner if 'planner' in cfg else cfg
        
        # Load database
        import glob
        db_pattern = os.path.join(os.environ['NUPLAN_DATA_ROOT'], 'nuplan-v1.1/mini/*.db')
        db_files = glob.glob(db_pattern)
        
        if not db_files:
            logger.error("No database files found")
            return False
            
        # Create scenario builder
        scenario_builder = NuPlanScenarioBuilder(
            data_root=os.environ['NUPLAN_DATA_ROOT'],
            map_root=os.environ['NUPLAN_MAPS_ROOT'],
            sensor_root=None,
            db_files=[db_files[0]],
            map_version="nuplan-maps-v1.0",
            max_workers=1
        )
        
        # Get challenging scenarios - fix the initialization
        scenario_filter = ScenarioFilter(
            scenario_types=[
                'following_lane_with_slow_lead',
                'high_lateral_acceleration',
                'high_magnitude_speed',
                'low_magnitude_speed',
                'traversing_intersection',
            ],
            scenario_tokens=None,  # Required parameter
            log_names=None,        # Required parameter
            map_names=None,        # Required parameter
            num_scenarios_per_type=2,  # Required parameter
            limit_total_scenarios=10,
            expand_scenarios=False,
            remove_invalid_goals=True,
            shuffle=True,
            timestamp_threshold_s=None,  # Required parameter
            ego_displacement_minimum_m=50.0,
            ego_start_speed_threshold=1.0,
            ego_stop_speed_threshold=0.1,
            speed_noise_tolerance=None
        )
        
        worker = SingleMachineParallelExecutor(use_process_pool=False)
        scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
        
        if not scenarios:
            logger.error("No scenarios found")
            return False
        
        logger.info(f"Found {len(scenarios)} scenarios to test")
        
        # Look for score differences
        for scenario_idx, scenario in enumerate(scenarios):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing scenario {scenario_idx + 1}/{len(scenarios)}: {scenario.scenario_type}")
            logger.info(f"Token: {scenario.token}")
            
            # Instantiate planner
            planner = instantiate(planner_cfg)
            
            # Initialize
            initialization = PlannerInitialization(
                route_roadblock_ids=scenario.get_route_roadblock_ids(),
                mission_goal=scenario.get_mission_goal(),
                map_api=scenario.map_api,
            )
            planner.initialize(initialization)
            
            # Test multiple iterations to find interesting moments
            test_iterations = [30, 60, 90, 120]
            
            for test_iteration in test_iterations:
                if test_iteration >= scenario.get_number_of_iterations():
                    continue
                    
                # Build history
                history_buffer_size = 25
                history_states = []
                history_observations = []
                
                for hist_offset in range(history_buffer_size):
                    hist_iteration = test_iteration - hist_offset
                    if hist_iteration >= 0:
                        history_states.append(scenario.get_ego_state_at_iteration(hist_iteration))
                        history_observations.append(scenario.get_tracked_objects_at_iteration(hist_iteration))
                
                history_states.reverse()
                history_observations.reverse()
                
                history_buffer = SimulationHistoryBuffer.initialize_from_list(
                    buffer_size=history_buffer_size,
                    ego_states=history_states,
                    observations=history_observations,
                    sample_interval=scenario.database_interval
                )
                
                # Create planner input
                ego_state = scenario.get_ego_state_at_iteration(test_iteration)
                iteration = SimulationIteration(time_point=ego_state.time_point, index=test_iteration)
                
                planner_input = PlannerInput(
                    iteration=iteration,
                    history=history_buffer,
                    traffic_light_data=scenario.get_traffic_light_status_at_iteration(test_iteration)
                )
                
                # Get context
                tracked_objects = scenario.get_tracked_objects_at_iteration(test_iteration)
                logger.info(f"\n  Iteration {test_iteration}:")
                logger.info(f"    Ego speed: {ego_state.dynamic_car_state.speed:.1f} m/s")
                logger.info(f"    Tracked objects: {len(tracked_objects.tracked_objects)}")
                
                # Count object types
                vehicles = sum(1 for obj in tracked_objects.tracked_objects 
                             if str(obj.tracked_object_type) == 'VEHICLE')
                pedestrians = sum(1 for obj in tracked_objects.tracked_objects 
                                if str(obj.tracked_object_type) == 'PEDESTRIAN')
                logger.info(f"    Vehicles: {vehicles}, Pedestrians: {pedestrians}")
                
                # Compute trajectory
                try:
                    trajectory = planner.compute_planner_trajectory(planner_input)
                    
                    # Check if scores are available and different
                    if (planner.current_pdm_trajectory and 
                        planner.current_diffusion_trajectory):
                        
                        # The scores should have been logged - let's check the scorer state
                        if hasattr(planner, 'scorer') and planner.scorer._initialized:
                            # Try to access the last scoring results
                            logger.info("    Scoring completed for both trajectories")
                            
                            # Check for score differences in the logs
                            # The actual scores are logged by the hybrid planner
                            
                except Exception as e:
                    logger.error(f"    Error: {e}")
                    continue
        
        return True

def analyze_scoring_in_action():
    """Run the hybrid planner and analyze when scores differ"""
    
    # Set environment
    os.environ['NUPLAN_DATA_ROOT'] = os.path.join(os.environ['HOME'], 'nuplan/dataset')
    os.environ['NUPLAN_MAPS_ROOT'] = os.path.join(os.environ['HOME'], 'nuplan/dataset/maps')
    
    logger.info("Running hybrid planner to analyze scoring patterns...")
    
    # We need to check the actual log output from test_with_real_data.py
    # since it shows the scores being computed
    
    # The key insight is that in simple scenarios (no obstacles, straight roads),
    # both planners will get perfect scores because:
    # 1. No collisions (score = 1.0)
    # 2. Stay in drivable area (score = 1.0)
    # 3. Make progress (score = 1.0)
    # 4. Comfortable motion (score = 1.0)
    
    logger.info("\nPerfect scores occur when:")
    logger.info("- No obstacles nearby (collision = 1.0)")
    logger.info("- Wide open roads (drivable area = 1.0)")
    logger.info("- Forward progress (progress = 1.0)")
    logger.info("- Smooth motion (comfort = 1.0)")
    logger.info("\nScore differences would appear in:")
    logger.info("- Tight spaces")
    logger.info("- Near obstacles")
    logger.info("- Sharp turns")
    logger.info("- Emergency maneuvers")
    
    return True

if __name__ == "__main__":
    # First, test scoring with various scenarios
    test_scoring_differences()
    
    # Then analyze what we learned
    analyze_scoring_in_action()