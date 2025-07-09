# test_scoring_verification.py
#!/usr/bin/env python3
"""
Verify that scoring actually differentiates between trajectories
"""
import sys
import os

sys.path.append(os.path.join(os.environ['HOME'], 'chimera'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/nuplan-devkit'))
sys.path.append(os.path.join(os.environ['HOME'], 'chimera/tuplan_garage'))

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

# Create two different trajectories
# Good trajectory - straight line
good_states = []
for i in range(40):
    t = i * 0.1
    x = i * 2.0  # 2 m/s forward
    y = 0.0
    heading = 0.0
    good_states.append(StateSE2(x, y, heading))

# Bad trajectory - swerving
bad_states = []
for i in range(40):
    t = i * 0.1
    x = i * 2.0
    y = np.sin(i * 0.5) * 2.0  # Swerving
    heading = 0.0
    bad_states.append(StateSE2(x, y, heading))

print(f"Good trajectory: straight line, final pos: ({good_states[-1].x}, {good_states[-1].y})")
print(f"Bad trajectory: swerving, final pos: ({bad_states[-1].x}, {bad_states[-1].y})")

# The scoring should give different scores for these trajectories