#!/bin/bash
# Setup environment for Chimera hybrid planner

export CHIMERA_ROOT="$HOME/chimera"
export PYTHONPATH="$CHIMERA_ROOT:$CHIMERA_ROOT/nuplan-devkit:$CHIMERA_ROOT/tuplan_garage:$CHIMERA_ROOT/diffusion_planner:$PYTHONPATH"

# NuPlan environment variables
export NUPLAN_DEVKIT_ROOT="$HOME/chimera/nuplan-devkit"
export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"

echo "Chimera environment configured!"
echo "PYTHONPATH includes:"
echo "  - $CHIMERA_ROOT"
echo "  - $CHIMERA_ROOT/nuplan-devkit"
echo "  - $CHIMERA_ROOT/tuplan_garage"
echo "  - $CHIMERA_ROOT/diffusion_planner"