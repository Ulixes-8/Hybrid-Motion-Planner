# Chimera Planner

A hybrid autonomous vehicle motion planner that intelligently combines rule-based (PDM-Closed) and learning-based (Diffusion Planner) approaches.

## Overview

Chimera leverages the strengths of both planning paradigms:
- **PDM-Closed**: Fast, reliable rule-based planner for normal driving scenarios
- **Diffusion Planner**: Sophisticated learning-based planner for complex scenarios

## Architecture

The hybrid planner implements:
- Dual-frequency execution (PDM @ 10Hz, Diffusion @ 2Hz)
- Intelligent switching based on scenario detection and performance scores
- Unified trajectory scoring using PDM's evaluation metrics

## Dependencies

- [nuPlan DevKit](https://github.com/motional/nuplan-devkit)
- [tuPlan Garage](https://github.com/autonomousvision/tuplan_garage)
- [Diffusion Planner](https://github.com/ZiYunan/Diffusion-Planner)

## Setup

1. Clone this repository
2. Set up the symbolic links to the required repositories
3. Source the environment setup: `source setup_env.sh`

## Status

🚧 Under active development
