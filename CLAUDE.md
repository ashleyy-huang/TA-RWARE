# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**TA-RWARE** (Task-Assignment Multi-Robot Warehouse) is a Gymnasium-compatible multi-agent reinforcement learning environment simulating a heterogeneous robot warehouse team:
- **AGVs** (Automated Guided Vehicles): transport shelves to/from goal locations
- **Pickers**: human-like agents that load/unload shelves from AGVs
- Objective: maximize order fulfillment (requested shelves delivered per hour)

## Installation

```bash
pip install -e .
```

Key dependencies: `numpy`, `gymnasium`, `pyglet==1.5.11`, `networkx==3.2.1`, `pyastar2d` (custom fork).

## Conda environment

All scripts must run via the project's conda env python:

- Python: `/mnt/sda/home/r147250250916/.conda/envs/tarware/bin/python`
- When invoking from outside the TA-RWARE root, set `PYTHONPATH=/mnt/sda/home/r147250250916/research/MARL/TA-RWARE`.
- Why: only this env has `gymnasium`, `tarware`, `pyastar2d`, etc. installed.

## Running

```bash
# Run heuristic baseline
/mnt/sda/home/r147250250916/.conda/envs/tarware/bin/python \
    scripts/run_heuristic.py --num_episodes=10000 --seed=0 --render

# Merge parallel heuristic shards into a single run (+ optional W&B upload)
/mnt/sda/home/r147250250916/.conda/envs/tarware/bin/python \
    scripts/merge_runs.py --run_glob "runs/heuristic_*_0510_0025"

# Use as a Gymnasium environment
import gymnasium as gym
env = gym.make("tarware-tiny-3agvs-2pickers-partialobs-v1")
```

Environment ID format: `tarware-{size}-{N}agvs-{M}pickers-{obs_type}obs-v1`
- Sizes: `tiny`, `small`, `medium`, `large`, `extralarge`
- Obs types: `global`, `partial`

## Linting

```bash
flake8 tarware/
```

Config in [.flake8](.flake8): max line length 89, max complexity 10.

## Architecture

### Core Components

**[tarware/warehouse.py](tarware/warehouse.py)** — Main `Warehouse(gym.Env)` class. Contains all environment logic: grid layout construction, A\* pathfinding, macro/micro action translation, collision resolution, reward computation, and episode management.

**[tarware/definitions.py](tarware/definitions.py)** — Enums: `AgentType`, `Action`, `Direction`, `RewardType`, `CollisionLayers`. The single source of truth for discrete types used throughout.

**[tarware/spaces/](tarware/spaces/)** — Observation space implementations:
- `MultiAgentBaseObservationSpace`: abstract base with `_VectorWriter` helper
- `MultiAgentGlobalObservationSpace`: all agents see complete state
- `MultiAgentPartialObservationSpace`: AGVs see shelf info + all agent positions; Pickers see only agent positions

**[tarware/heuristic.py](tarware/heuristic.py)** — FIFO-based heuristic baseline. Assigns nearest available agent to requests using a state machine (PICKING → DELIVERING → RETURNING). Use as a benchmark for RL policies.

**[tarware/rendering.py](tarware/rendering.py)** — Pyglet 2D visualization. AGVs render as hexagons, Pickers as diamonds.

**[tarware/utils/wrappers.py](tarware/utils/wrappers.py)** — Gymnasium wrappers: `FlattenAgents`, `DictAgents`, `FlattenSAObservation`, `SquashDones` for compatibility with standard RL libraries.

### Action Hierarchy

Actions are **macro** (high-level, one per step) translated internally to **micro** movements:
- Macro action: select a target location ID (shelf or goal)
- Micro actions: `NOOP`, `LEFT`, `RIGHT`, `FORWARD`, `TOGGLE_LOAD`
- Navigation: A\* pathfinding with collision-aware grids; dynamic path recalculation on conflicts

### Collision System

Four collision layers (`CollisionLayers`): AGVs, Pickers, Shelves, Carried Shelves. Conflict resolution uses graph algorithms (`networkx`) to detect cycles and resolve deadlocks. Stuck agents get exponential backoff and forced path recalculation.

### Reward Structure

- AGV: +1.0 for delivering a requested shelf to goal
- Picker: +0.1 for assisting load/unload
- All agents: −0.001 per step (encourages efficiency)

### Environment Registration

[tarware/\_\_init\_\_.py](tarware/__init__.py) registers all variant combinations with Gymnasium at import time. To add a new size variant, define its parameters in the `SIZE_PARAMS` dict there and add a corresponding registration call.
