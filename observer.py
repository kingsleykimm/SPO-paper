from acme.utils.observers import EnvLoopObserver
from typing import Dict, Union, List, Tuple
import math
import dm_env
import numpy as np


class ContObserver(EnvLoopObserver):
    """Observer that collects angle information for continuous Mujoco trajectories"""
    def __init__(self):
        self.observations = None
    def observe_first(self, env : dm_env.Environment, timestep: dm_env.TimeStep, action: np.ndarray) -> None:
        self.observations = List[Dict[str, float]]
        # timestep contains all the information
        current_timestep = Dict[str, float]
        observation = timestep.observation
        # we need the angles between trajectories as well as radii
        info = env.get_info()
        x_pos = info["x_position"]
        y_pos = info["y_position"]
        radius = math.sqrt(math.pow(x_pos, 2) + math.pow(y_pos, 2))
        current_timestep["radius"] = radius
        self.observations = [current_timestep]
    def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
        current_timestep = Dict[str, float]
        current_timestep = Dict[str, float]
        observation = timestep.observation
        # we need the angles between trajectories as well as radii
        info = env.get_info()
        x_pos = info["x_position"]
        y_pos = info["y_position"]
        radius = math.sqrt(math.pow(x_pos, 2) + math.pow(y_pos, 2))
        current_timestep["radius"] = radius
        self.observations.append(current_timestep)
    def get_metrics(self) -> Dict[str, float]:
        return {
            'episode_observations': self.observations
        }
    def get_episode_obs(self) -> List[Dict[str, float]]:
        return self.observations