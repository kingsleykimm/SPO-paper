from acme.utils.observers import EnvLoopObserver
from typing import Dict, Union, List, Tuple
import math
import dm_env
import numpy as np
import gym


class ContObserver(EnvLoopObserver):
    """Observer that collects angle information for continuous Mujoco trajectories"""
    def __init__(self):
        self.observations = None
    def observe_first(self, env : dm_env.Environment, timestep: dm_env.TimeStep, action: np.ndarray) -> None:
        self.actions = List[np.ndarray]
        self.timesteps = List[dm_env.TimeStep]
        self.radii = List[float]
        self.angles = List[float]
        # self.environments = List[dm_env.Environment]
        # timestep contains all the information
        # current_timestep = Dict[str, float]
        observation = timestep.observation
        # we need the angles between trajectories as well as radii
        x_pos = observation[0]
        y_pos = observation[1]
        # info = env.get_info()
        # x_pos = info["x_position"]
        # y_pos = info["y_position"]
        angle = math.atan2(y_pos, x_pos)
        radius = math.sqrt(math.pow(x_pos, 2) + math.pow(y_pos, 2))
        self.radii.append(radius)
        self.angles.append(angle)
        self.timesteps.append(timestep)
        self.actions.append(action)
        # self.observations = [current_timestep]
    def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
        # current_timestep = Dict[str, float]
        # current_timestep = Dict[str, float]
        observation = timestep.observation # still box from gym
        # we need the angles between trajectories as well as radii
        # constrained from [-1, 1]
        x_pos = observation[0]
        y_pos = observation[1]
        radius = math.sqrt(math.pow(x_pos, 2) + math.pow(y_pos, 2))
        angle = math.atan2(y_pos, x_pos)
        self.radii.append(radius)
        self.angles.append(angle)
        self.timesteps.append(timestep)
        self.actions.append(action)
        # self.environments.append(env)
        # current_timestep["radius"] = radius
        # current_timestep["angle"] = angle
        # self.observations.append(current_timestep)
    def get_metrics(self) -> Dict[str, float]:
        return {
                'radius' : self.radii,
                'angle': self.angles,
                'timestep': self.timesteps,
                'action': self.actions,
        }
    def get_episode_obs(self) -> List[Dict[str, float]]:
        return self.observations

        

