import acme
from acme import specs
from acme.agents.jax import normalization
from acme.agents.jax import sac
from acme.agents.jax.sac import builder
import helpers
from acme.jax import experiments
from acme.utils import lp_utils, observers
import math
from observer import ContObserver
import numpy as np

_VALID_PREFERENCES = ('intransitive', 'maximum_reward', 'noisy', 'non-markovian')

class SAC():
    def __init__(self,
                 env_name : str,
                 min_replay_size : int,
                 max_replay_size: int,
                 learning_rate : float,
                 seed : int,
                 num_steps : int,
                 preference_function : str):
        suite, task = env_name.split(':', 1)
        environment = helpers.make_environment(suite, task)
        self.preference_function = preference_function
        environment_spec = specs.make_environment_spec(environment)
        network_factory = (
            lambda spec: sac.make_networks(spec, hidden_layer_sizes=(256, 256, 256)))
        # Construct the agent.
        sac_config = sac.SACConfig(
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            batch_size=256,
            learning_rate=learning_rate,
            # target_entropy=sac.target_entropy_from_env_spec(environment_spec),
            entropy_coefficient=1e-4, # entropy coefficient should only be this for intransitive preferences
            input_normalization=normalization.NormalizationConfig()) # hyperparameters from paper
        sac_builder = builder.SACBuilder(sac_config) # adam optimizer is configured in builder
        observers = [ContObserver()]
        self.experiment = experiments.ExperimentConfig(
            builder=sac_builder,
            environment_factory=lambda seed: helpers.make_environment(suite, task),
            network_factory=network_factory,
            seed=seed,
            max_num_actor_steps=num_steps,
            observers=observers
            )
    def get_experiment_config(self):
        return self.experiment
    def get_preference_function(self):
        if self.preference_function == 'intransitive':
            return self.intransitive_reward_preference
        elif self.preference_function == 'maximum_reward':
            return self.max_reward_preference
        elif self.preference_function == 'noisy':
            return self.noisy_preference
        # TO-DO implement
        elif self.preference_function == 'non-markovian':
            pass
        raise ValueError('Please input an existing preference function')
    def angular_preference(self, traj_1_angle, traj_2_angle, angle):
        difference = math.fmod(traj_1_angle + angle/2.0 - traj_2_angle, 2 * math.pi)
        return difference < angle/2.0 or difference > 2 * math.pi - angle/2.0
    def distance_preference(self, traj_1_dist, traj_2_dist, dist_threshold):
        if traj_1_dist > dist_threshold and traj_2_dist > dist_threshold:
            return 1.0
        else:
            return 1.0 if traj_1_dist > traj_2_dist else 0.0
    def intransitive_reward_preference(self, traj_1, traj_2):
        # traj_1 and traj_2 are actually just metrics
        
        return (0.3 * self.distance_preference(traj_1["radius"][-1], traj_2["radius"][-1], 10.0) + 
                0.7 * self.angular_preference(traj_1["angle"][-1], traj_2["angle"][-1], math.pi/4))
    def max_reward_preference(self, trajectory_a, trajectory_b):
        # trajectory contains metrics
        timesteps_a = trajectory_a["timestep"]
        timesteps_b = trajectory_b["timestep"]
        reward_a = sum([x.reward for x in timesteps_a[1:]])
        reward_b = sum([x.reward for x in timesteps_b[1:]])
        return 2 * (reward_a > reward_b) - 1
    def noisy_preference(self, trajectory_a, trajectory_b, epsilon):
        return self.max_reward_preference(trajectory_a, trajectory_b) * np.random.binomial(1, epsilon)

def ContRM():
    def __init__(self, config):
        suite, task = config.env_name.split(':', 1)
        environment = helpers.make_environment(suite, task)
        self.batch_size = config.batch_size