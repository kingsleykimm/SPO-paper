from absl import flags
from acme import specs
from acme.agents.jax import normalization
from acme.agents.jax import sac
from acme.agents.jax.sac import builder
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils, observers
import launchpad as lp
import math
from observer import ContObserver

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'gym:Ant-v3', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 50_000, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')

class SAC():
    def __init__(self, config):
        suite, task = config.env_name.split(':', 1)
        environment = helpers.make_environment(suite, task)

        environment_spec = specs.make_environment_spec(environment)
        network_factory = (
            lambda spec: sac.make_networks(spec, hidden_layer_sizes=(256, 256, 256)))
        # Construct the agent.
        config = sac.SACConfig(
            min_replay_size=10000,
            max_replay_size=1000000,
            batch_size=256,
            learning_rate=3e-5,
            # target_entropy=sac.target_entropy_from_env_spec(environment_spec),
            entropy_coefficient=1e-4,
            input_normalization=normalization.NormalizationConfig()) # hyperparameters from paper
        sac_builder = builder.SACBuilder(config) # adam optimizer is configured in builder
        observers = [ContObserver()]
        self.experiment = experiments.ExperimentConfig(
            builder=sac_builder,
            environment_factory=lambda seed: helpers.make_environment(suite, task),
            network_factory=network_factory,
            seed=FLAGS.seed,
            max_num_actor_steps=FLAGS.num_steps,
            observers=observers
            )
    def get_experiment_config(self):
        return self.experiment
    def angular_preference(self, traj_1_angle, traj_2_angle, angle):
        difference = math.fmod(traj_1_angle + angle/2.0 - traj_2_angle, 2 * math.pi)
        return difference < angle/2.0 or difference > 2 * math.pi - angle/2.0
    def distance_preference(self, traj_1_dist, traj_2_dist, dist_threshold):
        if traj_1_dist > dist_threshold and traj_2_dist > dist_threshold:
            return 1.0
        else:
            return 1.0 if traj_1_dist > traj_2_dist else 0.0
    def preference_function(self, traj_1, traj_2):
        return (0.3 * self.distance_preference(traj_1.radius, traj_2.radius, 10.0) + 
                0.7 * self.angular_preference(traj_1.angle, traj_2.angle, math.pi/4))

def ContRM():
    def __init__(self, config):
        suite, task = config.env_name.split(':', 1)
        environment = helpers.make_environment(suite, task)