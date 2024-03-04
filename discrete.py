from absl import flags
from acme.agents.jax import ppo
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

FLAGS = flags.FLAGS



def build_experiment_config():
    """Builds PPO experiment config which can be executed in different ways."""
    # Create an environment, grab the spec, and use it to create networks.
    suite, task = FLAGS.env_name.split(':', 1)

    config = ppo.PPOConfig(
        normalize_advantage=True,
        normalize_value=True,
        learning_rate=1e-4,
        entropy_cost=1e-4,

        obs_normalization_fns_factory=ppo.build_mean_std_normalizer)
    ppo_builder = ppo.PPOBuilder(config)

    layer_sizes = (128, 128) #relu activation comes with it
    return experiments.ExperimentConfig(
        builder=ppo_builder,
        environment_factory=lambda seed: helpers.make_environment(suite, task),
        network_factory=lambda spec: ppo.make_networks(spec, layer_sizes),
        seed=FLAGS.seed,
        max_num_actor_steps=FLAGS.num_steps)


