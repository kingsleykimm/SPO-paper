from absl import flags
from acme.agents.jax import ppo
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'gym:HalfCheetah-v2', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 50_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 50_000, 'How often to run evaluation.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer('num_distributed_actors', 64,
                     'Number of actors to use in the distributed setting.')



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


def main(_):
    config = build_experiment_config()
    if FLAGS.run_distributed:
        program = experiments.make_distributed_experiment(
            experiment=config, num_actors=FLAGS.num_distributed_actors)
        lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
    else:
        experiments.run_experiment(
            experiment=config,
            eval_every=FLAGS.eval_every,
            num_eval_episodes=FLAGS.evaluation_episodes)


if __name__ == '__main__':
    app.run(main)