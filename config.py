"""Custom SPO Config"""
from absl import app, flags
from continuous import SAC
from typing_extensions import Protocol
from spo import SPORunner
FLAGS = flags.FLAGS

flags.DEFINE_integer('iterations', 100, 'Number of Loops of SPO')
flags.DEFINE_integer('queue_size', 10, 'Size of queue to hold trajectories')
flags.DEFINE_string('env_name', 'gym:Ant-v3', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('learning_rate', 3e-5, 'Learning rate of agent')
flags.DEFINE_integer('min_replay_size', 10000, 'Minimum replay buffer size')
flags.DEFINE_integer('max_replay_size', 1000000, 'Maximum replay buffer size')
flags.DEFINE_string('preference_function', 'intransitive', 'Preference Function to use')

class SPOConfig():
    def __init__(self):
        """
        Agent: contains most of the information for the RL agent, SAC, all the config will be handled on it's end
        
        """
        self.agent = SAC(FLAGS.env_name, FLAGS.min_replay_size, FLAGS.max_replay_size, FLAGS.learning_rate, FLAGS.seed, FLAGS.num_steps,
                            FLAGS.preference_function)
        self.iterations = FLAGS.iterations
        self.queue_size = FLAGS.queue_size
        self.preference_function = self.agent.get_preference_function()
# Plan for the SPO Config
# Things needed that are not given in Experiment Config:
# Num of iterations, Queue size, Preference_Function, 

def main(_):
    config = SPOConfig()
    runner = SPORunner(config)
    runner.run(runner.experiment)



if __name__ =='__main__':
    app.run(main)

