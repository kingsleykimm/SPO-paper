"""Custom SPO Config"""

from continuous import SAC
from typing_extensions import Protocol
from spo import SPORunner
import argparse
# # args = flags.FLAGS

# flags.DEFINE_integer('iterations', 100, 'Number of Loops of SPO')
# flags.DEFINE_integer('queue_size', 10, 'Size of queue to hold trajectories')
# flags.DEFINE_string('env_name', 'gym:Ant-v3', 'What environment to run')
# flags.DEFINE_integer('seed', 0, 'Random seed.')
# flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')
# flags.DEFINE_integer('learning_rate', 3e-5, 'Learning rate of agent')
# flags.DEFINE_integer('min_replay_size', 10000, 'Minimum replay buffer size')
# flags.DEFINE_integer('max_replay_size', 1000000, 'Maximum replay buffer size')
# flags.DEFINE_string('preference_function', 'intransitive', 'Preference Function to use')


class SPOConfig():
    def __init__(self):
        """
        Agent: contains most of the information for the RL agent, SAC, all the config will be handled on it's end
        
        """
        self.agent = SAC(args.env_name, args.min_replay_size, args.max_replay_size, args.learning_rate, args.seed, args.num_steps,
                            args.preference_function)
        self.iterations = args.iterations
        self.queue_size = args.queue_size
        self.preference_function = self.agent.get_preference_function()
# Plan for the SPO Config
# Things needed that are not given in Experiment Config:
# Num of iterations, Queue size, Preference_Function, 

def main(_):
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--iterations', type=int, default=100, help='Number of Loops of SPO')
    parser.add_argument('--queue_size', type=int, default=10, help='Size of queue to hold trajectories')
    parser.add_argument('--env_name', type=str, default='gym:Ant-v3', help='What environment to run')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_steps', type=int, default=1_000_000, help='Number of env steps to run.')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate of agent')
    parser.add_argument('--min_replay_size', type=int, default=10000, help='Minimum replay buffer size')
    parser.add_argument('--max_replay_size', type=int, default=1000000, help='Maximum replay buffer size')
    parser.add_argument('--preference_function', type=str, default='intransitive', help='Preference Function to use')

    args = parser.parse_args()
    config = SPOConfig()
    runner = SPORunner(config)
    runner.run(runner.experiment)



if __name__ =='__main__':
    main()

