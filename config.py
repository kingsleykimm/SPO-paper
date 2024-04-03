"""Custom SPO Config"""

from continuous import SAC, VanillaPPO
from typing_extensions import Protocol
from spo import SPORunner
import argparse
import random


class SPOConfig():
    def __init__(self, args):
        """
        Agent: contains most of the information for the RL agent, SAC, all the config will be handled on it's end
        
        """
        self.iterations = args.iterations
        self.queue_size = args.queue_size
        if args.agent == 'sac':
            self.agent = SAC(args.env_name, args.min_replay_size, args.max_replay_size, args.learning_rate, args.seed, args.num_steps,
                            args.preference_function)
            self.preference_function = self.agent.get_preference_function()
        elif args.agent == 'ppo':
            self.agent = VanillaPPO(args.env_name, args.min_replay_size, args.max_replay_size, args.learning_rate, args.seed, args.num_steps,
                           self.iterations )
        
        
        self.run_number = args.run_number
# Plan for the SPO Config
# Things needed that are not given in Experiment Config:
# Num of iterations, Queue size, Preference_Function, 

def main():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--iterations', type=int, default=100, help='Number of Loops of SPO')
    parser.add_argument('--queue_size', type=int, default=10, help='Size of queue to hold trajectories')
    parser.add_argument('--env_name', type=str, default='gym:Ant-v4', help='What environment to run')
    parser.add_argument('--seed', type=int, default=int(random.random() * 10000), help='Random seed.')
    parser.add_argument('--num_steps', type=int, default=100_000, help='Number of env steps to run.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate of agent')
    parser.add_argument('--min_replay_size', type=int, default=10000, help='Minimum replay buffer size')
    parser.add_argument('--max_replay_size', type=int, default=1000000, help='Maximum replay buffer size')
    parser.add_argument('--preference_function', type=str, default='intransitive', help='Preference Function to use')
    parser.add_argument('--run_number', type=str, default="0", help='Run Number')
    parser.add_argument('--agent', type=str, default='sac', help='Agent being used')
    args = parser.parse_args()
    config = SPOConfig(args)

    runner = SPORunner(config)
    runner.run(runner.experiment)
    print(args.seed)


if __name__ =='__main__':
    main()

