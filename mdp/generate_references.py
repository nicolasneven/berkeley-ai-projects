# generate_references.py
#=============================================================
# Generate references for project 3 - MDP and Reinforcement Learning

# Created by Shihan Lu, Feb 21, 2021
# If you find any mistakes about this template, please contact
# Shihan Lu (shihanlu@usc.edu) or post on Piazza.
#=============================================================


import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from code_template import value_iteration, policy_evaluation, policy_iteration, q_learning_1, q_learning_2, q_learning_3


if __name__ == '__main__':
    np.random.seed(42)
    num_cases = 3
    map = {}
    map[0] = [
            'SFFF',
            'FHFH',
            'FFFH',
            'HFFG'

            # 'SFFF',
            # 'FHFH',
            # 'FHFH',
            # 'HFFG'
        ]

    map[1] = [
            # 'SFFFF',
            # 'FHFHF',
            # 'FHFHF',
            # 'FFFHH',
            # 'HHFFG'
            # 'SHHF',
            # 'FFFH',
            # 'HHFH',
            # 'HFFG'

            'SHFF',
            'FFFH',
            'FFFH',
            'HFFG'
        ]

    map[2] = [
            # 'SFFFHF',
            # 'FFHFFH',
            # 'FHFHFH',
            # 'FHHFFF',
            # 'FFFFHF',
            # 'FFHFFG'
            # 'SHHFF',
            # 'FFHHH',
            # 'FHHFF',
            # 'FFFFF',
            # 'HFFFG'
            'SHFFF',
            'FFHFF',
            'FHHFF',
            'FFFFF',
            'FFFFG'
        ]

    gamma = np.random.uniform(0.9, 1.0, num_cases)
    alpha = np.random.uniform(0.4,0.6, num_cases)
    epsilon = np.random.uniform(0.5, 1.0, num_cases)

    np.savez('maps_params.npz',
             num_cases=num_cases,
             maps=map,
             gamma=gamma,
             alpha=alpha,
             epsilon=epsilon
             )

    fileNames = ['autograder_reference'+str(i)+'.npz' for i in range(num_cases)]

    for i in range(num_cases):
        # Q1: value iteration
        env = gym.make('FrozenLake-v0', desc=map[i])
        env.seed(42)
        policy1 = value_iteration(env, gamma=gamma[i], theta=0.0001)

        # Q2: policy evaluation
        env = gym.make('FrozenLake-v0', desc=map[i])
        env.seed(42)
        nS = env.nS
        nA = env.nA
        policy2 = np.ones([nS, nA]) / nA
        value2 = policy_evaluation(policy2, env, gamma=gamma[i], theta=0.0001)

        # Q3: policy iteration
        env = gym.make('FrozenLake-v0', desc=map[i])
        env.seed(42)
        policy3 = policy_iteration(env, gamma=gamma[i], theta=0.0001)

        # Q4: q-learning
        # version 1
        env = gym.make('FrozenLake-v0', desc=map[i])
        env.seed(42)
        Q4_1, policy4_1 = q_learning_1(env, alpha=alpha[i], gamma=gamma[i], epsilon=epsilon[i], num_episodes=500)

        # version 2
        env = gym.make('FrozenLake-v0', desc=map[i])
        env.seed(42)
        Q4_2, policy4_2 = q_learning_2(env, alpha=alpha[i], gamma=gamma[i], epsilon_0=epsilon[i], epsilon_min=0.01,
                                       Lambda=0.001, num_episodes=500)
        # version 3
        env = gym.make('FrozenLake-v0', desc=map[i])
        env.seed(42)
        Q4_3, policy4_3 = q_learning_3(env, alpha=alpha[i], gamma=gamma[i], epsilon=epsilon[i], num_episodes=500)


        np.savez(fileNames[i], num_cases=num_cases,
                 maps=map,
                 gamma=gamma,
                 alpha=alpha,
                 epsilon=epsilon,
                 policy1=policy1,
                 value2=value2,
                 policy3=policy3,
                 Q4_1=Q4_1,
                 policy4_1=policy4_1,
                 Q4_2=Q4_2,
                 policy4_2=policy4_2,
                 Q4_3=Q4_3,
                 policy4_3=policy4_3,
                 )