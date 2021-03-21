# code_template.py
#=============================================================
# Code template for project 3 - MDP and Reinforcement Learning

# Created by Shihan Lu, Feb 21, 2021
# If you find any mistakes about this template, please contact
# Shihan Lu (shihanlu@usc.edu) or post on Piazza.
#=============================================================


import argparse
import os
import gym
import copy
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map


def value_iteration(env, gamma=0.95, theta=0.0001): # Do not change variables
    ''' Performs value iteration for the given environment

    :param env: Unwrapped Open AI environment
    :param gamma: Decay rate
    :param theta: Acceptable convergence rate

    :return: policy, a |State|x|Action| matrix of the probability
            of taking an action in a given state. Store it in the
            form of a 2d list, e.g.
            [[1.   0.   0.   0.  ]
             [0.   0.   0.   1.  ]
             [1.   0.   0.   0.  ]
             [0.   0.   0.   1.  ]
             [1.   0.   0.   0.  ]
             [0.25 0.25 0.25 0.25]
             [0.5  0.   0.5  0.  ]
             [0.25 0.25 0.25 0.25]
             [0.   0.   0.   1.  ]
             [0.   1.   0.   0.  ]
             [1.   0.   0.   0.  ]
             [0.25 0.25 0.25 0.25]
             [0.25 0.25 0.25 0.25]
             [0.   0.   1.   0.  ]
             [0.   1.   0.   0.  ]
             [0.25 0.25 0.25 0.25]]

            The first element in the list [1. 0. 0. 0.] means that at
            the spot 1, the probability of moving left is 100%, the probabilities
            of moving right, up and down are all 0%.
    '''

    nS = env.nS # number of states
    nA = env.nA # number of actions
    envP = env.P # environment dynamics model, e.g. {0: [], 1: , 2: , 3:}

    V = np.zeros(nS) # initialize array of state values with all zeros
    policy = np.ones([nS, nA]) / nA  # dummy policy which has 0.25 probability for each action at any state.
    # Replace this policy with your implementation

    #==========================================
    "*** YOUR CODE HERE FOR VALUE ITERATION***"
    while True:
        delta = 0
        for s in range(nS):
            v = V[s] # archive old state value
            q = np.zeros(nA)
            for a in range(nA): # Loop through each action
                # For each action at this state, we have transition probabilities, next state, rewards and whether the game ended
                for prob, next_state, reward, done in envP[s][a]:
                    q[a] += prob * (reward + gamma * V[next_state]) # Get action-values from state-values
            V[s] = max(q)
            delta = max(delta, abs(v-V[s]))
        if delta < theta:
            break

    # extract the determinstic policy with optimal state values
    for s in range(nS):
        # Find best action based on optimal state-value
        q = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in envP[s][a]:
                q[a] += prob * (reward + gamma * V[next_state])

        best_a = np.argwhere(q == np.max(q)).flatten() # gives the position of largest q value
        policy[s] = np.sum([np.eye(nA)[i] for i in best_a], axis=0) / len(best_a)
    #==========================================
    return policy


def policy_evaluation(policy, env, gamma=0.95, theta=0.0001): # Do not change this line
    ''' Evaluate the state values following the given policy

    :param policy: 2D matrix |State|x|Action|, where each entry
        the probability of which action to take in each state
    :param env: Unwrapped OpenAI gym environment
    :param gamma: Discount factor
    :param theta: Stopping condition
    :return List of state value. The size of list equals the number of states, e.g.
            [0.00752935 0.00673683 0.01418894 0.00639463 0.01019074 0.
             0.03249842 0.         0.02525507 0.07092393 0.12265383 0.
             0.         0.15073275 0.41302033 0.        ]
    '''

    nS = env.nS # number of states
    nA = env.nA # number of actions
    envP = env.P # environment dynamics model
    V = np.zeros(nS) # initialize array V[s]=0 for all s

    #==========================================
    "*** YOUR CODE HERE FOR POLICY EVALUATION***"
    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            Vs = 0 # temp variable for updating V[s]
            for a, action_prob in enumerate(policy[s]):
                # Loop through to get transition probabilities, next state, rewards and whether the game ended
                temp_vs = 0
                for prob, next_state, reward, done in envP[s][a]:
                    # State-value function to get our values of states given policy
                    temp_vs += prob * (reward + gamma * V[next_state])

                Vs = Vs + action_prob * temp_vs
            V[s] = Vs
            delta = max(delta, np.abs(v - V[s]))

        if delta < theta:
            break
    #==========================================
    return V # Do not change this line


def policy_iteration(env, gamma=0.95, theta=0.0001):
    ''' Perform policy iteration with initially equally likely actions

    Initialize each action to be equally likely for all states
    Policy improvement is run until a stable policy is found

    :param env: Unwrapped OpenAI gym environment
    :param gamma: discount factor for both policy evaluation and policy improvement
    :para theta: Stopping condition for both policy evaluation and policy improvement
    :return policy, a |State|x|Action| matrix of the probability
            of taking an action per state.

            For the given parameters: gamma=0.95, theta=0.0001, you should get the following outputs:
            [[1.   0.   0.   0.  ]
             [0.   0.   0.   1.  ]
             [1.   0.   0.   0.  ]
             [0.   0.   0.   1.  ]
             [1.   0.   0.   0.  ]
             [0.25 0.25 0.25 0.25]
             [0.5  0.   0.5  0.  ]
             [0.25 0.25 0.25 0.25]
             [0.   0.   0.   1.  ]
             [0.   1.   0.   0.  ]
             [1.   0.   0.   0.  ]
             [0.25 0.25 0.25 0.25]
             [0.25 0.25 0.25 0.25]
             [0.   0.   1.   0.  ]
             [0.   1.   0.   0.  ]
             [0.25 0.25 0.25 0.25]]
    '''

    nS = env.nS # number of states
    nA = env.nA # number of actions
    envP = env.P # environment dynamics model
    V = np.zeros(nS) # initialize array V[s]=0 for all s
    policy = np.ones([nS, nA]) / nA # initialize policy that each action to be equally likely for all states

    #==========================================
    "*** YOUR CODE HERE FOR POLICY ITERATION***"
    while True:
        # 1. Policy evaluation
        V = policy_evaluation(policy, env, gamma, theta)

        # 2. Policy improvement
        policy_stable = True
        for s in range(nS):
            old_a = np.argwhere(policy[s] == np.max(policy[s])).flatten()  # action from the old policy
            # Loop through each action to find the best action from the updated policy
            q = np.zeros(nA)
            for a in range(nA):
                # For each action at this state, we have transition probabilities, next state, rewards and whether the game ended
                for prob, next_state, reward, done in envP[s][a]:
                    # Get our action-values from state-values
                    q[a] += prob * (reward + gamma * V[next_state])

            best_a = np.argwhere(q == np.max(q)).flatten()
            policy[s] = np.sum([np.eye(nA)[i] for i in best_a], axis=0) / len(best_a)

            if not np.array_equal(old_a, best_a):
                policy_stable = False

        if policy_stable:
            break
    #==========================================
    return policy  # do not change


def q_learning_1(env, alpha=0.5, gamma=0.95, epsilon=0.5, num_episodes=500):
    ''' Performs Q-learning version 1 for the given environment

    Note:
    - Initialize Q-table to all zeros
    - Utilize a vanilla epsilon-greedy method for action selection

    :param env: Unwrapped Open AI environment
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon: Epsilon in epsilon-greedy method
    :param num_episodes: Number of episodes to use for learning

    :return: Q table and policy

    For the given parameters: alpha=0.5, gamma=0.95, epsilon=0.1, num_episodes=500, you should get the following outputs
    Q table:
         [[0.        0.        0.        0.      ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.11875   0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.2375    0.       ]
         [0.        0.        0.5       0.1128125]
         [0.        0.        0.        0.       ]]

    policy
         [[0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.   0.   1.   0.  ]
         [0.   0.   1.   0.  ]
         [0.25 0.25 0.25 0.25]]
    '''

    # only use np.random.rand() for random sampling, and seed the random sampling as the following line
    np.random.seed(42) # do not change
    nS = env.nS # number of states
    nA = env.nA # number of actions
    Q = np.zeros([nS, nA]) # initialize Q-table with zeros
    policy = np.ones([nS, nA]) / nA # dummy policy that each action is equally likely

    #==========================================
    "*** YOUR CODE HERE FOR Q-LEARNING VERSION 1***"
    ##### ===== only exploit the state has been visited before; otherwise, do exploration
    for t in range(num_episodes):
        s = env.reset() # initialize S
        while True:
            # choose A from S using policy derived from Q-table by epsilon-greedy
            r = np.random.rand()
            if r < epsilon:
                a = np.random.choice(nA)
            else:
                a = np.argmax(Q[s])

            s_prime, reward, done, _ = env.step(a)
            Q[s][a] = Q[s][a] + alpha*(reward + gamma*np.max(Q[s_prime])-Q[s][a])
            s = s_prime
            if done:
                break

    # extract policy based on q-table
    for s in range(nS):
        q = Q[s]
        best_a = np.argwhere(q==np.max(q)).flatten() # gives the position of largest q value
        policy[s] = np.sum([np.eye(nA)[i] for i in best_a], axis=0) / len(best_a)
    #==========================================
    return Q, policy  # do not change



def q_learning_2(env, alpha=0.5, gamma=0.95, epsilon_0=1.0, epsilon_min = 0.01, Lambda = 0.001, num_episodes=500):
    ''' Performs Q-learning version 2 for the given environment

    Note:
    - Initialize Q-table to all zeros
    - Utilize a decay epsilon-greedy method for action selection

    :param env: Unwrapped Open AI environment
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon_0: Initial epsilon
    :param epsilon_min: minimal epsilon
    :param Lambda: a constant in exponential calculation
    :param num_episodes: Number of episodes to use for learning

    :return: Q table and policy

    For the given parameters: alpha=0.5, gamma=0.95, epsilon_0=1.0, epsilon_min = 0.01, Lambda = 0.001, num_episodes=500,
    you should get the following outputs.
    Q-table:
         [[0.36713299 0.37985477 0.35843552 0.35581387]
         [0.17517206 0.15872685 0.19903197 0.36888308]
         [0.40467574 0.40275562 0.49238606 0.40385159]
         [0.22828427 0.07587396 0.2194184  0.42327732]
         [0.45935126 0.2309994  0.27355389 0.21700694]
         [0.         0.         0.         0.        ]
         [0.26192058 0.28668218 0.19145272 0.14202863]
         [0.         0.         0.         0.        ]
         [0.25148164 0.33003944 0.31387329 0.55814431]
         [0.23155422 0.65397369 0.40905852 0.30759947]
         [0.60206605 0.21343306 0.50034703 0.38379698]
         [0.         0.         0.         0.        ]
         [0.         0.         0.         0.        ]
         [0.16369181 0.71453194 0.69310478 0.37951278]
         [0.68626787 0.74187664 0.95385906 0.68487487]
         [0.         0.         0.         0.        ]]

    Policy:
         [[0.   1.   0.   0. ]
         [0.   0.   0.   1.  ]
         [0.   0.   1.   0.  ]
         [0.   0.   0.   1.  ]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.   1.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.   0.   0.   1.  ]
         [0.   1.   0.   0.  ]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.   1.   0.   0.  ]
         [0.   0.   1.   0.  ]
         [0.25 0.25 0.25 0.25]]
    '''

    # only use np.random.rand() for random sampling, and seed the random sampling as the following line
    np.random.seed(42) # do not change
    nS = env.nS # number of states
    nA = env.nA # number of actions
    Q = np.zeros([nS, nA]) # initialize Q-table with zeros
    policy = np.ones([nS, nA]) / nA # dummy policy that each action is equally likely

    #==========================================
    "*** YOUR CODE HERE FOR Q-LEARNING VERSION 2***"
    #### ======= Decay threshold version epsilon-greedy

    for t in range(num_episodes):
        # we initialize the first state of the episode
        current_state = env.reset()

        # update epsilon
        exploration_proba = max(epsilon_min, epsilon_0 * np.exp(-Lambda * t))

        while True:
            # we sample a float from a uniform distribution over 0 and 1
            # if the sampled flaot is less than the exploration proba
            #     the agent selects arandom action
            # else
            #     he exploits his knowledge using the bellman equation
            if np.random.rand() < exploration_proba:
                action = np.random.choice(nA)
            else:
                action = np.argmax(Q[current_state, :])

            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            next_state, reward, done, _ = env.step(action)

            # We update our Q-table using the Q-learning iteration
            Q[current_state, action] = Q[current_state, action] + alpha * (
                    reward + gamma * np.max(Q[next_state, :]) - Q[current_state, action])

            current_state = next_state

            if done: # If the episode is finished, break
                break

    # extract policy based on q-table
    for s in range(nS):
        q = Q[s]
        best_a = np.argwhere(q==np.max(q)).flatten() # gives the position of largest q value
        policy[s] = np.sum([np.eye(nA)[i] for i in best_a], axis=0) / len(best_a)
    ## ==========================================================================
    return Q, policy # do not change



def q_learning_3(env, alpha=0.5, gamma=0.95, epsilon=0.5, num_episodes=500):
    ''' Performs Q-learning version 3 for the given environment

    Note:
    - Initialize Q-table to all zeros
    - Utilize an epsilon-greedy method for action selection

    :param env: Unwrapped Open AI environment
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon: Epsilon in epsilon-greedy method
    :param num_episodes: Number of episodes to use for learning

    :return: Q table and policy,

    For the given parameters: alpha=0.5, gamma=0.95, epsilon=0.5, num_episodes=500, you should get the following outputs.
    Q table:
         [[0.21151246 0.21022286 0.25999264 0.20899564]
         [0.09993746 0.14133315 0.0705775  0.18931574]
         [0.14854259 0.1505181  0.14683159 0.14670341]
         [0.13463091 0.13786784 0.05830575 0.14820557]
         [0.31862173 0.27826314 0.23927896 0.09306707]
         [0.         0.         0.         0.        ]
         [0.09036269 0.0509784  0.01831764 0.02214529]
         [0.         0.         0.         0.        ]
         [0.10107658 0.09926206 0.31749132 0.39445481]
         [0.35796593 0.44074778 0.11963118 0.296104  ]
         [0.28219434 0.06914525 0.30646221 0.19680597]
         [0.         0.         0.         0.        ]
         [0.         0.         0.         0.        ]
         [0.09177565 0.2651842  0.51977341 0.62812079]
         [0.62003614 0.70870562 0.858487   0.49727083]
         [0.         0.         0.         0.        ]]

    policy:
         [[0.   0.   1.   0. ]
         [0.   0.   0.   1.  ]
         [0.   1.   0.   0.  ]
         [0.   0.   0.   1.  ]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.   0.   0.   1.  ]
         [0.   1.   0.   0.  ]
         [0.   0.   1.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.   0.   0.   1.  ]
         [0.   0.   1.   0.  ]
         [0.25 0.25 0.25 0.25]]
    '''

    # only use np.random.rand() for random sampling, and seed the random sampling as the following line
    np.random.seed(42) # do not change
    nS = env.nS # number of states
    nA = env.nA # number of actions
    Q = np.zeros([nS, nA]) # initialize Q-table with zeros
    policy = np.ones([nS, nA]) / nA # dummy policy that each action is equally likely

    #==========================================
    "*** YOUR CODE HERE FOR Q-LEARNING VERSION 3***"
    ##### ===== only exploit the state has been visited before; otherwise, do exploration
    for t in range(num_episodes):
        s = env.reset() # initialize S
        while True:
            # choose A from S using policy derived from Q-table by epsilon-greedy
            r = np.random.rand()
            if r >= epsilon and not all(q == 0 for q in Q[s]):
                a = np.argmax(Q[s])
            else:
                a = np.random.choice(nA)

            s_prime, reward, done, _ = env.step(a)
            Q[s][a] = Q[s][a] + alpha*(reward + gamma*np.max(Q[s_prime])-Q[s][a])
            s = s_prime

            if done:
                break

    # extract policy based on q-table
    for s in range(nS):
        q = Q[s]
        best_a = np.argwhere(q == np.max(q)).flatten() # gives the position of largest q value
        policy[s] = np.sum([np.eye(nA)[i] for i in best_a], axis=0) / len(best_a)
    #==========================================
    return Q, policy


def get_discrete_state(state):
    '''
    convert the continuous state to discrete state in CartPole
    :param state: state
    :return: discrete state
    '''
    np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])  # discretization step
    discrete_state = state / np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(np.int))


def q_learning_cart(env, alpha=0.2, gamma=0.95, epsilon_0=1.0, epsilon_min = 0.01, Lambda = 0.001, num_episodes=60000):
    # you can change the arguments based on your implementation, except env

    ''' Performs Q-learning for the CartPole environment

    Note:
    - Initialize Q-table to all zeros
    - Utilize a decay epsilon-greedy method for action selection

    :param env: Unwrapped Open AI environment
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon_0: Initial epsilon
    :param epsilon_min: minimal epsilon
    :param Lambda: constant in exponential calculation
    :param num_episodes: Number of episodes to use for learning

    :return: Q table
    '''

    # only use np.random.rand() for random sampling, and seed the random sampling as the following line
    np.random.seed(42) # do not change
    nA = env.action_space.n  # number of actions
    state_space = [30, 30, 50, 50]  # create the discrete state space
    Q = np.random.uniform(low=0, high=1, size=(state_space + [env.action_space.n]))  # create Q-table
    # print(q_table.shape)

    #==========================================
    "*** YOUR CODE HERE FOR Q-LEARNING VERSION 2 FOR CARTPOLE***"
    #### ======= Decay threshold version epsilon-greedy
    exploration_proba_init = epsilon_0
    exploration_proba = exploration_proba_init

    for t in range(num_episodes):
        # we initialize the first state of the episode
        current_state = get_discrete_state(env.reset())

        # sum the rewards that the agent gets from the environment
        total_episode_reward = 0
        if t % 4000 == 0:
            print('episode:', t)
        while True:
            # we sample a float from a uniform distribution over 0 and 1
            # if the sampled flaot is less than the exploration proba
            #     the agent selects arandom action
            # else
            #     he exploits his knowledge using the bellman equation
            if np.random.rand() < exploration_proba:
                action = np.random.choice(nA)
            else:
                action = np.argmax(Q[current_state])

            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            next_state, reward, done, _ = env.step(action)

            # We update our Q-table using the Q-learning iteration
            next_state = get_discrete_state(next_state)
            Q[current_state + (action,)] = Q[current_state + (action,)] + alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[current_state + (action,)])

            total_episode_reward = total_episode_reward + reward
            current_state = next_state

            if done: # If the episode is finished, break
                break

            if t % 4000 == 0: #render
                env.render()

        exploration_proba = max(epsilon_min, exploration_proba_init * np.exp(-Lambda * t))

    env.close()
    ## ==========================================================================
    return Q # do not change


def compare_policies(policy1, policy2): # do not change this function
    policy1, policy2 = np.array(policy1), np.array(policy2)
    if np.shape(policy1) != np.shape(policy2):
        print("Two policies are not in the same size. Cannot compare")
        return
    l = len(policy1)
    return np.sum([1 if not np.array_equal(policy1[i], policy2[i]) else 0 for i in range(l)]) # the number of states
    # which have different policies


if __name__ == "__main__":
    # Set parser
    parser = argparse.ArgumentParser(description='Select a particular question')
    parser.add_argument("-q", "--question",
                        type=str, choices=['1', '2', '3', '4', '4-1', '4-2', '4-3'],
                        help="selection a question to test")
    # add a visualization argument
    parser.add_argument("-vis", "--visualization",
                        help="visualize the lake and execution",
                        action="store_true")
    args = parser.parse_args()

    os.system('') # trick way to colorize the console for environment rendering
    np.set_printoptions(suppress=True)
    np.random.seed(42) # seed numpy - do not change this line
    # random_map = generate_random_map(size=5, p=0.8) # generate a random map
    custom_map = [ # customize a map
        'SFFF',
        'FHFH',
        'FFFH',
        'HFFG'
    ]
    env = gym.make('FrozenLake-v0', desc=custom_map)
    #===========
    # env = gym.make('FrozenLake-v0', desc=random_map)
    # env.render()
    #===========
    # env_notSlippery = gym.make('FrozenLake-v0', desc=custom_map, is_slippery=False) # not slippery, so there is no transition probability
    env.seed(42)  # seed env - do not change this line
    print('Environment loaded!\n')

    #====== Useful functions to help your coding and visualize the environment
    if args.visualization:
        print('Environment and execution visualization:')
        env.reset() # reset the environment
        env.render() # render the environment

        random_action = np.random.choice(env.nA) # take a random action
        new_state, reward, done, info = env.step(random_action) # execute the action
        # print(new_state, reward, done, info)
        env.render() # render the environment again

        print('\nReset state:')
        env.reset()
        env.render()
        print("\n")
    # ========================

    # Driver codes to test the functions
    "*** Task 1: use value iteration to find the policy ***"
    if args.question == '1':
        print("Test task 1 - value iteration\n")
        policy1 = value_iteration(env)
        print(policy1)

    "*** Task 2: use policy evaluation to find the state values ***"
    if args.question == '2':
        print("Test task 2 - policy evaluation\n")
        policy2 = np.ones([env.nS, env.nA]) / env.nA  # policy that each action is equally likely
        value2 = policy_evaluation(policy2, env)
        print(value2)

    "*** Task 3: use policy iteration to find the policy ***"
    if args.question == '3':
        print("Test task 3 - policy iteration\n")
        policy3 = policy_iteration(env)
        print(policy3)

    "*** Task 4: use Q-Learning to learn the Q table and policy***"
    if args.question == '4-1':
        print("Test task 4 - q-learning version 1, lake with no shifted ice")
        # env.render()
        q_table, policy = q_learning_1(env)
        print('\n q_table from q-learning:\n', q_table)
        print('\n policy from q-learning:\n', policy)

        # compare with the policy from value iteration
        env.reset()
        policy_valueIter = value_iteration(env)
        print('\n policy from value iteration:\n', policy_valueIter)
        print('\n Number of states having different policies compared to value iteration:', compare_policies(policy, policy_valueIter))

    if args.question == '4-2':
        print("Test task 4 - q-learning version 2, lake with no shifted ice")
        # env.render()
        q_table, policy = q_learning_2(env)
        print('\n q_table from q-learning:\n', q_table)
        print('\n policy from q-learning:\n', policy)

        # compare with the policy from value iteration
        env.reset()
        policy_valueIter = value_iteration(env)
        print('\n policy from value iteration:\n', policy_valueIter)
        print('\n Number of states having different policies compared to value iteration:', compare_policies(policy, policy_valueIter))

    if args.question == '4-3':
        print("Test task 4 - q-learning version 3, lake with no shifted ice")
        # env.render()
        q_table, policy = q_learning_3(env)
        print('\n q_table from q-learning:\n', q_table)
        print('\n policy from q-learning:\n', policy)

        # compare with the policy from value iteration
        env.reset()
        policy_valueIter = value_iteration(env)
        print('\n policy from value iteration:\n', policy_valueIter)
        print('\n Number of states having different policies compared to value iteration:', compare_policies(policy, policy_valueIter))

