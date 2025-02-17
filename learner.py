import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from matplotlib import pyplot as plt
import gymnasium as gym

from environment import IndexSelectionEnv
from ReplayMemory import ReplayMemory, Transition
from DQN import DQN
from preprocessor import Preprocessor

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

'''
HYPERPARAMETERS
move into config
'''
BATCH_SIZE = 32
DISCOUNT_RATE = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
UPDATE_RATE = 0.005
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 100000
NN_HIDDEN_LAYERS = [64, 64, 64]

ALPHA = 0.5
BETA = 0.5
SPACE_BUDGET = 2000000

'''
ENVIRONMENT
'''
p = Preprocessor()
p.build_workload_matrix(SPACE_BUDGET)
replicas = ['1']
gym.register(
    id='gymnasium_env/IndexSelectionEnv',
    entry_point=IndexSelectionEnv
)
env = gym.make('gymnasium_env/IndexSelectionEnv', 1000, None, replicas=replicas, candidates=p.candidates, cols_to_table=p.cols_to_table, candidate_sizes=p.candidate_sizes, templates=p.templates, queries=p.templates, space_budget=SPACE_BUDGET, alpha=ALPHA, beta=BETA, mode = 'cost')

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = np.size(state)

policy_net = DQN(n_observations, n_actions, NN_HIDDEN_LAYERS).to(device)
target_net = DQN(n_observations, n_actions, NN_HIDDEN_LAYERS).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = ReplayMemory(REPLAY_BUFFER_SIZE)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * DISCOUNT_RATE) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def learn():
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        print('*** this is episode', i_episode)
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*UPDATE_RATE + target_net_state_dict[key]*(1-UPDATE_RATE)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

    return state, info

config = learn()
print('LEARNED CONFIGURATION')
for idx, replica in enumerate(config[0].tolist()[0]):
    print('--- replica', idx)
    print('space:', config[1]['spaces_used'][idx], '/', SPACE_BUDGET)
    print('indexes:')
    print(replica)
    for can_idx, include in enumerate(replica):
        if include == 1:
            print('-', p.candidates[can_idx], '(size: %d)' % p.candidate_sizes[p.candidates[can_idx]])
