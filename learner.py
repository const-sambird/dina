import argparse
import math
import random
import time
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
from qnn import QuantumDQN
from preprocessor import Preprocessor
from profiling import Profiler
from database import Replica
from router import Router
from tpch_generator import TPCHGenerator
from tpcds_generator import TPCDSGenerator
from workload_manager import WorkloadManager

import wandb
import os

def get_replicas(path = './replicas.csv') -> list[Replica]:
    replicas = []
    with open(path, 'r') as infile:
        lines = infile.readlines()
        for config in lines:
            fields = config.split(',')
            replicas.append(
                Replica(
                    id=fields[0],
                    hostname=fields[1],
                    port=fields[2],
                    dbname=fields[3],
                    user=fields[4],
                    password=fields[5]
                )
            )
    return replicas

def create_nets(n_qubits, quantum, n_observations, n_actions, qnn_output, num_shots, device, param_layers) -> tuple[DQN | QuantumDQN]:
    if quantum:
        policy_net = QuantumDQN(n_observations, n_qubits, n_actions, param_layers=param_layers, qnn_output=qnn_output, n_shots=num_shots, torch_device=device).to(device)
        target_net = QuantumDQN(n_observations, n_qubits, n_actions, param_layers=param_layers, qnn_output=qnn_output, n_shots=num_shots, torch_device=device).to(device)
    else:
        policy_net = DQN(n_observations, n_actions, NN_HIDDEN_LAYERS).to(device)
        target_net = DQN(n_observations, n_actions, NN_HIDDEN_LAYERS).to(device)
    return policy_net, target_net


def select_action(state, mask, timestep, epsilon = None):
    #print('mask:', mask)
    if epsilon is None:
        epsilon = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * timestep / EPS_DECAY)
    if epsilon > eps_threshold:
        print(f'exploitation ({epsilon} > {eps_threshold})')
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            predictions = policy_net(state).cpu()
            masked = predictions * mask
            return masked.max(1).indices.view(1, 1).to(device=device)
    else:
        print(f'exploration ({epsilon} < {eps_threshold})')
        return torch.tensor([[env.action_space.sample(mask=mask)]], device=device, dtype=torch.long)

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

has_gradients = False

def optimize_model():
    global has_gradients
    if len(memory) < BATCH_SIZE:
        return
    has_gradients = True
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

def learn(router: Router):
    num_episodes = max(args.num_epochs)

    for i_episode in range(num_episodes):
        opt_times = []
        print('*** this is episode', i_episode)
        manager.update_workload()
        return_state = None
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state, info['mask'], i_episode)
            observation, reward, terminated, truncated, info = env.step(action.item())
            this_reward = reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
                return_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            tic_opt = time.time()
            # Perform one step of the optimization (on the policy network)
            optimize_model()
            toc_opt = time.time()
            print(f'optimisation this step took {toc_opt - tic_opt} seconds')
            opt_times.append(toc_opt - tic_opt)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*UPDATE_RATE + target_net_state_dict[key]*(1-UPDATE_RATE)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * i_episode / EPS_DECAY)
                
                if has_gradients:
                    gradients = policy_net.qnn.qnn.qnn.weight.grad.cpu() \
                                if IS_QUANTUM \
                                else policy_net.layers[-1].weight.grad.cpu()
                    mean_grad = torch.mean(torch.abs(gradients)) if has_gradients else -1
                    norm_grad = np.linalg.norm(gradients)

                wandb.log({
                    'episodes': t + 1,
                    'mean_opt_time': sum(opt_times)/len(opt_times),
                    'workload_cost': sum(router.replica_costs),
                    'reward': this_reward,
                    'epsilon': eps_threshold,
                    'skew': info['skew'],
                    'max_overage': (max(info['spaces_used']) - SPACE_BUDGET) / SPACE_BUDGET,
                    'gradient_mean': mean_grad if has_gradients else 0,
                    'gradient_norm': norm_grad if has_gradients else 0
                })
                plot_durations()

                if (i_episode + 1) in args.num_epochs and not (i_episode + 1) == max(args.num_epochs):
                    report_learned_config(get_final_state(router, False))

                break

    if return_state is not None:
        state = return_state

    return get_final_state(router, True)

def get_final_state(router: Router, should_log: bool):
    print('***** generating final index configuration!')
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # get best action (no exploration)
        action = select_action(state, info['mask'], 0, 1)
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        if done: break
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    if should_log:
        wandb.log({
            'episodes': t + 1,
            'workload_cost': sum(router.replica_costs),
            'reward': reward,
            'skew': info['skew'],
            'max_overage': (max(info['spaces_used']) - SPACE_BUDGET) / SPACE_BUDGET
        })
    return state, info

def report_learned_config(config):
    final_state = config[0].tolist()[0]
    manager.set_to_full()

    router.evaluate(final_state)
    
    manager.set_to_partial()
    config_output_for_benchmarker = []

    print('LEARNED CONFIGURATION')
    learned_config = []
    for idx, replica in enumerate(final_state):
        this_config = []
        print('--- replica', idx)
        print('space:', config[1]['spaces_used'][idx], '/', SPACE_BUDGET)
        print('indexes:')
        print(replica)
        for can_idx, include in enumerate(replica):
            if include == 1:
                this_config.append(p.candidates[can_idx])
                print('-', p.candidates[can_idx])
                config_output_for_benchmarker.append(f'{idx},' + ','.join(p.candidates[can_idx]))
        learned_config.append(this_config)
    print('ROUTEING TABLE')
    print(router.routes)

    print('=' * 20)
    print('OUTPUT RECOMMENDATION FOR BENCHMARKING MODULE\n')
    print('Index configuration:')
    print(' '.join(config_output_for_benchmarker))
    print('\nRouteing table:')
    print(','.join(map(str, router.routes)))
    if RUN_TYPE == 'low_data':
        print('\nTraining templates:')
        print(','.join(map(str, manager.partial_templates())))
    print('=' * 20)

    return learned_config, router.routes

def preconfigure_wandb():
    with open("wandb.env") as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            os.environ[key] = val

def create_arguments():
    parser = argparse.ArgumentParser()

    # the more relevant ones
    parser.add_argument('-q', '--quantum', action='store_true', help='use quantum neural networks instead of classical ones')
    parser.add_argument('-n', '--num-qubits', type=int, default=8, help='the number of qubits to use in the quantum neural nets')
    parser.add_argument('-b', '--space-budget', type=int, default=1e9, help='the amount of space on each replica that the indexes are allowed to take (in bytes)')
    parser.add_argument('-s', '--scale-factor', type=int, default=1, help='TPC-H scale factor')
    parser.add_argument('-e', '--num-epochs', type=int, nargs='+', default=[100], help='number of learning episodes')
    parser.add_argument('-w', '--max-index-width', type=int, help='maximum number of columns that may form an index')
    parser.add_argument('-m', '--benchmark-mode', type=str, choices=['cost', 'exe'], default='cost', help='benchmark execution mode -- \'cost\' for the cost estimator, \'exe\' for actual execution times')
    parser.add_argument('-o', '--num-shots', type=int, default=1024, help='number of samples to take from the quantum neural network')
    parser.add_argument('-g', '--generate-queries', action='store_true', help='generate new queries from the templates')
    parser.add_argument('-t', '--queries-per-template', type=int, default=10, help='number of queries per template that are in the workload or should be generated')
    parser.add_argument('-W', '--workload', type=str, choices=['tpc-h', 'tpc-ds'], default='tpc-h', help='the workload to run (TPC-H, TPC-DS)')

    # these ones can probably be left to the defaults
    parser.add_argument('--batch-size', type=int, default=32, help='the batch size to feed into the neural network')
    parser.add_argument('--discount-rate', type=float, default=0.99, help='the discount rate for the reinforcement learner')
    parser.add_argument('--eps-start', type=float, default=0.9, help='the starting probability of the reinforcement learner exploration rate')
    parser.add_argument('--eps-end', type=float, default=0.05, help='the ending probability of the reinforcement learner exploration rate')
    parser.add_argument('--eps-decay', type=float, default=250, help='the rate at which the exploration probability decays')
    parser.add_argument('--update-rate', type=float, default=0.005, help='the rate at which the policy nets are updated')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='the rate at which the q-learner learns')
    parser.add_argument('--replay-buffer', type=int, default=100000, help='the size of the replay buffer')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[64, 64, 64], help='the hidden layers in the neural network, number of neurons (classical only. ignored for quantum)')
    parser.add_argument('--workload-factor', type=float, default=0.5, help='the weight that the workload time should take in the reward function')
    parser.add_argument('--skew-factor', type=float, default=0.5, help='the weight that the workload skew should take in the reward function')
    parser.add_argument('--qnn-output', type=str, choices=['trunc', 'layer'], help='how should we map the output probabilities from the QNN to actions? [trunc]ate them to fit or add a classical [layer] (quantum only)')
    parser.add_argument('--seed', type=int, default=None, help='the seed for the PRNG used in exploration')
    parser.add_argument('--dry-run', action='store_true', help='do not enable logging to weights & biases for this run')
    parser.add_argument('--workload-dir', type=str, default='./workload', help='the directory where the workload .sql files and template assignment .csv are kept')
    parser.add_argument('--template-dir', type=str, default='./templates', help='the path to the query templates to generate the workload')
    parser.add_argument('--save-model', action='store_true', help='write the model weights to disk after training is complete')
    parser.add_argument('--load-model', action='store_true', help='load model weights from disk before training starts')
    parser.add_argument('--param-layers', type=int, default=3, help='the number of repetitions of the ansatz setup')
    parser.add_argument('--train-fraction', type=float, default=0.2, help='what proportion of the workload should be in the training set?')

    parser.add_argument('run_type', type=str, choices=['recommend', 'low_data', 'drift'],
                        help='what experiment should we run? recommend indexes (normal), low data (limited templates), or workload drift')

    return parser.parse_args()

if __name__ == '__main__':
    preconfigure_wandb()
    wandb.login()
    args = create_arguments()
    '''
    HYPERPARAMETERS
    '''
    EXE_MODE = args.benchmark_mode
    WORKLOAD = args.workload

    BATCH_SIZE = args.batch_size
    DISCOUNT_RATE = args.discount_rate
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_DECAY = args.eps_decay # remove?
    UPDATE_RATE = args.update_rate
    LEARNING_RATE = args.learning_rate
    REPLAY_BUFFER_SIZE = args.replay_buffer
    NN_HIDDEN_LAYERS = args.hidden_layers
    QNN_OUTPUT = args.qnn_output
    SEED = args.seed

    ALPHA = args.workload_factor
    BETA = args.skew_factor
    SPACE_BUDGET = args.space_budget

    NUM_QUBITS = args.num_qubits
    IS_QUANTUM = args.quantum
    NUM_SHOTS = args.num_shots
    GENERATE_QUERIES = args.generate_queries
    NUM_REPETITIONS = args.param_layers
    TRAIN_FRACTION = args.train_fraction

    RUN_TYPE = args.run_type

    '''
    ENVIRONMENT
    '''
    profiler = Profiler()
    if WORKLOAD == 'tpc-h':
        print('generating TPC-H queries!')
        generator = TPCHGenerator(args.template_dir, args.queries_per_template, args.workload_dir, args.scale_factor)
    elif WORKLOAD == 'tpc-ds':
        print('generating TPC-DS queries!')
        generator = TPCDSGenerator(args.template_dir, args.queries_per_template, args.workload_dir, args.scale_factor)
    replicas = get_replicas()

    if GENERATE_QUERIES:
        generator.create_queries()
    
    queries, templates = generator.get_workload()
    manager = WorkloadManager(queries, templates, RUN_TYPE, TRAIN_FRACTION)

    random.seed(SEED)
    if SEED is not None:
        torch.manual_seed(SEED)
        for replica in replicas:
            with replica.connection().cursor() as cur:
                if SEED == 0:
                    to_set = 0
                else:
                    to_set = 1 / SEED
                cur.execute('SELECT setseed(%f);' % to_set)
            replica.commit()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    if torch.cuda.is_available():
        print('found CUDA!')
    elif torch.backends.mps.is_available():
        print('found MPS!')
    else:
        print('****** torch did not find CUDA/MPS! *******')
    
    budget_gb = SPACE_BUDGET / 1e9
    maxwidth = 'all' if args.max_index_width is None else args.max_index_width

    run = wandb.init(
        project='qdina',
        name=f'{'cl' if not IS_QUANTUM else 'q' + str(NUM_QUBITS)}-n{len(replicas)}-w{maxwidth}-b{budget_gb:.1f}',
        config={
            'EXE_MODE': EXE_MODE,
            'BATCH_SIZE': BATCH_SIZE,
            'SPACE_BUDGET': SPACE_BUDGET,
            'IS_QUANTUM': IS_QUANTUM,
            'NUM_QUBITS': NUM_QUBITS,
            'MAX_INDEX_WIDTH': args.max_index_width,
            'NUM_EPOCHS': args.num_epochs,
            'SCALE_FACTOR': args.scale_factor,
            'DISCOUNT_RATE': DISCOUNT_RATE,
            'EPS_START': EPS_START,
            'EPS_END': EPS_END,
            'EPS_DECAY': EPS_DECAY,
            'UPDATE_RATE': UPDATE_RATE,
            'LEARNING_RATE': LEARNING_RATE,
            'REPLAY_BUFFER_SIZE': REPLAY_BUFFER_SIZE,
            'NN_HIDDEN_LAYERS': NN_HIDDEN_LAYERS,
            'ALPHA': ALPHA,
            'BETA': BETA,
            'QNN_OUTPUT': QNN_OUTPUT,
            'SEED': SEED,
            'NUM_REPLICAS': len(replicas),
            'NUM_SHOTS': NUM_SHOTS,
            'NUM_REPETITIONS': NUM_REPETITIONS,
            'TRAIN_FRACTION': TRAIN_FRACTION,
            'WORKLOAD': WORKLOAD,
            'RUN_TYPE': RUN_TYPE
        },
        mode='disabled' if args.dry_run else 'online'
    )

    wandb.define_metric('episodes', summary='mean')
    wandb.define_metric('mean_opt_time', summary='mean')

    # instantiate database connections
    for replica in replicas:
        replica.connection()

    tic = time.time()
    p = Preprocessor(profiler, replicas[0], args.max_index_width, queries, templates)
    p.preprocess(SPACE_BUDGET)

    # reset from any previous runs
    for replica in replicas:
        replica.drop_all_indexes(p.tables, EXE_MODE)

    router = Router(p.workload, templates, p.tables, replicas, p.candidates, p.cols_to_table, profiler, EXE_MODE, manager)

    gym.register(
        id='gymnasium_env/IndexSelectionEnv',
        entry_point=IndexSelectionEnv
    )
    env = gym.make('gymnasium_env/IndexSelectionEnv', 1000, None, profiler=profiler, replicas=replicas,
                   router=router, candidates=p.candidates, tables=p.tables, cols_to_table=p.cols_to_table,
                   templates=p.templates, queries=p.templates, space_budget=SPACE_BUDGET, alpha=ALPHA, beta=BETA,
                   mode=EXE_MODE)

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    env.action_space.seed(SEED)
    state, info = env.reset(seed=SEED)
    n_observations = np.size(state)

    if IS_QUANTUM:
        print(f'{n_actions} actions, {NUM_QUBITS} qubits (encodes {2**NUM_QUBITS})')
    else:
        print(f'{n_actions} actions')

    if args.load_model:
        policy_net = torch.load('./policy.pt')
        target_net = torch.load('./target.pt')
    else:
        policy_net, target_net = create_nets(NUM_QUBITS, IS_QUANTUM, n_observations, n_actions, QNN_OUTPUT, NUM_SHOTS, device, NUM_REPETITIONS)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memory = ReplayMemory(REPLAY_BUFFER_SIZE)

    config = learn(router)
    toc = time.time()

    if args.save_model:
        torch.save(policy_net, './policy.pt')
        torch.save(target_net, './target.pt')

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

    print('Generating routeing table...')
    # router expects the format [ { table: [cols,] } ]
    
    learned_config, routes = report_learned_config(config)

    # close database replica connections
    for replica in replicas:
        replica.close()
    
    print('PROFILING RESULTS')
    print(profiler.times())
    print('TOTAL EXECUTION TIME: %.2fs' % (toc - tic))

    wandb.summary['learned_config'] = learned_config
    wandb.summary['spaces_used'] = config[1]['spaces_used']
    wandb.summary['routing_table'] = router.routes
    wandb.summary['profiling_results'] = profiler.times()
    wandb.summary['recommendation_time'] = toc - tic
    
    wandb.finish()
