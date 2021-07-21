import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
from collections import namedtuple
from itertools import count

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import box
import os, shutil, sys, argparse
from buffer import ReplayBuffer
from model import Network

# ---------------------------------------MY EDITIONS-------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch Stocks Prediction Model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device')
parser.add_argument('--save', type=str, default='DQN', help='name of the file')
parser.add_argument('--max_episodes', type=int, default=200, help="number of episodes")
parser.add_argument('--with_buffer', action='store_false', help='using replay buffer')
parser.add_argument('--with_target', action='store_false', help='using the target network')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

save = 'saves/{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
os.mkdir(os.path.join(save))
scripts_to_save = ['DQN.py', 'buffer.py', 'model.py']
os.mkdir(os.path.join(save, 'scripts'))
for script in scripts_to_save:
    dst_file = os.path.join(save, 'scripts', os.path.basename(script))
    shutil.copyfile(script, dst_file)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info('Experiment dir : {}'.format(save))
logging.info('Args: {}'.format(args))
# ------------------------------------------------------------------------------------------------------------

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'not_done'))

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
plt.ion()

# look for a gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Parameters
network_params = {
    'state_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'hidden_dim': 154                   # 64
}

training_params = {
    'batch_size': 407,                  # 256,
    'gamma': 0.95,                      # 0.95,
    'epsilon_start': 1.1,               # 1.1
    'epsilon_end': 0.0433,              # 0.05,
    'epsilon_decay': 0.258,             # 0.95,
    'target_update': 'hard',            # use 'soft' or 'hard'
    'tau': 0.01,                        # 0.01,  # relevant for soft update
    'target_update_period': 7,          # 15,    # relevant for hard update
    'grad_clip': 0.244,                 # 0.1
}
network_params = box.Box(network_params)
params = box.Box(training_params)

# Target Network
if not args.with_target:                # without target network
    params.target_update = 'hard'
    params.target_update_period = 1        # updating every weights update

# Replay Buffer
if args.with_buffer:                    # with Replay Buffer
    buffer = ReplayBuffer(100000)
else:                                   # without Replay Buffer
    buffer = ReplayBuffer(1)
    params.batch_size = 1


logging.info("network_params: {}".format(network_params))
logging.info("training_params: {}".format(params))

# Build neural networks
policy_net = Network(network_params, device).to(device)
# TODO: build the target network and set its weights to policy_net's wights (use state_dict from pytorch)
target_net = Network(network_params, device).to(device)
target_net.load_state_dict(policy_net.state_dict())         # copying weights of policy network

optimizer = optim.Adam(policy_net.parameters())

epsilon = params.epsilon_start


# ============================================================================
# Plotting function
def plot_graphs(all_scores, all_losses, all_errors, axes):
    axes[0].plot(range(len(all_scores)), all_scores, color='blue')
    axes[0].set_title('Score over episodes')
    axes[1].plot(range(len(all_losses)), all_losses, color='blue')
    axes[1].set_title('Loss over episodes')
    axes[2].plot(range(len(all_errors)), all_errors, color='blue')
    axes[2].set_title('Mean Q error over episodes')
    plt.subplots_adjust(hspace=.6)


# Training functions
def select_action(s):
    '''
    This function gets a state and returns an action.
    The function uses an epsilon-greedy policy.
    :param s: the current state of the environment
    :return: a tensor of size [1,1] (use 'return torch.tensor([[action]], device=device, dtype=torch.long)')
    '''
    global epsilon
    # TODO: implement action selection.
    # epsilon update is done outside!
    # the actions are taken from the policy network, in an epsilon-greedy manner
    net_out = policy_net(s)
    if torch.ones(1).bernoulli(min(epsilon, 1)):
        # pick random action
        return torch.randint(len(net_out)+1, (1, 1), device=device, dtype=torch.long)
    else:
        # pick argmax action
        return torch.tensor([[torch.argmax(net_out)]], device=device, dtype=torch.long)


def train_model():
    # Pros tips: 1. There is no need for any loop here!!!!! Use matrices!
    #            2. Use the pseudo-code.

    if len(buffer) < params.batch_size:
        # not enough samples
        return 0, 0

    # sample mini-batch
    transitions = buffer.sample(params.batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    next_states_batch = torch.cat(batch.next_state)
    reward_batch = torch.cat(batch.reward)
    not_done_batch = batch.not_done
    #not_done_batch = torch.cat(batch.not_done)

    # Compute curr_Q = Q(s, a) - the model computes Q(s), then we select the columns of the taken actions.
    # Pros tips: First pass all s_batch through the network
    #            and then choose the relevant action for each state using the method 'gather'
    # TODO: fill curr_Q - part a
    # propagating the states through the network, then taking the value of picked actions
    curr_Q = policy_net(state_batch).gather(1, action_batch)

    # Compute expected_Q (target value) for all states.
    # Don't forget that for terminal states we don't add the value of the next state.
    # Pros tips: Calculate the values for all next states ( Q_(s', max_a(Q_(s')) )
    #            and then mask next state's value with 0, where not_done is False (i.e., done).
    # TODO: fill expected_Q - part a
    if torch.cuda.is_available():
        mask = torch.cuda.FloatTensor(not_done_batch)
    else:
        mask = torch.FloatTensor(not_done_batch)
    expected_Q = reward_batch + \
                 params.gamma * target_net(next_states_batch).max(dim=1).values * mask
    #               gamma     *         (Q_(s', max_a(Q)')                      *       Mask of termination states

    # Compute Huber loss. Smoother than MSE
    loss = F.smooth_l1_loss(curr_Q.squeeze(), expected_Q)

    # Optimize the model
    loss.backward()
    # clip gradients to help convergence
    nn.utils.clip_grad_norm_(policy_net.parameters(), params.grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    estimation_diff = torch.mean(curr_Q - expected_Q).item()

    return loss.item(), estimation_diff


# ============================================================================
def cartpole_play():

    FPS = 25
    visualize = 'True'

    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env,'recording',force=True)
    net = Network(network_params, device).to(device)
    logging.info('load best model ...')
    net.load_state_dict(torch.load(os.path.join(save, 'best.dat')))

    logging.info('make movie ...')
    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False)).float()
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if visualize:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    logging.info("Total reward: %.2f" % total_reward)
    logging.info("Action counts:", c)
    env.close()


# ============================================================================
# Training loop
max_episodes = args.max_episodes        # 200
max_score = 500
task_score = 0
# performances plots
all_scores = []
all_losses = []
all_errors = []
fig, axes = plt.subplots(3, 1)

# train for max_episodes
for i_episode in range(max_episodes):
    epsilon = max(epsilon*params.epsilon_decay, params.epsilon_end)
    ep_loss = []
    ep_error = []
    # Initialize the environment and state
    state = torch.tensor([env.reset()], device=device).float()
    done = False
    score = 0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        score += reward

        next_state = torch.tensor([next_state], device=device).float()
        reward = torch.tensor([reward], device=device).float()
        # Store the transition in memory
        buffer.push(state, action, next_state, reward, not done)

        # Update state
        state = next_state

        # Perform one optimization step (on the policy network)
        loss, Q_estimation_error = train_model()

        # save results
        ep_loss.append(loss)
        ep_error.append(Q_estimation_error)

        # soft target update
        if params.target_update == 'soft':
            # TODO: Implement soft target update - part d
            # theta' <- T * theta + (1-T) * theta'
            sd_policy = policy_net.state_dict()     # theta
            sd_target = target_net.state_dict()     # theta'
            # soft update for all the parameters
            for key in sd_policy:
                sd_target[key] = sd_policy[key] * params.tau + sd_target[key] * (1 - params.tau)

        if done or t >= max_score:
            logging.info("Episode: {} | Current target score {} | Score: {}".format(i_episode+1, task_score, score))
            break

    # plot results
    all_scores.append(score)
    all_losses.append(np.average(ep_loss))
    all_errors.append(np.average(ep_error))
    plot_graphs(all_scores, all_losses, all_errors, axes)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.0001)
    if (i_episode + 1) % 50 == 0:
        fig.savefig(os.path.join(save, f'episode_{i_episode + 1:04}.png'))

    # hard target update. Copying all weights and biases in DQN
    if params.target_update == 'hard':
        # TODO: Implement hard target update - part c
        # Copy the weights from policy_net to target_net after every x episodes
        if i_episode % params.target_update_period == 0:         # every 15 episodes: theta' <- theta
            target_net.load_state_dict(policy_net.state_dict())

    # update task score
    if min(all_scores[-5:]) > task_score:
        task_score = min(all_scores[-5:])
        # TODO: store weights
        torch.save(policy_net.state_dict(), os.path.join(save, 'best.dat'))


logging.info('------------------------------------------------------------------------------')
logging.info('Final task score = ' + str(task_score))

plt.ioff()
plt.show()
fig.savefig(os.path.join(save, f'episodes_end.png'))

cartpole_play()
