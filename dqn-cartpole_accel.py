#!/usr/bin/env python

# Derived from https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb
# DQN without a frozen target network

import datetime
import time

seed_value = 324267

import math
import os 
import random 
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F


os.environ['PYTHONHASHSEED']=str(seed_value) 
random.seed(seed_value) 
np.random.seed(seed_value) 
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import gym
# CartPole-v0 Environment
env_id = "CartPole-v0"
env = gym.make(env_id)
env.seed(seed_value);

USE_GPU = False

# Use CUDA
USE_CUDA = torch.cuda.is_available() and USE_GPU

if USE_CUDA:
    torch.cuda.manual_seed(seed_value)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# REPLAY BUFFER

from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module): #base model
    def __init__(self, num_inputs, num_actions, HIDDEN_LAYER_WIDTH):
        super(DQN, self).__init__()
        
        self.action_dim = num_actions
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, HIDDEN_LAYER_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_WIDTH, num_actions)
        )

    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                state   = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0).to(device)
                q_values = self.forward(state)
                action  = q_values.max(dim=1)[1].item()
            else:
                action = random.randrange(self.action_dim)
        return action

# e-greedy exploration
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


# MODEL
# only one NN for estimating Q-values
model = DQN(env.observation_space.shape[0], 
            env.action_space.n,
            128)    
model = model.to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=0.1)
criterion = nn.MSELoss()

# REPLAY BUFFER
replay_buffer = ReplayBuffer(capacity=1000)

def update_target(current_model, target_model):
    target.load_state_dict(model.state_dict())
    
def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.tensor(np.float32(state)      ,dtype=torch.float32).to(device)
    next_state = torch.tensor(np.float32(next_state) ,dtype=torch.float32, requires_grad=False).to(device)
    action     = torch.tensor(action                ,dtype=torch.long).to(device)
    reward     = torch.tensor(reward                ,dtype=torch.float32).to(device)
    done       = torch.tensor(done                  ,dtype=torch.float32).to(device)

    q_values = model(state)
    q_value  = q_values.gather(dim=1, index=action.unsqueeze(dim=1)).squeeze(dim=1)

    next_q_values = model(next_state)
    next_q_value  = next_q_values.max(dim=1)[0]
    
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = criterion(q_value, expected_q_value.detach())
       
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.to('cpu')



if __name__ == '__main__':
    print('torch version: {}'.format(torch.__version__))                                                       
    from_time = time.time()                                                                                         
    # Training BEGINS
    num_frames = 50_000
    batch_size = 32
    gamma      = 0.9

    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size)
    # Training ENDS
    
    running_time = time.time() - from_time
    print('running time: {}'.format(running_time))
    





