
'''
USAGE:  python dqn-cartpole.py 123 base_config
        python dqn-cartpole.py <seed> <config>
'''

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb
# DQN without a frozen target network


# In[2]:


import yaml
import datetime
import sys

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

from torch.utils.tensorboard import SummaryWriter
# %reload_ext tensorboard
# %tensorboard --port=9806 --logdir ./runs


# In[3]:


# experiment_no = 'exp_1'
# experiment_no = 'base_config'
experiment_no  =  sys.argv[2]


seed_value = sys.argv[1]
seed_value = 789987 # sys.argv[1]

# Writer will output to ./runs/ directory by default
writer_dir = './runs/' + experiment_no + '_' + str(seed_value) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(writer_dir)
print("EXPERIMENT: ", experiment_no, "\tSEED: ", seed_value)


# In[4]:


# FROM CONFIG FILE
config_path =  './' + experiment_no + '.yaml' # sys.argv[2]
config = yaml.safe_load(open(config_path,'r'))
  
import math
import os 
import random 
import numpy as np 
import tensorflow as tf 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F


# In[5]:


os.environ['PYTHONHASHSEED']=str(seed_value) 
random.seed(seed_value) 
np.random.seed(seed_value) 
tf.random.set_seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[6]:


import gym
# CartPole-v0 Environment
env_id = "CartPole-v0"
env = gym.make(env_id)
env.seed(seed_value);


# In[7]:


USE_GPU = config['USE_GPU']

# Use CUDA
USE_CUDA = torch.cuda.is_available() and USE_GPU

if USE_CUDA:
    torch.cuda.manual_seed(seed_value)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# In[8]:


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


# In[9]:


# Deep Q-Networks

class DQN(nn.Module): #base model
    def __init__(self, num_inputs, num_actions, HIDDEN_LAYER_WIDTH):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], HIDDEN_LAYER_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_WIDTH, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                state   = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_value = self.forward(state)
                action  = q_value.max(1)[1].data[0].item()
            else:
                action = random.randrange(env.action_space.n)
        return action


# In[10]:


# e-greedy exploration

epsilon_start = config['EPSILON_START']
epsilon_final = config['EPSILON_FINAL']
epsilon_decay = config['EPSILON_DECAY']

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


# In[11]:


# plt.plot([epsilon_by_frame(i) for i in range(10000)])


# In[12]:


# MODEL
if (config['MODEL_NAME']=='DQN'):
    model = DQN(env.observation_space.shape[0], 
                env.action_space.n,
                config['HIDDEN_LAYER_WIDTH'])
else: #default model is DQN class
    model = DQN(env.observation_space.shape[0], 
                env.action_space.n,
                config['HIDDEN_LAYER_WIDTH'])    

model = model.to(device)


# In[13]:


# OPTIMIZER
if (config['OPTIMIZER']=='Adam'):
    optimizer = optim.Adam(model.parameters(), 
                           lr=config['LEARNING_RATE'])
elif (config['OPTIMIZER']=='SGD'):
    optimizer = optim.SGD(model.parameters(), 
                           lr=config['LEARNING_RATE'])
else: #default optimizer is Adam
    optimizer = optim.Adam(model.parameters(), 
                           lr=config['LEARNING_RATE'])


# In[14]:


# CRITERION
if (config['CRITERION']=='MSE'):
    criterion = nn.MSELoss()
elif (config['CRITERION']=='HUBER'):
    criterion = nn.SmoothL1Loss()
else: #default criterion is MSELoss
    criterion = nn.MSELoss()


# In[15]:


# REPLAY BUFFER
replay_buffer = ReplayBuffer(capacity=config['REPLAY_BUFFER_SIZE'])


# In[16]:


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.tensor(np.float32(state)      ,dtype=torch.float32).to(device)
    next_state = torch.tensor(np.float32(next_state) ,dtype=torch.float32, requires_grad=False).to(device)
    action     = torch.tensor(action                ,dtype=torch.long).to(device)
    reward     = torch.tensor(reward                ,dtype=torch.float32).to(device)
    done       = torch.tensor(done                  ,dtype=torch.float32).to(device)

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = criterion(q_value, expected_q_value)
       
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.to('cpu')


# In[17]:


# def plot(frame_idx, rewards, losses):
#     clear_output(True)
#     plt.figure(figsize=(20,5))
#     plt.subplot(131)
#     plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
#     plt.plot(rewards)
#     plt.subplot(132)
#     plt.title('loss')
#     plt.plot(losses)
#     plt.show()


# In[18]:


# Training

num_frames = config['TIMESTEPS']
batch_size = config['BATCH_SIZE']
gamma      = config['GAMMA']

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
        writer.add_scalar('episode_reward', episode_reward, global_step=frame_idx)
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
                

        writer.add_scalar('loss', loss.item(), global_step=frame_idx)
        writer.add_scalar('epsilon', epsilon, global_step=frame_idx)
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(name, param.data, global_step=frame_idx)
        
#     if frame_idx % 200 == 0:
#         plot(frame_idx, all_rewards, losses)


# In[ ]:




