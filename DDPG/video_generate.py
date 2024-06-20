import os
import copy
import random
import collections
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt

from omegaconf import OmegaConf


def loading_prms(agent, path: str):
    checkpoint = torch.load(path)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor_target.load_state_dict(checkpoint['target_actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.critic_target.load_state_dict(checkpoint['target_critic_state_dict'])


class ActorCritic(nn.Module):
    @torch.no_grad()
    def __init__(self, config):
        super().__init__()
        self.data = []
        self.config = config

        # actor: policy network
        self.actor = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim),
            nn.Tanh(), # continuous action, bound output to [-1, 1]
        )

        # critic: Q(s, a) network
        self.critic = nn.Sequential(
            nn.Linear(self.config.state_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

        # we need target networks:
        self.actor_target, self.critic_target = copy.deepcopy(self.actor), copy.deepcopy(self.critic)

        # load them to gpu (if available)
        self.to(self.config.device)


prm_path = 'save_prms/last_attempt.pth'

env_name = 'Hopper-v4'
env = gym.make(env_name, render_mode='rgb_array')
env = gym.wrappers.RecordVideo(env, video_folder='./videos')

AC_config = OmegaConf.create({
    # RL parameter
    'gamma': 0.99,

    # replay memory
    'buffer_limit': int(1e5),
    'batch_size': 32,

    # neural network parameters
    'device': 'cpu',
    'hidden_dim': 64,
    'state_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.shape[0],

    # learning parameters
    'lr_actor': 0.0005,
    'lr_critic': 0.001,
    'tau': 0.005,
})

agent = ActorCritic(AC_config)

if os.path.exists(prm_path):
    loading_prms(agent, prm_path)
    print('>> Load prms sucessfully.')
else:
    print('>> Not found prms.')

s, _ = env.reset()
terminated, truncated, epi_rew = False, False, 0

while not (terminated or truncated):
    # get action from actor network
    s_tensor = torch.from_numpy(s).float().unsqueeze(0)  # (1, state_dim) = (1, 17)
    a = agent.actor(s_tensor).detach().numpy()[0]  # (action_dim,) = (6,)

    n_s, r, terminated, truncated, _ = env.step(a)

    # state transition
    s = n_s

    # record reward
    epi_rew += r

env.close()

print(epi_rew)