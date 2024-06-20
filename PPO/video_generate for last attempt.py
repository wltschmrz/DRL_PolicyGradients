import os
import random

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(512,), activation_fn=F.relu):
        super(MLPGaussianPolicy, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.mu_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))

        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))

        return mu, log_std.exp()


class MLPStateValue(nn.Module):
    def __init__(self, state_dim, hidden_dims=(512,), activation_fn=F.relu):
        super(MLPStateValue, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)

        return x


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64), activation_fn=torch.tanh,
                 n_steps=2048, n_epochs=10, batch_size=64, policy_lr=0.0003, value_lr=0.0003,
                 gamma=0.99, lmda=0.95, clip_ratio=0.2, vf_coef=1.0, ent_coef=0.01,
                 ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = MLPGaussianPolicy(state_dim, action_dim, hidden_dims, activation_fn).to(self.device)
        self.value = MLPStateValue(state_dim, hidden_dims, activation_fn).to(self.device)

    @torch.no_grad()
    def act(self, s, training=True):
        self.policy.train(training)

        s = torch.as_tensor(s, dtype=torch.float, device=self.device)
        mu, std = self.policy(s)
        z = torch.normal(mu, std) if training else mu
        action = torch.tanh(z)

        return action.cpu().numpy()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loading_prms(agent, path: str):
    checkpoint = torch.load(path)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.value.load_state_dict(checkpoint['value_state_dict'])


prm_path = './save_prms/last_attempt.pth'

env_name = 'Ant-v4'

seed = 42
seed_all(seed)

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPO(
    state_dim,
    action_dim
)

if os.path.exists(prm_path):
    loading_prms(agent, prm_path)
    print('>> Load prms sucessfully.')
else:
    print('>> Not found prms.')

env = gym.make('Ant-v4', render_mode='rgb_array')
env = gym.wrappers.RecordVideo(env, video_folder='./videos')
agent.policy.eval()
agent.value.eval()
total_reward = 0
s, _ = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    a = agent.act(s)
    s_prime, r, terminated, truncated, _ = env.step(a)
    s = s_prime
    total_reward += r
env.close()
print(total_reward)