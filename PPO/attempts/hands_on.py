import os
import sys
import random

import copy
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader

from typing import Tuple, List
from torchtyping import TensorType


class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()
        self.conf = config
        self.shared_fc = nn.Sequential(
            nn.Linear(self.conf['state_dim'], self.conf['hidden_dim']),
            nn.Tanh(),
            nn.Linear(self.conf['hidden_dim'], self.conf['hidden_dim']),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(self.conf['hidden_dim'], self.conf['action_dim'])
        self.log_std_layer = nn.Linear(self.conf['hidden_dim'], self.conf['action_dim'])
        self.to(self.conf['device'])

    def forward(self, x: TensorType["batch", 27]) -> (TensorType["batch", 8], TensorType["batch", 1]):
        x = self.shared_fc(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)

        return mean, std

    def sample_action(self, x: TensorType[1, 27]) -> (TensorType["batch", 8], TensorType["batch", 1]):
        mean, std = self.forward(x)
        normal_dist = Normal(mean, std)
        sample_action = normal_dist.sample()
        action_log_prob = normal_dist.log_prob(sample_action).sum(
            dim=-1)  # 그 행동 벡터의 log_prob은 각 차원의 확률 밀도 함수의 곱이나, log 형태에서는 합으로 표현됨.

        return sample_action, action_log_prob


class ValueNetwork(nn.Module):
    def __init__(self, config):
        super(ValueNetwork, self).__init__()
        self.conf = config
        self.fc = nn.Sequential(
            nn.Linear(self.conf['state_dim'], self.conf['hidden_dim']),
            nn.Tanh(),
            nn.Linear(self.conf['hidden_dim'], self.conf['hidden_dim']),
            nn.Tanh(),
            nn.Linear(self.conf['hidden_dim'], 1)
        )
        self.to(self.conf['device'])

    def forward(self, x: TensorType["batch", 27]) -> TensorType["batch"]:
        x = self.fc(x)
        return x


def compute_advantages_and_returns(trajectories_: List[List[Tuple[np.ndarray, np.ndarray, np.float32, np.ndarray, np.float32, np.float32]]],
                                   value_nn, config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute advantages and returns for each trajectory.

    Parameters:
    - trajectories_: List of trajectories, where each trajectory is a list of tuples
                    (state, action, reward, next_state, done, action_log_prob)
    - value_nn: Function to compute state values
    - config: Dictionary containing 'gamma', 'lambda_generalized_advantage_edtimation', and 'device'

    Returns:
    - advantages_: NumPy array of computed advantages
    - returns_: NumPy array of computed returns
    """
    advantages_ = []
    returns_ = []

    for trajectory_ in trajectories_:
        G = np.float32(0.0)
        advantage = np.float32(0.0)
        for t in reversed(range(len(trajectory_))):
            state_, action_, reward_, next_state_, done_, action_log_prob_ = trajectory_[t]
            G = reward_ + config['gamma'] * G

            state_ = torch.from_numpy(state_).float().unsqueeze(0).to(config['device'])  # >> Tensor: (1, 27)
            next_state_ = torch.from_numpy(next_state_).float().unsqueeze(0).to(config['device'])  # >> Tensor: (1, 27)
            v_next_state = np.float32(value_nn(next_state_).detach().cpu().numpy().squeeze())  # >> np.float32
            v_state = np.float32(value_nn(state_).detach().cpu().numpy().squeeze())  # >> np.float32

            delta = reward_ + config['gamma'] * v_next_state * (1 - done_) - v_state
            advantage = delta + config['gamma'] * config['lambda_gae'] * advantage
            advantages_.insert(0, advantage.squeeze())
            returns_.insert(0, G)

    advantages_ = np.array(advantages_).astype(np.float32)
    returns_ = np.array(returns_).astype(np.float32)

    return advantages_, returns_


def convert_trajectories_to_batches(trajectories_) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert trajectories into batches of states, actions, dones, and action_log_probs.

    Parameters:
    - trajectories_: List of trajectories, where each trajectory is a list of tuples
                    (state, action, reward, next_state, done, action_log_prob)

    Returns:
    - states_: NumPy array of states
    - actions_: NumPy array of actions
    - dones_: NumPy array of dones
    - action_log_probs_: NumPy array of action_log_probs
    """
    states_ = []
    actions_ = []
    dones_ = []
    action_log_probs_ = []

    for trajectory_ in trajectories_:
        for step_ in trajectory_:
            state_, action_, reward_, next_state_, done_, action_log_prob_ = step_
            states_.append(state_)
            actions_.append(action_)
            dones_.append(done_)
            action_log_probs_.append(action_log_prob_)

    states_ = np.array(states_).astype(np.float32)
    actions_ = np.array(actions_).astype(np.float32)
    dones_ = np.array(dones_).astype(np.float32)
    action_log_probs_ = np.array(action_log_probs_).astype(np.float32)

    return states_, actions_, dones_, action_log_probs_


def train_step(config, batch_, policy_nn, value_nn, policy_optim, value_optim):
    """
    Update policy and value function using the collected batch.

    Parameters:
    - policy_nn: Policy network
    - value_nn: Value function network
    - policy_optim: Optimizer for policy network
    - value_optim: Optimizer for value function network
    - config: Dictionary containing configuration parameters like 'device', 'num_epochs', 'clip_epsilon'
    - batch_: List of batches, where each batch is a tuple of
             (state_batch, action_batch, old_action_log_prob_batch, advantage_batch, return_batch)
    """
    state_batch, action_batch, old_action_log_prob_batch, advantage_batch, return_batch = batch_

    state_batch = state_batch.to(config['device'])  # >> Tensor: (64, 27)
    action_batch = action_batch.to(config['device'])  # >> Tensor: (64, 8)
    old_action_log_prob_batch = old_action_log_prob_batch.to(config['device'])  # >> Tensor: (64,)
    advantage_batch = advantage_batch.to(config['device'])  # >> Tensor: (64,)
    return_batch = return_batch.view(-1, 1).to(config['device'])  # >> Tensor: (64,)

    # Calculate the ratio (πθ(a|s) / πθ_old(a|s))
    mean, std = policy_nn(state_batch)  # >> Tensor: (64, 8) / Tensor: (64, 8)
    normal_dist = Normal(mean, std)
    new_action_log_prob = normal_dist.log_prob(action_batch).sum(dim=-1)  # >> Tensor: (64,)

    ratio = torch.exp(new_action_log_prob - old_action_log_prob_batch)  # >> Tensor: (64,)

    # Compute the surrogate loss
    surrogate1 = ratio * advantage_batch  # >> Tensor: (64,)
    surrogate2 = torch.clip(ratio, 1 - config['clip_epsilon'], 1 + config['clip_epsilon']) * advantage_batch  # >> Tensor: (64,)
    policy_loss = -torch.min(surrogate1, surrogate2).mean()

    # Compute the value function loss
    value_loss = F.mse_loss(value_nn(state_batch), return_batch)

    # Update policy θ using gradient descent
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    # Update value function ϕ using gradient descent
    value_optim.zero_grad()
    value_loss.backward()
    value_optim.step()

    return policy_loss.item(), value_loss.item()


def loading_prms(model_policy, model_value, path: str):
    checkpoint = torch.load(path)
    model_policy.load_state_dict(checkpoint['policy_state_dict'])
    model_value.load_state_dict(checkpoint['value_state_dict'])


def saving_prms(model_policy, model_value, path: str):
    torch.save({
        'policy_state_dict': model_policy.state_dict(),
        'value_state_dict': model_value.state_dict()
    }, path)
    print('>> save complete.')


if __name__ == '__main__':

    # ------------- HYPER-PARAMETERS SETTING ------------- #

    env_name = 'Ant-v4'
    env = gym.make(env_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prm_path = './save_prms/prms.pth'

    conf = OmegaConf.create({
        'device': device,

        'num_iterations': 1000,
        'num_actors': 1,
        'max_timesteps_per_trajectory': 2048,
        'num_epochs': 10,
        'batch_size': 64,
        'policy_lr': 3e-4,  # 파라미터 조정 필요
        'value_lr': 3e-5,  # 파라미터 조정 필요

        'state_dim': int(env.observation_space.shape[0]),
        'hidden_dim': 64,
        'action_dim': int(env.action_space.shape[0]),

        'gamma': 0.99,
        'lambda_gae': 0.95,
        'clip_epsilon': 0.2
    })

    # Initialize policy πθ and value function Vϕ with random weights θ and ϕ
    policy_nn = PolicyNetwork(conf)
    value_nn = ValueNetwork(conf)
    policy_optimizer = torch.optim.Adam(policy_nn.parameters(), lr=conf['policy_lr'])
    value_optimizer = torch.optim.Adam(value_nn.parameters(), lr=conf['value_lr'])

    # Load weights
    if os.path.exists(prm_path):
        loading_prms(policy_nn, value_nn, prm_path)
        print('>> Load prms sucessfully.')
    else:
        print('>> Not found prms.')

    # ------------- START TRAINING ------------- #
    log_total_rewards, log_p_losses, log_v_losses = [], [], []
    for iteration in tqdm(range(1, conf['num_iterations'] + 1)):
        # Collect set of trajectories {τ_i} by running policy πθ in the environment
        trajectories = []
        for actor in range(conf['num_actors']):
            trajectory, total_reward = [], 0
            state, _ = env.reset()  # >> ndarray: (27,)
            for t in range(conf['max_timesteps_per_trajectory']):
                # Sample action from policy
                state = torch.from_numpy(state).float().unsqueeze(0).to(conf['device'])  # >> Tensor: (1, 27)

                action, action_log_prob = policy_nn.sample_action(state)  # >> Tensor: (1, 8) / Tensor: (1,)

                action = action.detach().cpu().squeeze().numpy()  # >> ndarray: (8,)
                action_log_prob = np.float32(action_log_prob.detach().cpu().numpy().squeeze())  # float32: ()

                next_state, reward, terminated, truncated, _ = env.step(action)
                # >> ndarray: (27,) / float64: () / bool / bool / _

                total_reward += reward
                done = terminated or truncated
                trajectory.append((
                    state.cpu().squeeze().numpy(),  # >> ndarray: (27,)
                    action,  # >> ndarray: (8,)
                    np.float32(reward),  # >> float32: ()
                    next_state,  # >> ndarray: (27,)
                    np.float32(done),  # >> float32: ()
                    action_log_prob  # >> float32: ()
                                   ))
                state = next_state
                if done:
                    break
            trajectories.append(trajectory)
            log_total_rewards.append(total_reward)

        # Compute advantages and returns for each trajectory
        advantages, returns = compute_advantages_and_returns(trajectories, value_nn, conf)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # >> ndarray: (1000) / ndarray: (1000)

        # Convert trajectories into batches
        states, actions, dones, action_log_probs = convert_trajectories_to_batches(trajectories)
        # >> ndarray: (1000, 27) / ndarray: (1000, 8) / ndarray: (1000,) / ndarray: (1000,)

        dataset = TensorDataset(torch.from_numpy(states).float(), torch.from_numpy(actions).float(),
                                torch.from_numpy(action_log_probs).float(), torch.from_numpy(advantages).float(),
                                torch.from_numpy(returns).float())
        generate_batches = DataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)

        # Update policy and value function using the collected batch
        for epoch in range(conf['num_epochs']):
            for batch in generate_batches:
                p_loss, v_loss = train_step(conf, batch, policy_nn, value_nn, policy_optimizer, value_optimizer)
                log_p_losses.append(p_loss)
                log_v_losses.append(v_loss)

        # Update old policy to current policy
        # πθ_old.load_state_dict(πθ.state_dict())
        print(f'\niter{iteration}// t_rew: {log_total_rewards[-1]}, p_loss: {log_p_losses[-1]}, v_loss: {log_v_losses[-1]}')

    env.close()
    saving_prms(policy_nn, value_nn, prm_path)
