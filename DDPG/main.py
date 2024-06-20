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


# replay buffer
class ReplayBuffer():
    def __init__(self, config):
        self.config = config
        self.buffer = collections.deque(maxlen=self.config.buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, next_s_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, next_s, done = transition
            s_lst.append(s.tolist())
            a_lst.append(a.tolist())
            r_lst.append([r])
            next_s_lst.append(next_s.tolist())
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.Tensor(s_lst), torch.Tensor(a_lst), torch.Tensor(r_lst), torch.Tensor(next_s_lst), torch.Tensor(
            done_mask_lst)

    def size(self):
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


# moving average over the neural network parameters
def soft_update(net, net_target, tau):
    # for each parameters,
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        # mix the target and current parameters with the ratio of (1 - tau) : (tau)
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.data = []
        self.config = config

        # create replay buffer
        self.memory = ReplayBuffer(self.config)
        # set exploration noise
        self.action_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.config.action_dim))

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

        # we use different learning rates for actor and critic networks
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

        # parameter for soft update
        self.tau = self.config.tau

    # training function
    def update(self):
        # randomly sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)

        # compute target q values
        target_q_values = rewards + self.config.gamma * dones * self.critic_target(torch.cat([next_states, self.actor_target(next_states)], dim=1))  # (32, 1)

        # compute q loss
        critic_loss = F.mse_loss(self.critic(torch.cat([states, actions], dim=1)), target_q_values)  # Scalar

        # compute gradient & update
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # compute actor loss
        actor_loss = - self.critic(torch.cat([states, self.actor(states)], dim=1)).mean()

        # compute gradient & update
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft update
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)

        return actor_loss.item(), critic_loss.item()


def loading_prms(agent, path: str):
    checkpoint = torch.load(path)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor_target.load_state_dict(checkpoint['target_actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.critic_target.load_state_dict(checkpoint['target_critic_state_dict'])


def saving_prms(agent, path: str):
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'target_actor_state_dict': agent.actor_target.state_dict(),
        'target_critic_state_dict': agent.critic_target.state_dict(),
    }, prm_path)
    print('>> save complete.')


if __name__ == "__main__":

    # continuous environment
    env = gym.make('Hopper-v4')

    prm_path = 'save_prms/last_attempt.pth'

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
        'lr_critic': 0.003,
        'tau': 0.005,
    })

    num_epis, epi_rews, ac_losses, cri_losses = 80000, [], [], []
    agent = ActorCritic(AC_config)

    if os.path.exists(prm_path):
        loading_prms(agent, prm_path)
        print('>> Load prms sucessfully.')
    else:
        print('>> Not found prms.')

    for n_epi in tqdm(range(num_epis)):
        state, _ = env.reset()
        terminated, truncated = False, False
        epi_rew = 0

        while not (terminated or truncated):
            # get action from actor network
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # (1, state_dim) = (1, 17)
            action = agent.actor(state_tensor).detach().numpy()[0]  # (action_dim,) = (6,)

            # add noise for better exploration
            action = action + agent.action_noise()  # (action_dim,) = (6,)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # save transition to replay buffer
            agent.memory.put((state, action, reward, next_state, terminated or truncated))

            # state transition
            state = next_state

            # record reward
            epi_rew += reward

        # enough memory
        if agent.memory.size() > 5000:
            # off-line training
            for i in range(10):
                ac_loss, cri_loss = agent.update()
                ac_losses.append(ac_loss)
                cri_losses.append(cri_loss)
        if n_epi % 1000 == 0:
            saving_prms(agent, prm_path)
            print(epi_rew)

        epi_rews += [epi_rew]

    env.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    y_lists = [epi_rews, ac_losses, cri_losses]
    titles = ["epi_rews", "actor_losses", "critic_losses"]
    colors = ['b-', 'b-', 'b-']

    for ax, y, title, color in zip(axes, y_lists, titles, colors):
        ax.plot(y, color, label=title)
        ax.set_title(title)
        ax.set_xlabel("episode number")
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    plt.show()



