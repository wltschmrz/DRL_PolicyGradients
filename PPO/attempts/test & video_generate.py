# Record video of the trained agent
import os
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
from omegaconf import OmegaConf

class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()
        self.conf = config
        self.shared_fc = nn.Sequential(
            nn.Linear(self.conf['state_dim'], self.conf['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.conf['hidden_dim'], self.conf['hidden_dim']),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(self.conf['hidden_dim'], self.conf['action_dim'])
        self.log_std_layer = nn.Linear(self.conf['hidden_dim'], self.conf['action_dim'])
        self.to(self.conf['device'])

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.shared_fc(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)

        return mean, std

    def sample_action(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
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
            nn.ReLU(),
            nn.Linear(self.conf['hidden_dim'], self.conf['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.conf['hidden_dim'], 1)
        )
        self.to(self.conf['device'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x


def loading_prms(model_policy, model_value, path: str):
    checkpoint = torch.load(path)
    model_policy.load_state_dict(checkpoint['policy_state_dict'])
    model_value.load_state_dict(checkpoint['value_state_dict'])


if __name__ == '__main__':
    os.environ['MUJOCO_GL'] = 'glfw'
    env = gym.make('Ant-v4', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, video_folder='./videos')

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
        'value_lr': 3e-4,  # 파라미터 조정 필요

        'state_dim': int(env.observation_space.shape[0]),
        'hidden_dim': 64,
        'action_dim': int(env.action_space.shape[0]),

        'gamma': 0.99,
        'lambda_gae': 0.95,
        'clip_epsilon': 0.2
    })

    # Initialize policy πθ and value function Vϕ with random weights θ and ϕ
    πθ = PolicyNetwork(conf)
    Vϕ = ValueNetwork(conf)

    try:
        loading_prms(πθ, Vϕ, prm_path)
        print('Load prms sucessfully.')
    except Exception as e:
        print('Not found prms.')

    πθ.eval()
    Vϕ.eval()

    total_reward = 0

    s, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
        a, _ = πθ.sample_action(s)
        s, r, terminated, truncated, _ = env.step(a.detach().cpu().squeeze().numpy())
        total_reward += r
    env.close()
    print(total_reward)