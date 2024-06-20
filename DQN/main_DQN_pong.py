import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# 상태 전처리 함수
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame

# DQN 네트워크 정의
class DQN(nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 경험 재플레이 버퍼 설정
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


# 상태 변환 함수
def stack_frames(frames, new_frame, is_new_episode):
    new_frame = np.expand_dims(new_frame, axis=0)
    if is_new_episode:
        frames = np.repeat(new_frame, 4, axis=0)
    else:
        frames = np.concatenate((frames[1:], new_frame), axis=0)
    return frames

# TD 손실 계산 함수
def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values = policy_net(state)
    next_q_values = target_net(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

if __name__ == "__main__":

    # 환경 설정
    env = gym.make("ALE/Pong-v5")

    # DQN 학습을 위한 하이퍼파라미터 설정
    gamma = 0.99
    batch_size = 32
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 0.000099
    target_update_interval = 1000
    replay_buffer_capacity = 100000
    learning_rate = 0.01  # 0.0001

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 네트워크 및 버퍼 생성
    action_space = env.action_space.n
    policy_net = DQN(action_space).to(device)
    target_net = DQN(action_space).to(device)
    policy_net.load_state_dict(torch.load("./drive/MyDrive/dqn_pong.pth"))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    # Epsilon-greedy 전략
    epsilon = epsilon_start

    # 로그 저장을 위한 리스트 초기화
    episode_rewards = []
    losses = []

    # 학습 루프
    num_episodes = 1000

    for episode in tqdm(range(num_episodes)):
        state = preprocess_frame(env.reset()[0])
        frames = stack_frames(None, state, True)
        episode_reward = 0

        for t in range(10000):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(frames).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_frame(next_state)
            next_frames = stack_frames(frames, next_state, False)
            replay_buffer.push(frames, action, reward, next_frames, done)

            frames = next_frames
            episode_reward += reward

            if done:
                break

            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size)
                losses.append(loss)

            if t % target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_final, epsilon - epsilon_decay)
        episode_rewards.append(episode_reward)

        if episode % 10 == 0:
            torch.save(policy_net.state_dict(), f"./drive/MyDrive/dqn_pong.pth")
            print(f"Episode {episode}, Reward: {episode_reward}")

    env.close()

    # 손실 및 에피소드 리워드 플롯
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Loss over Time")
    plt.plot(losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.title("Episode Rewards over Time")
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.show()