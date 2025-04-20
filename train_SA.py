import gc
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from variables import *
from collections import deque
from breaker import BreakoutEnv
from dqn_SA import CNN_DQN

torch.cuda.empty_cache()
gc.collect()

print(torch.cuda.memory_allocated())
#region Hyperparameters
TRAIN = True               # train or test
LOAD_MODEL = False
SAVE_MODEL = True

BUFFER_SIZE = int(1e4)      # replay buffer size
BATCH_SIZE = 64             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 5e-3                  # for soft update of target parameters
LR_START = 1e-4             # initial learning rate
LR_END = 1e-6               # end epsilon
LR_DECAY = 0.999995         # epsilon decay
EPS_START = 1.0             # initial epsilon
EPS_END = 0.01              # end epsilon
EPS_DECAY = 0.9955           # epsilon decay
UPDATE_EVERY = 5            # how often to hard update the target network
FRAME_SEQ = 2

MAX_TIMESTEPS = 1000
N_EPISODES = 800
SAVE_EVERY = 50             # how often to save the network
# SAVE_DIR = "./!BB/train2/"
SAVE_DIR = "/content/drive/MyDrive/CCU/BB/train_SAT/"
MODEL_SELECT = "model_episode1550.pth"
LOG_SAVE_NAME = "out_SAT"

#endregion


class DDQNAgent:
    def __init__(self, input_shape, num_actions=3, lr=1e-4, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.99, batch_size=32, buffer_size=1e4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.q_network = CNN_DQN(input_shape, num_actions).to(self.device)
        self.target_network = CNN_DQN(input_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.memory = deque(maxlen=buffer_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size

    def select_action(self, state):
        """state 形狀: (batch, seq_len, 1, H, W)"""
        state = state.to(self.device)
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # 打磚塊只有 3 種動作
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                #print(q_values)
                #print(torch.argmax(q_values, dim=1).item())
                return torch.argmax(q_values, dim=1).item()

    def store_experience(self, state_seq, action, reward, next_state, done):
        """存入真實/虛擬經驗池 - 修改為使用4幀"""
        state_seq = state_seq.to(self.device)
        next_state = next_state.to(self.device)
        self.memory.append((state_seq, action, reward, next_state, done))


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def preprocess_frame(self, frame):
        """轉換影像為 Autoencoder 模型輸入格式"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 轉灰階
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)  # (H, W) → (1, H, W)
        return torch.tensor(frame, device=self.device)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        memory = self.memory
        batch_size = self.batch_size
        optimizer = self.optimizer

        batch = random.sample(memory, batch_size)
        state_seq, actions, rewards, next_state, dones = zip(*batch)

        # 將列表中的 sequence（長度=4，每個 frame shape: (1, 1, H, W)）轉換為 tensor
        # 組成的 shape: (batch_size, seq_len, 1, H, W)
        states = torch.stack([torch.cat(list(frames), dim=0) for frames in state_seq], dim=0).to(self.device)

        next_state_single = torch.stack(next_state).to(self.device)
        next_state_single = next_state_single.unsqueeze(1)
        # 利用當前狀態的最後 3 幀，加上新得到的 next_state 來生成新的 4 幀 next state
        next_states = torch.cat((states[:, -(FRAME_SEQ - 1):, :, :, :], next_state_single), dim=1)

        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # DDQN 計算
        q_values = self.q_network(states).gather(1, actions)
        next_actions = torch.argmax(self.q_network(next_states), dim=1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions)

        targets = rewards + (1 - dones) * self.gamma * next_q_values
        loss = nn.MSELoss()(q_values, targets.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# if __name__ == '__main__':
env = BreakoutEnv()
state = env.reset()
steps_per_episode = []
done = False
goal = 0
steps = 0
avg_score = 0
successes = 0
total_steps = 0
state_buffer = deque(maxlen=FRAME_SEQ)
success_history = deque(maxlen=10)


if TRAIN:
    agent = DDQNAgent(input_shape=(4, screen_height, screen_width), num_actions=3,
                    lr=LR_START,
                    epsilon=EPS_START,
                    epsilon_decay=EPS_DECAY,
                    epsilon_min=EPS_END,
                    gamma=GAMMA,
                    batch_size=BATCH_SIZE,
                    buffer_size=BUFFER_SIZE)
#     if LOAD_MODEL:
#         agent.q_network.load_state_dict(
#         torch.load(SAVE_DIR + MODEL_SELECT))
#         agent.target_network.load_state_dict(
#         torch.load(SAVE_DIR + MODEL_SELECT))
# else:
#     agent = DDQNAgent(input_shape=(2, screen_height, screen_width), num_actions=3)
#     agent.q_network.load_state_dict(
#     torch.load(SAVE_DIR + MODEL_SELECT))
#     agent.target_network.load_state_dict(
#     torch.load(SAVE_DIR + MODEL_SELECT))
#     agent.q_network.eval()
#     agent.target_network.eval()

    for episode in range(N_EPISODES):
        torch.cuda.empty_cache()
        gc.collect()
        state = env.reset()
        first_frame = env._get_frame()  # 取得當前畫面
        first_tensor = agent.preprocess_frame(first_frame)
        state_buffer = deque([first_tensor]*FRAME_SEQ, maxlen=FRAME_SEQ)
        steps = 0
        frame_count = 0  # 用來追蹤已經收集的幀數

        for step in range(MAX_TIMESTEPS):
            # print(torch.cuda.memory_allocated())
            current_frame = env._get_frame()  # 取得當前畫面
            current_tensor = agent.preprocess_frame(current_frame)
            state_buffer.append(current_tensor)

            cv2.imshow("Breakout Frame", current_frame)
            cv2.waitKey(1)

            # 組合後形狀: (seq_len, 1, H, W)，再 unsqueeze(0) 變成 (1, seq_len, 1, H, W)
            state_seq = torch.stack(list(state_buffer), dim=0).unsqueeze(0)

            action = agent.select_action(state_seq)
            next_frame, reward, done, dead = env.step(action)
            next_tensor = agent.preprocess_frame(next_frame)
            agent.store_experience(state_seq, action, reward, next_tensor, done)

            agent.train()
            # loss = agent.train()
            # total_loss += loss

            steps += 1

            if done:
                goal += 1
                # agent.epsilon *= 0.95
                break
            if env.dead:
                break

        agent.epsilon = max(agent.epsilon * EPS_DECAY, agent.epsilon_min)

        avg_score = avg_score + env.score
        success_history.append(env.dead)
        success_rate = sum(success_history) / len(success_history)  # 計算成功率

        # update target network
        if episode % UPDATE_EVERY == 0:
            agent.update_target_network()

        # 根據成功率調整 epsilon
        if success_rate < 0.2:  # 如果最近成功率 < 20%
            agent.epsilon = max(agent.epsilon * 0.9, agent.epsilon_min)  # 加速衰減

        if episode % 5 == 0:
            avg_score /= 5
            steps_per_episode.append(avg_score)
            avg_score = 0

        if episode % 10 == 1 :
            plt.plot(steps_per_episode)
            plt.xlabel("Episode")
            plt.ylabel("Total reward")
            plt.title("Steps per Episode over Training")
            plt.show()

        # save the network every episode
        # if episode % SAVE_EVERY == 0 and SAVE_MODEL:
        #     torch.save(agent.q_network.state_dict(), SAVE_DIR + f'model_episode{episode}.pth')

        outStr = "Episode: {}  Steps: {}  Reward: {}  Done: {}  Epsilon: {:.3f}".format(
                        episode, steps, env.score, done, agent.epsilon
                    )
        print(outStr)
        outputPath = SAVE_DIR + LOG_SAVE_NAME + '.txt'
        with open(outputPath, 'a') as outfile:
            outfile.write(outStr+"\n")

    # 繪製學習曲線
    plt.plot(steps_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Steps per Episode over Training")
    plt.show()
    torch.save(agent.q_network.state_dict(), SAVE_DIR + f'model_final.pth')



