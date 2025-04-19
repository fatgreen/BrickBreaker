import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from breaker import BreakoutEnv
from collections import deque
from variables import *
from dqn_BB import CNN_DQN
from collections import deque

#region Hyperparameters
TRAIN = True               # train or test
LOAD_MODEL = False
SAVE_MODEL = True

BUFFER_SIZE = int(5e4)      # replay buffer size
BATCH_SIZE = 8              # minibatch size
GAMMA = 0.99                # discount factor
TAU = 5e-3                  # for soft update of target parameters
LR_START = 1e-4             # initial learning rate
EPS_START = 1.0             # initial epsilon 
EPS_END = 0.01              # end epsilon
EPS_DECAY = 0.995           # epsilon decay
UPDATE_EVERY = 5            # how often to hard update the target network

LR_END = 1e-6               # end epsilon
LR_DECAY = 0.999995         # epsilon decay

MAX_TIMESTEPS = 1000
N_EPISODES = 500
SAVE_DIR = "./!BB/train2/"
MODEL_SELECT = "model_episode1550.pth"
LOG_SAVE_NAME = "out2"

SAVE_EVERY = 50            # how often to save the network
#endregion


class DDQNAgent:
    def __init__(self, input_shape, num_actions, lr=0.0001, gamma=0.99, buffer_size=6000, batch_size=16):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.q_network = CNN_DQN(input_shape, num_actions).to(self.device)
        self.target_network = CNN_DQN(input_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.memory = deque(maxlen=buffer_size)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        state = state.to(self.device)  
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # 打磚塊只有 3 種動作
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                #print(q_values)
                #print(torch.argmax(q_values, dim=1).item())
                return torch.argmax(q_values, dim=1).item()
            
    def store_experience(self, frames, action, reward, next_state, done):
        """存入真實/虛擬經驗池 - 修改為使用3幀"""
        # 確保所有張量都在正確的設備上
        frames = [frame.to(self.device) for frame in frames]
        next_state = next_state.to(self.device)
    
        memory = self.memory
        memory.append((frames, action, reward, next_state, done))

    def train(self):
        memory = self.memory
        batch_size = self.batch_size
        optimizer = self.optimizer
    
        if len(memory) < batch_size:
            return
    
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
    
        # 處理每個batch中的3幀 - 修正維度問題
        # 正確的組合方式是先確保每個幀都是正確的形狀
        prev_frame1, current_frame = zip(*states)  

        prev_frame1 = torch.stack(prev_frame1).squeeze(1).to(self.device) 
        current_frame = torch.stack(current_frame).squeeze(1).to(self.device) 
        
        # **拼接成 (batch_size, 2, H, W)**
        states = torch.cat(( prev_frame1, current_frame), dim=1)

        # 處理next_states - 確保每個next_state都是 [1, 1, H, W]
        next_state = torch.stack(next_states).squeeze(1).to(self.device)  # (batch_size, 1, H, W)
        next_states_copy  = next_state.clone()

        next_states = torch.cat((next_state, next_states_copy), dim=1)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
    
        # DQN 計算
        q_values = self.q_network(states).gather(1, actions)
        next_actions = torch.argmax(self.q_network(next_states), dim=1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions)
    
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        loss = nn.MSELoss()(q_values, targets.detach())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def preprocess_frame(self, frame):
        """轉換影像為 Autoencoder 模型輸入格式"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 轉灰階
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)  # (H, W) → (1, H, W)
        frame = np.expand_dims(frame, axis=0)  # (1, H, W) → (1, 1, H, W)
        return torch.tensor(frame, device=self.device)


# if __name__ == '__main__':
env = BreakoutEnv()
state = env.reset()
steps_per_episode = []  
goal = 0
steps = 0
avg_score = 0
successes = 0
total_steps = 0
success_history = deque(maxlen=10)


if TRAIN:
    agent = DDQNAgent(input_shape=(2, screen_height, screen_width), num_actions=3,
                    lr=LR_START, 
                    gamma=GAMMA, 
                    buffer_size=BUFFER_SIZE, 
                    batch_size=BATCH_SIZE) 
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
        state = env.reset()
        frame_buffer = []
        steps = 0
        frame_count = 0  # 用來追蹤已經收集的幀數
        
        for step in range(MAX_TIMESTEPS):
            current_frame = env._get_frame()  # 取得當前畫面
            current_tensor = agent.preprocess_frame(current_frame)
            
            # 收集2幀
            frame_buffer.append(current_tensor)
            frame_count += 1
            
            cv2.imshow("Breakout Frame", current_frame)
            cv2.waitKey(1)
            
            # 只有在收集到4幀後才選擇動作並訓練
            if frame_count % 2 == 0:
                # 確保我們有2幀
                if len(frame_buffer) > 2:
                    frame_buffer = frame_buffer[-2:]  # 只保留最後2幀
                
                # 如果還沒收集到2幀，等待
                if len(frame_buffer) < 2:
                    continue
                    
                # 將2幀堆疊在一起
                state_input = torch.cat(frame_buffer, dim=1)  # (1, 4, H, W)
                
                action = agent.select_action(state_input)
                
                next_frame, reward, done, dead = env.step(action)
                next_tensor = agent.preprocess_frame(next_frame)
                agent.store_experience(frame_buffer, action, reward, next_tensor, done)
                
                # 訓練模型
                agent.train()
                # total_loss += loss
                
                # cv2.imshow("Breakout Frame", current_frame)
                steps += 1
            
                if done:
                    goal += 1
                    # agent.epsilon *= 0.95
                    break
                if env.dead:
                    break
            
        # update target network
        if episode % UPDATE_EVERY == 0:
            agent.update_target_network()
        
        agent.epsilon = max(agent.epsilon * EPS_DECAY, agent.epsilon_min)
        
        avg_score = avg_score + env.score
        success_history.append(dead)  
        success_rate = sum(success_history) / len(success_history)  # 計算成功率
        
        # 根據成功率調整 epsilon
        if success_rate < 0.2:  # 如果最近成功率 < 20%
            agent.epsilon = max(agent.epsilon * 0.9, agent.epsilon_min)  # 加速衰減
        
        if episode % 5 == 0:
            avg_score /= 5
            steps_per_episode.append(avg_score) 
            avg_score = 0
        
        # if episode % 10 == 1 :
            # plt.plot(steps_per_episode)
            # plt.xlabel("Episode")
            # plt.ylabel("Total reward")
            # plt.title("Steps per Episode over Training")
            # plt.show()
        
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



