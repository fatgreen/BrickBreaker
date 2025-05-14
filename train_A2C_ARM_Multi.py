# region Import Libraries
import gc
import cv2
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

from envir.variables import *
from envir.breaker import BreakoutEnv
# from variables import *
# from breaker import BreakoutEnv

from collections import deque
# from A2C_ARM import *
from A2C import *
from ARM import *
#endregion

#region Hyperparameters
TRAIN = True               # train or test
LOAD_MODEL = False
SAVE_MODEL = True

# Optimizer
GAMMA = 0.99                
LR_CRT = 1e-4             
LR_ACT = 1e-3
ENTROPY_COEF = 0.05
EPS_START = 1.0             
FRAME_SEQ = 2

# ARM
HID_SIZE = 256

# Training Settings
N_ENVS = 8
UPDATE_EVERY = 15
SAVE_EVERY = 1000            # how often to save the network
MAX_TIMESTEPS = 1000
N_EPISODES = int(5e3)
TEST_EPISODES = 10
RNN_MODE = "ARM"  

# Save and Load Settings
SAVE_DIR = "./output_A2C/"
# SAVE_DIR = "/content/drive/MyDrive/CCU/BB/Out_A2C/"
MODEL_SELECT = "model_episode250.pth"
LOG_SAVE_NAME = "out_A2C"
#endregion

torch.cuda.empty_cache()
gc.collect()
torch.autograd.set_detect_anomaly(True)

"""A2C Agent with ARM"""
class ACAgent:
    def __init__(self, input_shape, num_actions, actor_lr, critic_lr, gamma):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.hid_size = HID_SIZE

        self.hx0 = torch.zeros(1, self.hid_size, device=self.device)
        self.cx0 = torch.zeros(1, self.hid_size, device=self.device)
        self.hidden = (self.hx0, self.cx0)
        self.hidden_learn = (self.hx0, self.cx0)
        self.hidden_Nlearn = (self.hx0, self.cx0)
        self.alpha_prev = torch.zeros(1, self.hid_size, device=self.device)
        self.alpha_learn = torch.zeros(1, self.hid_size, device=self.device)
        self.alpha_Nlearn = torch.zeros(1, self.hid_size, device=self.device)

        self.cnn = CNN_LAYER(input_shape).to(self.device)
        self.proj = nn.Linear(36864, self.hid_size).to(self.device)  # (1, 36864) -> (1,256)

        if RNN_MODE == "ARM":
            self.rnn = ARM(input_size = self.hid_size, hidden_size = self.hid_size, num_env=1, device = self.device).to(self.device)
        elif RNN_MODE == "LSTM":
            self.rnn = nn.LSTMCell(input_size=36864, hidden_size=self.hid_size).to(self.device)
        elif RNN_MODE == "GRU":
            self.rnn = nn.GRUCell(input_size=36864, hidden_size=256).to(self.device)
        else:
            raise ValueError("Invalid RNN mode. Choose from 'ARM', 'LSTM', or 'GRU'.")
        
        self.actor = Actor(self.hid_size, num_actions).to(self.device)
        self.critic = Critic(self.hid_size).to(self.device)

        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer  = torch.optim.Adam(
            list(self.rnn.parameters()) + list(self.actor.parameters()), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.rnn.parameters()) + list(self.critic.parameters()), lr=critic_lr)

    def preprocess_frame(self, frame):
        """State frame preprocess"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)  # (H, W) → (1, H, W)
        frame = np.expand_dims(frame, axis=0)  # (H, W) → (1, 1, H, W)
        return torch.tensor(frame, device=self.device)
    
    def take_action(self, state):
        state = state.to(self.device)
        with torch.no_grad(): 
            feat = self.cnn(state)
            feat = feat.view(feat.size(0),-1)
            feat = self.proj(feat)  
            (hx_new, Yt), alpha_new, _, _ = self.rnn(feat, self.hidden, self.alpha_prev)
        Yt = Yt.view(Yt.size(0), -1)
        self.hidden     = (hx_new.detach(), Yt.detach())    # 更新 hidden
        self.alpha_prev = alpha_new.detach()       # 更新 alpha
        
        probs = self.actor(Yt)  # (1, num_actions)
        # 创建以probs为标准类型的数据分布
        dist = torch.distributions.Categorical(probs)
        # 随机选择一个动作 tensor-->int
        action = dist.sample()                            # Tensor([a])
        log_prob = dist.log_prob(action)                  # Tensor([log π(a|s)])
        entropy  = dist.entropy()                         # Tensor([H(π(·|s))])
        return action.item(), log_prob.detach().cpu(), entropy.detach().cpu()

    # 模型更新
    def learn(self, transition_dict):
        # Rollout buffer loading
        # actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        states = torch.stack(transition_dict['states'],dim=0).to(self.device)
        log_probs  = torch.stack(transition_dict['log_probs'],dim=0).to(self.device) 
        entropies = torch.stack(transition_dict['entropies'],dim=0).to(self.device)  # (T, )
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.stack(transition_dict['next_states'],dim=0).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        """Critic:  state value Q(s) Q(s')"""
        # Q(s) with ARM
        T, C1, C2, H, W = states.shape
        hidden = self.hidden_learn
        alpha_prev = self.alpha_learn
        Yt_list = []

        for t in range(T):              # T * (1 ,1 ,H, W)
            frame = states[t]           # (1, 1, H, W)
            feat = self.cnn(frame)
            feat = feat.view(feat.size(0),-1)
            feat = self.proj(feat)      # (1, 36864) -> (1, 256)
            
            (hx_new, Yt), alpha_new, _, _ = self.rnn(feat, hidden, alpha_prev)
            hidden     = (hx_new, Yt)    # 更新 hidden
            alpha_prev = alpha_new       # 更新 alpha
            Yt_list.append(Yt)
        self.hidden_learn = (hidden[0].detach(), hidden[1].detach())
        self.alpha_learn = alpha_prev.detach()
        
        Yt_tensor = torch.cat(Yt_list, dim=0)
        values = self.critic(Yt_tensor.detach())        # (B,1)


        # Q(s') with ARM
        hidden = self.hidden_Nlearn
        alpha_prev = self.alpha_Nlearn
        Yt_list = []
        for t in range(T):              # T * (1 ,1 ,H, W)
            frame = next_states[t]           # (1, 1, H, W)
            feat = self.cnn(frame)
            feat = feat.view(feat.size(0),-1)
            feat = self.proj(feat)      # (1, 36864) -> (1, 256)
            
            (hx_new, Yt), alpha_new, _, _ = self.rnn(feat, hidden, alpha_prev)
            hidden     = (hx_new, Yt)    # 更新 hidden
            alpha_prev = alpha_new       # 更新 alpha
            Yt_list.append(Yt)
        self.hidden_Nlearn = (hidden[0].detach(), hidden[1].detach())
        self.alpha_Nlearn = alpha_prev.detach()

        Yt_tensor = torch.cat(Yt_list, dim=0)
        next_values = self.critic(Yt_tensor.detach())   # (B,1)

        # TD target (Q)：r + γ * V(s') * (1 - done)
        td_targets = rewards + self.gamma * next_values * (1 - dones)

        # TD error (Advantage) = target - V(s)
        advantage = td_targets - values
        # Normalize advantage
        T = advantage.numel()
        if T > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Critic loss: MSE
        critic_loss = F.mse_loss(values, td_targets.detach())

        # Actor loss: -E[ log π(a|s) * TD_error ]
        # 其中 advantage.detach() 防止梯度回流到 Critic
        policy_loss = - (log_probs * advantage.detach()).mean()
        entropy_loss = - entropies.mean()  # 我们想最大化熵，所以在 loss 里是负项
        actor_loss  = policy_loss + ENTROPY_COEF * entropy_loss

        total_loss = actor_loss + critic_loss
        # print("policy loss: {:.3f}".format(policy_loss))
        # print("entropy loss:{:.3f}".format(entropy_loss))
        # print("actor loss:{:.3f}".format(actor_loss))
        # print("critic loss:{:.3f}".format(critic_loss))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return actor_loss, critic_loss          # actor loss 非常小


def show_multi(frames, window_name="MultiEnv", cols=4):
    """
    frames: list 或 array，长度 N， 每帧 shape = (H, W) 或 (H, W, C)
    cols: 每行展示多少列
    """
    N = len(frames)
    rows = math.ceil(N/cols)
    # 如果画面不够，补全空白帧
    blank = np.zeros_like(frames[0])
    while len(frames) < rows*cols:
        frames.append(blank)

    # 按行拼接
    row_imgs = []
    for r in range(rows):
        row = cv2.hconcat(frames[r*cols:(r+1)*cols])
        row_imgs.append(row)
    grid = cv2.vconcat(row_imgs)

    cv2.imshow(window_name, grid)
    cv2.waitKey(1)

# region Initialize environment
# env = BreakoutEnv()
# state = env.reset()
envs  = [BreakoutEnv() for _ in range(N_ENVS)]
states = [env.reset() for env in envs]  # list of length NUM_ENVS
steps_per_episode = []
done = False
goal = 0
steps = 0
avg_score = 0
successes = 0
total_steps = 0
return_list = []  
act_loss_buffer = []
crt_loss_buffer = []
living_history = deque(maxlen=10)
transition_dict = [ {
    'states': [], 'actions': [], 'next_states': [],
    'log_probs': [], 'entropies': [], 'rewards': [], 'dones': []
} for _ in range(N_ENVS) ]
# endregion

# ———— MAIN ————
if __name__ == "__main__":
    if TRAIN:
        agent = ACAgent(input_shape=(1, screen_height, screen_width), 
                        num_actions=3,
                        actor_lr=LR_ACT, 
                        critic_lr=LR_CRT, 
                        gamma=GAMMA)

        for episode in range(N_EPISODES):
            """
            Initialize environment every episode
            """
            torch.cuda.empty_cache()
            gc.collect()
            states = [env.reset() for env in envs]
            steps = [0 for _ in range(N_ENVS)]
            done = False

            transition_dict = [ {
                'states': [], 'actions': [], 'next_states': [],
                'log_probs': [], 'entropies': [], 'rewards': [], 'dones': []
            } for _ in range(N_ENVS) ]
            
            act_loss_buffer.clear()
            crt_loss_buffer.clear()
            act_loss_buffer = []
            crt_loss_buffer = []

            # Initialize parameters for ARM every start at the episode
            hx0 = torch.zeros(1, HID_SIZE, device=agent.device)
            cx0 = torch.zeros(1, HID_SIZE, device=agent.device)
            agent.hidden = (hx0, cx0)
            agent.hidden_learn = (hx0, cx0)
            agent.hidden_Nlearn = (hx0, cx0)
            agent.alpha_prev = torch.zeros(1, HID_SIZE, device=agent.device)
            agent.alpha_learn = torch.zeros(1, HID_SIZE, device=agent.device)
            agent.alpha_Nlearn = torch.zeros(1, HID_SIZE, device=agent.device)

            for step in range(MAX_TIMESTEPS):
                # get frame from envs
                current_frames_multi, current_tensors_multi = [], []
                for i in range(N_ENVS):
                    current_frame = envs[i]._get_frame()
                    current_tensor = agent.preprocess_frame(current_frame)      # (1,1,192,192)
                    current_frames_multi.append(current_frame)
                    current_tensors_multi.append(current_tensor)
                current_tensors = torch.cat(current_tensors_multi, dim=0)       # (N,1,192,192)
                print(current_tensors.shape)
                
                show_multi(current_frames_multi, window_name="MultiEnv", cols=4)
                # cv2.imshow("Breakout Frame", current_frame)
                # cv2.waitKey(1)

                # take action and step
                next_tensor_multi, dead_multi = [], []
                for i in range(N_ENVS):
                    action, logp, ent = agent.take_action(current_tensors[i])
                    next_frame, reward, done, dead = envs[i].step(action)
                    next_tensor = agent.preprocess_frame(next_frame)
                    next_tensor_multi.append(next_tensor)

                    transition_dict[i]['states'].append(current_tensors[i])
                    transition_dict[i]['actions'].append(action)
                    transition_dict[i]['next_states'].append(next_tensor)
                    transition_dict[i]['log_probs'].append(logp)
                    transition_dict[i]['entropies'].append(ent)
                    transition_dict[i]['rewards'].append(reward)
                    transition_dict[i]['dones'].append(done)
                    steps[i] += 1
                next_tensors = torch.cat(next_tensor_multi, dim=0)

                if step % UPDATE_EVERY == 0:
                    """
                    Learn every UPDATE_EVERY steps
                    """
                    loss_act, loss_crt = agent.learn(transition_dict)
                    act_loss_buffer.append(loss_act.item())
                    crt_loss_buffer.append(loss_crt.item())
                    transition_dict.clear()
                    transition_dict = [ {
                        'states': [], 'actions': [], 'next_states': [],
                        'log_probs': [], 'entropies': [], 'rewards': [], 'dones': []
                    } for _ in range(N_ENVS) ]

                if done:
                    goal += 1
                    break
                if env.dead:
                    break

            """Epsiode end"""

            avg_score = avg_score + env.score
            living_history.append(1-env.dead)
            living_rate = np.mean(living_history)
            return_list.append(env.score)

            if episode % 5 == 0:
                avg_score /= 5
                steps_per_episode.append(avg_score)
                avg_score = 0

            if episode % 2000 == 1 :
                plt.plot(steps_per_episode)
                plt.xlabel("Episode")
                plt.ylabel("Total reward")
                plt.title("Steps per Episode over Training")
                plt.show()

            # save the network every episode
            if episode % SAVE_EVERY == 0 and SAVE_MODEL:
                torch.save(agent.actor.state_dict(), SAVE_DIR + f'actor_model_episode{episode}.pth')
                torch.save(agent.critic.state_dict(), SAVE_DIR + f'critic_model_episode{episode}.pth')

            outStr = "Episode: {}  Steps: {}  Reward: {}  Living_Rate/10: {:.3f}  Done: {} | Actor_loss: {:.3f}  Critic_loss: {:.3f}".format(
                            episode, steps, env.score, living_rate, done, np.mean(act_loss_buffer), np.mean(crt_loss_buffer))
            print(outStr)
            outputPath = SAVE_DIR + LOG_SAVE_NAME + '.txt'
            with open(outputPath, 'a') as outfile:
                outfile.write(outStr+"\n")
        """Training end"""
        # 繪製學習曲線
        plt.plot(steps_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.title("Steps per Episode over Training")
        plt.show()
        torch.save(agent.q_network.state_dict(), SAVE_DIR + f'model_final.pth')
    
    else:
        """Test the model"""
        agent = ACAgent(input_shape=(1, screen_height, screen_width), 
                        num_actions=3,
                        actor_lr=LR_ACT, 
                        critic_lr=LR_CRT, 
                        gamma=GAMMA)
        agent.actor.load_state_dict(torch.load(SAVE_DIR + MODEL_SELECT))
        agent.actor.eval()
        agent.critic.load_state_dict(torch.load(SAVE_DIR + MODEL_SELECT))
        agent.critic.eval()

        for episode in range(TEST_EPISODES):
            state = env.reset()
            done = False
            steps = 0
            while not done:
                current_frame = env._get_frame()
                current_tensor = agent.preprocess_frame(current_frame)
                cv2.imshow("Breakout Frame", current_frame)
                cv2.waitKey(1)
                action, _, _ = agent.take_action(current_tensor)
                next_frame, reward, done, dead = env.step(action)
                next_tensor = agent.preprocess_frame(next_frame)
                steps += 1
                if done:
                    goal += 1
                    break
                if env.dead:
                    break
            avg_score = avg_score + env.score
            living_history.append(1-env.dead)
            living_rate = np.mean(living_history)
            return_list.append(env.score)
            outStr = "Episode: {}  Steps: {}  Reward: {}  Living_Rate/10: {:.3f}  Done: {}".format(
                            episode, steps, env.score, living_rate, done)
            print(outStr)
            outputPath = SAVE_DIR + LOG_SAVE_NAME + '.txt'
            with open(outputPath, 'a') as outfile:
                outfile.write(outStr+"\n")
        """Testing end"""
        # 繪製學習曲線
        plt.plot(return_list)
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.title("Steps per Episode over Training")
        plt.show()
    """Testing end"""



                    
