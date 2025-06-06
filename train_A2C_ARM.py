# region Import Libraries
import gc
import cv2
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

GAMMA = 0.99                # discount factor
LR_CRT = 1e-4             # initial learning rate
LR_ACT = 1e-3
ENTROPY_COEF = 0.01
EPS_START = 1.0             # initial epsilon
FRAME_SEQ = 4

HID_SIZE = 256

UPDATE_EVERY = 10
SAVE_EVERY = 1000             # how often to save the network
MAX_TIMESTEPS = 1000
N_EPISODES = int(1e4)
TEST_EPISODES = 10

RNN_MODE = "ARM"  

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

        self.rnn = ARM(input_size = self.hid_size, hidden_size = self.hid_size, num_env=1, device = self.device).to(self.device)
        
        self.actor = Actor(self.hid_size, num_actions).to(self.device)
        self.critic = Critic(self.hid_size).to(self.device)

        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer  = torch.optim.RMSprop(
            list(self.rnn.parameters()) + list(self.actor.parameters()), lr=actor_lr, eps = 1e-5)
        self.critic_optimizer = torch.optim.RMSprop(
            list(self.rnn.parameters()) + list(self.critic.parameters()), lr=critic_lr, eps = 1e-5)

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
            feat = self.cnn(state)              # (4, 1, 1, H, W) -> (1, 64, 24, 24)
            feat = feat.view(feat.size(0),-1)   # (1, 64, 24, 24) -> (1, 36864)
            feat = self.proj(feat)              # (1, 36864) -> (1, 256)
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
        # states = torch.stack(transition_dict['states'],dim=0).to(self.device)
        log_probs  = torch.stack(transition_dict['log_probs'],dim=0).to(self.device) 
        entropies = torch.stack(transition_dict['entropies'],dim=0).to(self.device)  # (T, )
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        yt_tensors = torch.stack(transition_dict['yt_tensors'], dim=0).to(self.device)  # (T, 1, 256)

        """Critic:  state value Q(s) Q(s')"""
        values = self.critic(yt_tensors)
        values = values.view(-1) 
        with torch.no_grad():
            if dones[-1] == 1:
                R = torch.zeros(1, 1).to(self.device)  # (1, 1)

            else:    
                # 取最后一步的价值作为 bootstrap
                R  = values[-1].detach()                 # scalar

        T = rewards.size(0)
        returns = torch.zeros_like(rewards)

        for t in reversed(range(T)):
            R = rewards[t] + self.gamma * R 
            returns[t] = R
        returns = returns.to(self.device)
        
        # print(returns)  # (T, 1)
        advantage = returns - values.detach()  # TD_error = G - V(s)

        if T > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        returns = returns.view(-1) 

        critic_loss = 0.25 * F.mse_loss(values, returns.detach())  # Critic loss: MSE

        # Actor loss: -E[ log π(a|s) * TD_error ]
        # 其中 advantage.detach() 防止梯度回流到 Critic
        # policy_loss = - (log_probs * advantage.detach()).mean()
        policy_loss  = - (log_probs * advantage.detach().view(-1)).mean()
        entropy_loss = - entropies.mean()  # 我们想最大化熵，所以在 loss 里是负项
        actor_loss  = policy_loss + ENTROPY_COEF * entropy_loss

        total_loss = actor_loss + critic_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return actor_loss, critic_loss          # actor loss 非常小
    


env = BreakoutEnv()
steps_per_episode = []
goal = 0
avg_score = 0
successes = 0
total_steps = 0
return_list = []  
act_loss_buffer = []
crt_loss_buffer = []
state_buffer = deque(maxlen=FRAME_SEQ)
living_history = deque(maxlen=10)

# ———— 3. 主流程示例 ————
if __name__ == "__main__":
    if TRAIN:
        agent = ACAgent(input_shape=(1, screen_height, screen_width), 
                        num_actions=3,
                        actor_lr=LR_ACT, 
                        critic_lr=LR_CRT, 
                        gamma=GAMMA)

        for episode in range(N_EPISODES):
            """Initialize environment every episode"""

            torch.cuda.empty_cache()
            gc.collect()
            state = env.reset()
            current_tensor = agent.preprocess_frame(state)      # (1,1,192,192)
            state_buffer = deque([current_tensor]*FRAME_SEQ, maxlen=FRAME_SEQ)
            steps = 0
            done = False

            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'log_probs': [],
                'entropies': [],
                'rewards': [],
                'dones': [],
                'yt_tensors': [],
            }
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
                current_frame = env._get_frame()  # 取得當前畫面
                current_tensor = agent.preprocess_frame(current_frame)      # (1,1,192,192)
                state_buffer.append(current_tensor)

                cv2.imshow("Breakout Frame", current_frame)
                cv2.waitKey(1)

                state_seq = torch.stack(list(state_buffer), dim=0)      # (4, 1, 1, H, W)

                action, logp, ent = agent.take_action(state_seq)
                next_frame, reward, done, dead = env.step(action)
                next_tensor = agent.preprocess_frame(next_frame)

                transition_dict['states'].append(state_seq)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_tensor)
                transition_dict['log_probs'].append(logp)
                transition_dict['entropies'].append(ent)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['yt_tensors'].append(agent.hidden[1])

                steps += 1

                # if (step+1) % UPDATE_EVERY == 0 or done:
                if done or dead:
                    """Learn every UPDATE_EVERY steps"""
                    loss_act, loss_crt = agent.learn(transition_dict)
                    act_loss_buffer.append(loss_act.item())
                    crt_loss_buffer.append(loss_crt.item())
                    transition_dict.clear()
                    transition_dict = {
                        'states': [],
                        'actions': [],
                        'next_states': [],
                        'log_probs': [],
                        'entropies': [],
                        'rewards': [],
                        'dones': [],
                        'yt_tensors': [],
                    }

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

            if (episode+1) % 5 == 0:
                avg_score /= 5
                steps_per_episode.append(avg_score)
                avg_score = 0

            if (episode+1) % 1000 == 0 :
                plt.plot(steps_per_episode)
                plt.xlabel("Episode")
                plt.ylabel("Total reward")
                plt.title("Steps per Episode over Training")
                plt.show()

            # save the network every episode
            if (episode+1) % SAVE_EVERY == 0 and SAVE_MODEL:
                torch.save(agent.actor.state_dict(), SAVE_DIR + f'actor_model_episode{episode+1}.pth')
                torch.save(agent.critic.state_dict(), SAVE_DIR + f'critic_model_episode{episode+1}.pth')

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



                    
