import gc
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

from variables import *
from collections import deque
from breaker import BreakoutEnv
from A2C import *
from ARM import *

torch.cuda.empty_cache()
gc.collect()

#region Hyperparameters
TRAIN = True               # train or test
LOAD_MODEL = False
SAVE_MODEL = True

GAMMA = 0.985                # discount factor
LR_CRT = 1e-4             # initial learning rate
LR_ACT = 1e-3
ENTROPY_COEF = 0.05
EPS_START = 1.0             # initial epsilon
FRAME_SEQ = 2

UPDATE_EVERY = 5
SAVE_EVERY = 50             # how often to save the network
MAX_TIMESTEPS = 1000
N_EPISODES = int(1e4)
# SAVE_DIR = "./!BB/train2/"
SAVE_DIR = "/content/drive/MyDrive/CCU/BB/Out_A2C/"
MODEL_SELECT = "model_episode1550.pth"
LOG_SAVE_NAME = "out_A2C"

#endregion


# ———— 2. Agent 和训练循环 ————
class ACAgent:
    def __init__(self, input_shape, num_actions, actor_lr, critic_lr, gamma):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma

        self.actor = Actor(input_shape, num_actions).to(self.device)
        self.critic = Critic(input_shape).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def preprocess_frame(self, frame):
        """State frame preprocess"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)  # (H, W) → (1, H, W)
        frame = np.expand_dims(frame, axis=0)  # (H, W) → (1, 1, H, W)
        return torch.tensor(frame, device="cpu")
    
    def take_action(self, state):
        state = state.to(self.device) 
        probs = self.actor(state)
        # 创建以probs为标准类型的数据分布
        dist = torch.distributions.Categorical(probs)
        # 随机选择一个动作 tensor-->int
        action = dist.sample()                            # Tensor([a])
        log_prob = dist.log_prob(action)                  # Tensor([log π(a|s)])
        entropy  = dist.entropy()                         # Tensor([H(π(·|s))])
        return action.item(), log_prob, entropy

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

        # Critic:  state value Q(s) Q(s')
        values = self.critic(states)        # (B,1)
        next_values = self.critic(next_states)   # (B,1)

        # TD target (Q)：r + γ * V(s') * (1 - done)
        td_targets = rewards + self.gamma * next_values * (1 - dones)

        # TD error (Advantage) = target - V(s)
        advantage = td_targets - values
        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Critic loss: MSE
        critic_loss = F.mse_loss(values, td_targets.detach())

        # Actor loss: -E[ log π(a|s) * TD_error ]
        # 其中 advantage.detach() 防止梯度回流到 Critic
        policy_loss = - (log_probs * advantage.detach()).mean()
        entropy_loss = - entropies.mean()  # 我们想最大化熵，所以在 loss 里是负项
        actor_loss  = policy_loss + ENTROPY_COEF * entropy_loss

        # print("policy loss: {:.3f}".format(policy_loss))
        # print("entropy loss:{:.3f}".format(entropy_loss))
        # print("actor loss:{:.3f}".format(actor_loss))
        # print("critic loss:{:.3f}".format(critic_loss))


        # 同步更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss, critic_loss          # actor loss 非常小



env = BreakoutEnv()
state = env.reset()
steps_per_episode = []
done = False
goal = 0
steps = 0
avg_score = 0
successes = 0
total_steps = 0
return_list = []  
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
            torch.cuda.empty_cache()
            gc.collect()
            state = env.reset()
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
            }

            for step in range(MAX_TIMESTEPS):
            # print(torch.cuda.memory_allocated())
                current_frame = env._get_frame()  # 取得當前畫面
                current_tensor = agent.preprocess_frame(current_frame)      # (1,1,192,192)

                # cv2.imshow("Breakout Frame", current_frame)
                # cv2.waitKey(1)

                action, logp, ent = agent.take_action(current_tensor)
                next_frame, reward, done, dead = env.step(action)
                next_tensor = agent.preprocess_frame(next_frame)

                transition_dict['states'].append(current_tensor)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_tensor)
                transition_dict['log_probs'].append(logp)
                transition_dict['entropies'].append(ent)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                steps += 1

                if step % UPDATE_EVERY == 0:
                    loss_act, loss_crt = agent.learn(transition_dict)
                    transition_dict = {
                        'states': [],
                        'actions': [],
                        'next_states': [],
                        'log_probs': [],
                        'entropies': [],
                        'rewards': [],
                        'dones': [],
                    }

                if done:
                    goal += 1
                    break
                if env.dead:
                    break

            loss_act, loss_crt = agent.learn(transition_dict)

            avg_score = avg_score + env.score
            living_history.append(1-env.dead)
            living_rate = np.mean(living_history)
            # success_rate = sum(success_history) / len(success_history)  # 計算成功率

            return_list.append(env.score)

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

            outStr = "Episode: {}  Steps: {}  Reward: {}  Living_Rate/10: {:.3f}  Done: {} | Critic_loss: {:.3f}  Actor_loss: {:.3f}".format(
                            episode, steps, env.score, living_rate, done, loss_crt, loss_act)
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


                    
