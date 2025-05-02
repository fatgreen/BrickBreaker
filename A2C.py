import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ARM import *

class CNN_LAYER(nn.Module):
    def __init__(self, input_shape):
        super(CNN_LAYER, self).__init__()
        # Convolution
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
        )
    def forward(self, x):
        """
        Input : x = (1, 1, H, W) or (len(episode), 1, 1,  H, W)
        Output： feature map
        """
        if len(x.shape) == 4:
            feat = self.conv_layers(x)      # (B, feat_dim)
        elif len(x.shape) == 5:
            B, T, C, H, W = x.size()
            x = x.view(B*T, C, H, W)        # 移除第 2 維 
            feat = self.conv_layers(x)
        else:
            print("state.shape error:", x.shape)
        return feat


# Actor 网络
class Actor(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Actor, self).__init__()
        # self.cnn = CNN_LAYER(input_shape)
        self.fc_layer = nn.Sequential(
            # nn.Linear(36864, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):

        # x = self.cnn(state)
        # x = x.view(x.size(0),-1)        # (1,36864)
        pi = self.fc_layer(state)

        return pi


# Critic 网络
class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        # self.cnn = CNN_LAYER(input_shape)
        self.fc_layer = nn.Sequential(
            # nn.Linear(36864, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        # x = self.cnn(state)
        # x = x.view(x.size(0),-1)
        
        q = self.fc_layer(state)

        return q
