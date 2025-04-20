import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
# 資料集處理類別
class MazeDataset(Dataset):
    def __init__(self, data):
        self.prev_frames = np.array(data["current_frames"], dtype=np.float32)  # (N, 2, H, W)
        self.actions = np.array(data["actions"], dtype=np.int64)  # (N,)
        self.next_frames = np.array(data["next_frames"], dtype=np.float32)  # (N, H, W)
    
    def __len__(self):
        return len(self.prev_frames)
    
    def __getitem__(self, idx):
        prev_frame_1 = np.expand_dims(self.prev_frames[idx][0], axis=0)  # (1, H, W)
        prev_frame_2 = np.expand_dims(self.prev_frames[idx][1], axis=0)  # (1, H, W)
        next_frame = np.expand_dims(self.next_frames[idx], axis=0)  # (1, H, W)
        action = self.actions[idx]

        prev_frame_1 = torch.tensor(prev_frame_1, dtype=torch.float32) / 255.0
        prev_frame_2 = torch.tensor(prev_frame_2, dtype=torch.float32) / 255.0
        next_frame = torch.tensor(next_frame, dtype=torch.float32) / 255.0
        action = torch.tensor(action, dtype=torch.long)
        
        return prev_frame_1, prev_frame_2, action, next_frame


# Autoencoder 類別
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # **Encoder: 接收兩張影像**
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # (2, H, W) → (16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (16, H/2, W/2) → (32, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, H/4, W/4) → (64, H/8, W/8)
            nn.ReLU(),
        )

        # **Action embedding**
        self.action_embed = nn.Embedding(3, 64 * 24 * 24)  # 嵌入維度對應 Encoder 輸出
        
        # **Decoder: 復原影像**
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, H/8, W/8) → (64, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, H/4, W/4) → (32, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, H/2, W/2) → (16, H, W)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # (16, H, W) → (1, H, W)
            nn.Sigmoid(),  # 確保輸出範圍在 [0, 1]
        )

    def forward(self, x, action):
        """接收 (frame_t-2, frame_t-1) 預測 frame_t"""
        #x = torch.cat((frame_t_2, frame_t_1), dim=1)  # 合併為 (2, H, W)
        encoded = self.encoder(x)  # (batch_size, 64, H/8, W/8)

        # **Action 嵌入**
        batch_size = encoded.shape[0]
        action_embedded = self.action_embed(action).view(batch_size, 64, 24, 24)

        # **合併影像與動作**
        combined = torch.cat((encoded, action_embedded), dim=1)  # (batch_size, 128, H/8, W/8)

        # **解碼回 (1, H, W)**
        decoded = self.decoder(combined)

        return decoded
