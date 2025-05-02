import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN_DQN, self).__init__()
        
        # 輸入形狀應該是 (4, H, W) - 從 Maze.render() 得到的灰階圖像
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        hidden_size = 256
        
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
        
        self.fc_embed = nn.Linear(36864, hidden_size)  # 添加映射層
        self.fc = nn.Linear(hidden_size, num_actions)

        # Self-Attention 
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)

        # Dueling DQN 
        self.Alinear1 = nn.Linear(hidden_size, 256)
        self.Alinear2 = nn.Linear(256, num_actions)

        self.Vlinear1 = nn.Linear(hidden_size, 256)
        self.Vlinear2 = nn.Linear(256, 1)  

        self.relu = nn.LeakyReLU()
    
    def _get_conv_output_size(self):
        # 創建一個模擬輸入來獲取卷積層的輸出尺寸
        dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
        conv_output = self.conv_layers(dummy_input)
        return int(torch.prod(torch.tensor(conv_output.shape[1:])))  # 乘積所有維度（除了批次維度）
    
    def forward(self, x):
        """
        修改後的輸入 x 的形狀為 (batch, seq_len, 1, H, W)
        輸出： (batch, hidden_size) 或 (batch, num_actions) 根據你的需求
        """
        batch, seq_len, C, H, W = x.size()
        # 合併 batch 和 seq_len 維度
        x = x.view(batch*seq_len, C, H, W)  # 移除第 3 維 (C=1)
        x = self.conv_layers(x)
        
        # 展平
        x = x.view(x.size(0), -1)       # (batch * seq_len, flatten_size)
        x = self.fc_embed(x)            # (batch * seq_len, hidden_size=256)
        x = x.view(batch, seq_len, -1)  # (batch, seq_len, hidden_size=256)

        # self attention
        attn_output, _ = self.attention(x, x, x)  # (batch, seq_len, hidden_size)
        
        # x = torch.max(attn_output, dim=1).values  # max pooling over time
        x = torch.mean(attn_output, dim=1)  # 平均池化
        # x = attn_output[:, -1, :]  # 只取最後時間步
        # x = attn_output

        q = self.fc(x)             # 這裡 self.fc 是將 hidden_size 映射到 num_actions 的線性層

        return q

        # # Dueling DQN 部分
        # Ax = self.relu(self.Alinear1(x))
        # # Ax = self.dropout(Ax)  # 使用 Dropout
        # Ax = self.Alinear2(Ax)  # 最後一層沒有激活函數

        # Vx = self.relu(self.Vlinear1(x))
        # # Vx = self.dropout(Vx)  # 使用 Dropout
        # Vx = self.Vlinear2(Vx)  # 最後一層沒有激活函數

        # # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # q = Vx + (Ax - Ax.mean(dim=1, keepdim=True))
        # return q
