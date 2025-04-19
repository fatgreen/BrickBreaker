import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN_DQN, self).__init__()
        
        # 輸入形狀應該是 (1, H, W) - 從 Maze.render() 得到的灰階圖像
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        hidden_size = 128
        
        # Convolution
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  
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
        # self.fc_input_dim = self._get_conv_output_size(input_shape)

        self.fc_embed = nn.Linear(36864, hidden_size)  # 添加映射層

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
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # 添加批次維度
            print("sqeez")
            
        # 卷積處理
        x = self.conv_layers(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        x = self.fc_embed(x)  # (batch, hidden_size)

        # self attention
        seq_len = 4  # seq_len > 1, to make self attention has multi sequences
        x = x.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, fc_input_dim)

        # 通過 Self-Attention 層
        attn_output, _ = self.attention(x, x, x)  # (batch, seq_len, hidden_size)
        
        # 取最後一個時間步的輸出
        x = attn_output[:, -1, :]  # 只取最後時間步
        
        return x
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