import torch
import torch.nn as nn
import torch.nn.functional as F

class ARM(nn.Module):
    def __init__(self, input_size, hidden_size, num_env, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.Wf = nn.Linear(input_size, hidden_size)  # (256 -> |a|*16)   Wf
        self.Wh = nn.Linear(hidden_size, hidden_size)  # (|a|*16 -> |a|*16)   Wh
        self.Wx = nn.Linear(hidden_size, input_size)  # (|a|*16 -> 256)   from Wx.+b
        self.Yh = nn.Linear(input_size, hidden_size)  # (256 -> |a|*16)   from tranform to yt

        self.a = nn.Parameter(torch.ones(num_env, 1), requires_grad=True)   # a from tanh(a.)
        self.beta = nn.Parameter(torch.ones(num_env, 1), requires_grad=True)   # beta from alpha(t)
        self.g = nn.Parameter(torch.normal(mean=0, std=1, size=(num_env, input_size)), requires_grad=True)   # g from g & 1-g

        # self.hx = nn.Parameter(torch.FloatTensor(num_env, hidden_size), requires_grad=False)
        # self.hx_prev = nn.Parameter(torch.FloatTensor(num_env, hidden_size), requires_grad=False)
        # self.alpha_prev = nn.Parameter(torch.FloatTensor(num_env, 256), requires_grad=False)
        # self.g = nn.Parameter(torch.FloatTensor(num_env, input_size), requires_grad=True).to(device)

        self.device = device

    def forward(self, state, hidden, alpha_prev):  # use yt
        hx, yx = hidden
        alpha_prev = alpha_prev.to(self.device)
        """Attention Weight Output"""
        s = self.Wf(state)              # lineared state
        h = self.Wh(hx)                 # lineared h(t-1)
        s_h_sum = s + h                 # state + h(t-1)  state 跟 h(t-1) 融合
        # kt = tanh(a * (s + h(t-1)))   a 是可學習參數，決定每個維度信息融合強度 透過tanh生成 [-1, 1] gating kernal = kt
        k_t = torch.tanh(self.a * s_h_sum)      
        s_t = self.Wx(k_t)                      # st = Wx * kt  kt投射回256維度 生成注意力分數 = st
        alpha_now = F.log_softmax(s_t, dim=1)   # alpha(t) = softmax(st)   變為和為1的 Attention Score(t)
        # g 是可學習參數，控制 alpha_prev 和 alpha_now 的融合程度  得出最新的 Attention Score(t) = alpha(t)
        alpha_new = torch.sigmoid(alpha_now * self.g + alpha_prev * (1 - self.g))  
        # beta 是可學習參數，越大輸出值越極端，反之平緩，類似硬性注意力機制
        Yt = self.Yh(F.log_softmax(alpha_new * self.beta, dim=1) * state)  # state * alpha_beta_softmax(t)  乘上正規化注意力分數

        """Recurrent Output"""
        rt = torch.sigmoid(s_h_sum)         # rt = sigmoid(state + hidden state)  重置閘門
        zt = torch.sigmoid(s_h_sum)         # zt = sigmoid(state + hidden state)  更新閘門
        # rt * h 決定多少歷史需保留 再混和當前state tanh()後形生成候選hidden state = h_hat
        hx_hat = torch.tanh((rt * h) + s)   # hx_hat = tanh(rt * h + state)   After tanh()
        # zt 決定用多少新h_hat和多少舊h(t-1) 完成GRU更新
        hx_new = hx_hat * zt + hx * (1 - zt)    # (hx_hat * zt)(New) + (hx(t-1) * (1 - zt))(Old) = ht
        # print(self.a.shape)
        # print('a:', self.a)
        # print("a.mean", self.a.mean())
        # print('beta:', self.beta)
        # print('g:', self.g)
        # print('arm')

        return (hx_new, Yt), alpha_new.detach(), self.a.mean(), self.beta.mean()


if __name__ == '__main__':
    state = torch.FloatTensor(6, 256).cuda()
    hx = torch.FloatTensor(6, 96).cuda()
    cx = torch.FloatTensor(6, 96).cuda()
    hidden = (hx, cx)
    alpha = torch.FloatTensor(6, 256).cuda()
    rnn = ARM(state.shape[1], hx.shape[1], 6).cuda()
    hidden = rnn(state, hidden, alpha)
    # for name, parameters in rnn.named_parameters():
    #     print(name, ':', parameters.size())
    # print(list(rnn.parameters()))