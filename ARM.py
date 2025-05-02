import torch
import torch.nn as nn
import torch.nn.functional as F

class ARM(nn.Module):
    def __init__(self, input_size, hidden_size, num_env, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.W = nn.Linear(input_size, hidden_size)  # (256 -> |a|*16)   Wf
        self.U = nn.Linear(hidden_size, hidden_size)  # (|a|*16 -> |a|*16)   Wh
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
        s = self.W(state)
        h = self.U(hx)
        s_h_sum = s + h
        k_t = torch.tanh(self.a * s_h_sum)
        s_t = self.Wx(k_t)
        alpha_now = F.log_softmax(s_t, dim=1)
        alpha_new = torch.sigmoid(alpha_now * self.g + alpha_prev * (1 - self.g))
        Yt = self.Yh(F.log_softmax(alpha_new * self.beta, dim=1) * state)  # use Yt, not hx
        r = torch.sigmoid(s_h_sum)
        z = torch.sigmoid(s_h_sum)
        hx_hat = torch.tanh((r * h) + s)
        hx_new = hx_hat * z + hx * (1 - z)
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