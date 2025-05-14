
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).major,
      torch.cuda.get_device_properties(0).minor)



returns = []
rewards = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # bootstrap
G = 50
for reward in reversed(rewards):  # 反向遍历
    G = reward + 0.99 * G
    returns.insert(0, G)  # 前插构造 [G_t, G_{t+1}, ..., G_{t+n-1}]

print(returns)