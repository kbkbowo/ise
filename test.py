import torch

x = torch.tensor([0.7066, 0.0267, 0.0330, 0.7064])
x = x.unsqueeze(0)

# x_normalized = torch.nn.functional.normalize(x, p=2, dim=1)

y = torch.softmax(x * 3, dim=1)
print(y)

z = torch.distributions.Categorical(y).sample()
print(z)
