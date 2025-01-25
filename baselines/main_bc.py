import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ddpm import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from get_dino_feature import get_dino_feature

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim 
        emb = math.log(self.theta) / (half_dim)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Model(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.channels = 1
        self.self_condition = False
        
        self.down1 = nn.Linear(1, 256)
        self.down2 = nn.Linear(256, 256)
        self.down3 = nn.Linear(256, 256)
        
        self.up1 = nn.Linear(256, 256)
        self.up2 = nn.Linear(256, 256)
        self.up3 = nn.Linear(256, 1)
        
        dim = 1
        fourier_dim = 2 * dim
        time_dim = dim
        sinusoidal_pos_emb_theta = 10000
        sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
        
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.condition_mlp = nn.Sequential(
            nn.Linear(6144, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        self.task_condition_mlp = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
    def forward(self, x, t, c, task_cond):
        # print(self.down1(x).shape, self.time_mlp(t).shape)
        # c_emb = self.condition_mlp(c)
        # task_c_emb = self.task_condition_mlp(task_cond)
        t_emb = self.time_mlp(t)
        x1 = self.down1(x) + t_emb
        x2 = self.down2(x1) + t_emb
        x3 = self.down3(x2) + t_emb
        
        x4 = self.up1(x3) + x2 + t_emb
        x5 = self.up2(x4) + x1 + t_emb
        x6 = self.up3(x5) + t_emb

        # x6 = x6.unsqueeze(1)
        
        return x6
    
class BCDataset(Dataset):
    def __init__(self):
        super().__init__()
        
        push_bar_identity, push_bar_feature = get_dino_feature('push_bar')
        pick_bar_identity, pick_bar_feature = get_dino_feature('pick_bar')
        open_box_identity, open_box_feature = get_dino_feature('open_box')
        turn_faucet_identity, turn_faucet_feature = get_dino_feature('turn_faucet')

        open_box_identity = np.repeat(open_box_identity, 12)
        open_box_feature = np.repeat(open_box_feature, 12, axis=0)
        turn_faucet_identity = np.repeat(turn_faucet_identity, 12)
        turn_faucet_feature = np.repeat(turn_faucet_feature, 12, axis=0)

        push_bar_task_onehot = np.repeat(np.array([[1., 0., 0., 0.,]]), len(push_bar_identity), axis=0)
        pick_bar_task_onehot = np.repeat(np.array([[0., 1., 0., 0.,]]), len(pick_bar_identity), axis=0)
        open_box_task_onehot = np.repeat(np.array([[0., 0., 1., 0.,]]), len(open_box_identity), axis=0)
        turn_faucet_task_onehot = np.repeat(np.array([[0., 0., 0., 1.,]]), len(turn_faucet_identity), axis=0)

        self.identity_tensor = torch.tensor(
            np.concatenate((push_bar_identity, pick_bar_identity, open_box_identity, turn_faucet_identity), axis=0),
            dtype=torch.float
        ).unsqueeze(1)

        self.condition_tensor = torch.tensor(
            np.concatenate((push_bar_feature, pick_bar_feature, open_box_feature, turn_faucet_feature), axis=0),
            dtype=torch.float
        )
        
        self.task_onehot_tensor = torch.tensor(
            np.concatenate([
                push_bar_task_onehot,
                pick_bar_task_onehot,
                open_box_task_onehot,
                turn_faucet_task_onehot
            ], axis=0),
            dtype=torch.float
        )
        
        print(self.task_onehot_tensor.shape)
        
    def __len__(self):
        return len(self.identity_tensor)
    
    def __getitem__(self, index):
        # print(self.identity_tensor[index].shape, self.condition_tensor[index].shape)
        return self.identity_tensor[index], self.condition_tensor[index], self.task_onehot_tensor[index]

model = Model()

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 1,
    timesteps = 100,
    objective = 'pred_v'
).to('cuda')

# Or using trainer

# dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below


dataset = BCDataset()

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 70000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)
