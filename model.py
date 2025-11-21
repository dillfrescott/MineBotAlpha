import torch
import torch.nn as nn
import math
from x_transformers import Encoder

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        ex = x[:, 0].unsqueeze(1) * emb.unsqueeze(0)
        ey = x[:, 1].unsqueeze(1) * emb.unsqueeze(0)
        ez = x[:, 2].unsqueeze(1) * emb.unsqueeze(0)
        
        emb_x = torch.cat((ex.sin(), ex.cos()), dim=-1)
        emb_y = torch.cat((ey.sin(), ey.cos()), dim=-1)
        emb_z = torch.cat((ez.sin(), ez.cos()), dim=-1)
        
        return emb_x + emb_y + emb_z

class MinecraftAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config['dim']
        self.max_actions = config['max_action_history']
        
        self.block_emb = nn.Embedding(config['num_block_types'], self.dim)
        self.light_emb = nn.Embedding(17, self.dim)
        self.inventory_emb = nn.Embedding(config['num_block_types'], self.dim)
        
        self.coord_emb = SinusoidalPosEmb(self.dim)
        
        self.spatial_conv = nn.Conv3d(
            in_channels=self.dim, 
            out_channels=self.dim,
            kernel_size=3,
            padding=1
        )
        self.spatial_norm = nn.LayerNorm(self.dim)
        
        self.status_emb = nn.Linear(9 + self.max_actions * 12, self.dim)
        
        self.transformer = Encoder(
            dim=self.dim,
            depth=config['depth'],
            heads=config['heads'],
            rotary_pos_emb=True,
            ff_glu=True
        )
        
        self.memory = nn.GRUCell(self.dim, self.dim)
        
        self.action_head = nn.Linear(self.dim, 12)
        self.craft_head = nn.Linear(self.dim, 4)
        self.value_head = nn.Linear(self.dim, 1)
        
    def forward(self, voxels, status, inventory, action_history, hx=None):
        B = voxels.shape[0]
        voxel_ids = voxels[:, 0, :]
        light_levels = voxels[:, 1, :]
        
        x_block = self.block_emb(voxel_ids)
        x_light = self.light_emb(light_levels)
        x_vox = x_block + x_light
        
        x_vox = x_vox.permute(0, 2, 1)
        spatial_size = int(round(x_vox.shape[-1] ** (1/3)))
        x_vox = x_vox.reshape(B, self.dim, spatial_size, spatial_size, spatial_size)
        
        x_vox = self.spatial_conv(x_vox)
        x_vox = x_vox.mean(dim=[2, 3, 4])
        x_vox = self.spatial_norm(x_vox)
        x_vox = x_vox.unsqueeze(1)
        
        raw_status = status.squeeze(1)
        core_status = raw_status[:, :9]
        coords = raw_status[:, 9:]
        
        x_coords = self.coord_emb(coords).unsqueeze(1)
        
        act_flat = action_history.reshape(B, -1)
        status_input = torch.cat([core_status, act_flat], dim=-1)
        x_stat = self.status_emb(status_input).unsqueeze(1)
        
        inventory_ids = inventory.squeeze(1)
        x_inv = self.inventory_emb(inventory_ids)
        x_inv = x_inv.mean(dim=1).unsqueeze(1)
        
        x = torch.cat([x_vox, x_stat, x_inv, x_coords], dim=1)
        x = self.transformer(x)
        
        x_out = x.mean(dim=1)
        
        if hx is None:
            hx = torch.zeros(B, self.dim, device=x.device)
            
        hx = self.memory(x_out, hx)
        
        action_logits = self.action_head(hx)
        craft_logits = self.craft_head(hx)
        value = self.value_head(hx).squeeze(-1)
        
        return action_logits, craft_logits, value, hx