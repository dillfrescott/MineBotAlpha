import torch
import torch.nn as nn
from x_transformers import Encoder

class MinecraftAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config['dim']
        self.max_actions = config['max_action_history']
        
        self.block_emb = nn.Embedding(config['num_block_types'], self.dim)
        self.light_emb = nn.Embedding(17, self.dim)
        self.inventory_emb = nn.Embedding(config['num_block_types'], self.dim)
        
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
        
        self.action_head = nn.Linear(self.dim, 12)
        self.craft_head = nn.Linear(self.dim, 4)
        self.value_head = nn.Linear(self.dim, 1)
        
    def forward(self, voxels, status, inventory, action_history):
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
        
        act_flat = action_history.reshape(B, -1)
        status_input = torch.cat([status.squeeze(1), act_flat], dim=-1)
        x_stat = self.status_emb(status_input).unsqueeze(1)
        
        inventory_ids = inventory.squeeze(1)
        x_inv = self.inventory_emb(inventory_ids)
        x_inv = x_inv.mean(dim=1).unsqueeze(1)
        
        x = x_vox + x_stat + x_inv
        x = self.transformer(x)
        
        action_logits = self.action_head(x[:, 0])
        craft_logits = self.craft_head(x[:, 0])
        value = self.value_head(x[:, 0]).squeeze(-1)
        
        return action_logits, craft_logits, value