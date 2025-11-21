import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from x_transformers import Encoder
from titans_pytorch import NeuralMemory

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

class DynamicsModel(nn.Module):
    def __init__(self, dim, num_actions):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        self.head_next_state = nn.Linear(dim, dim)
        self.head_reward = nn.Linear(dim, 1)

    def forward(self, state, action_indices):
        act_emb = self.action_emb(action_indices)
        x = torch.cat([state, act_emb], dim=-1)
        feat = self.net(x)
        next_state = self.head_next_state(feat)
        reward = self.head_reward(feat)
        return next_state, reward

class MinecraftAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config['dim']
        self.max_actions = config['max_action_history']
        self.num_actions = 15
        
        self.block_emb = nn.Embedding(config['num_block_types'], self.dim)
        self.light_emb = nn.Embedding(17, self.dim)
        self.equip_emb = nn.Embedding(config['num_block_types'], self.dim)
        self.name_emb = nn.Embedding(config['num_block_types'], self.dim)
        self.inventory_emb = nn.Embedding(config['num_block_types'], self.dim)
        
        self.coord_emb = SinusoidalPosEmb(self.dim)
        
        self.spatial_conv = nn.Conv3d(
            in_channels=self.dim, 
            out_channels=self.dim,
            kernel_size=3,
            padding=1
        )
        self.spatial_norm = nn.LayerNorm(self.dim)
        
        self.status_emb = nn.Linear(12 + self.max_actions * 15, self.dim)
        
        self.transformer = Encoder(
            dim=self.dim,
            depth=config['depth'],
            heads=config['heads'],
            rotary_pos_emb=True,
            ff_glu=True
        )
        
        self.mem = NeuralMemory(dim=self.dim, chunk_size=64)
        self.memory = nn.GRUCell(self.dim, self.dim)
        
        self.dynamics = DynamicsModel(self.dim, self.num_actions)
        self.imagination_proj = nn.Linear(self.dim * self.num_actions, self.dim)
        
        self.action_head = nn.Linear(self.dim * 2, self.num_actions)
        self.craft_head = nn.Linear(self.dim, 4)
        self.value_head = nn.Linear(self.dim, 1)
        
    def forward(self, voxels, status, inventory, action_history, hx=None):
        B = voxels.shape[0]
        voxel_ids = voxels[:, 0, :]
        light_levels = voxels[:, 1, :]
        hand_ids = voxels[:, 2, :]
        helm_ids = voxels[:, 3, :]
        chest_ids = voxels[:, 4, :]
        leg_ids = voxels[:, 5, :]
        boot_ids = voxels[:, 6, :]
        name_ids = voxels[:, 7, :]
        
        x_block = self.block_emb(voxel_ids)
        x_light = self.light_emb(light_levels)
        x_hand = self.equip_emb(hand_ids)
        x_helm = self.equip_emb(helm_ids)
        x_chest = self.equip_emb(chest_ids)
        x_leg = self.equip_emb(leg_ids)
        x_boot = self.equip_emb(boot_ids)
        x_name = self.name_emb(name_ids)
        
        x_vox = x_block + x_light + x_hand + x_helm + x_chest + x_leg + x_boot + x_name
        
        x_vox = x_vox.permute(0, 2, 1)
        spatial_size = int(round(x_vox.shape[-1] ** (1/3)))
        x_vox = x_vox.reshape(B, self.dim, spatial_size, spatial_size, spatial_size)
        
        x_vox = self.spatial_conv(x_vox)
        x_vox = x_vox.mean(dim=[2, 3, 4])
        x_vox = self.spatial_norm(x_vox)
        x_vox = x_vox.unsqueeze(1)
        
        raw_status = status.squeeze(1)
        core_status = raw_status[:, :12]
        coords = raw_status[:, 12:]
        
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
        x, _ = self.mem(x)
        
        if hx is None:
            hx = torch.zeros(B, self.dim, device=x.device)
            
        hx = self.memory(x_out, hx)
        
        all_actions = torch.arange(self.num_actions, device=x.device).expand(B, self.num_actions)
        hx_expanded = hx.unsqueeze(1).expand(B, self.num_actions, self.dim)
        
        pred_next_states, pred_rewards = self.dynamics(hx_expanded, all_actions)
        
        imagined_context = pred_next_states.reshape(B, -1)
        imagined_emb = self.imagination_proj(imagined_context)
        
        policy_input = torch.cat([hx, imagined_emb], dim=-1)
        
        action_logits = self.action_head(policy_input)
        craft_logits = self.craft_head(hx)
        value = self.value_head(hx).squeeze(-1)
        
        return action_logits, craft_logits, value, hx, pred_next_states, pred_rewards