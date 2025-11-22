import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Encoder

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
        
        self.coord_proj = nn.Linear(3, self.dim)
        
        self.spatial_conv = nn.Conv3d(
            in_channels=self.dim, 
            out_channels=self.dim,
            kernel_size=3,
            padding=1
        )
        self.spatial_norm = nn.LayerNorm(self.dim)
        
        self.status_emb = nn.Linear(26 + self.max_actions * 15, self.dim)
        
        self.transformer = Encoder(
            dim=self.dim,
            depth=config['depth'],
            heads=config['heads'],
            rotary_pos_emb=True,
            ff_glu=True
        )
        
        self.predict_next_state = nn.Linear(self.dim, self.num_actions * self.dim)
        self.predict_reward = nn.Linear(self.dim, self.num_actions)
        
        self.imagination_proj = nn.Linear(self.dim * self.num_actions, self.dim)
        
        self.action_head = nn.Linear(self.dim * 2, self.num_actions)
        self.craft_head = nn.Linear(self.dim, 4)
        self.value_head = nn.Linear(self.dim, 1)
        
    def forward(self, voxels, status, inventory, action_history, hx=None):
        B = voxels.shape[0]
        device = voxels.device

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
        x_equip = (self.equip_emb(hand_ids) + self.equip_emb(helm_ids) + 
                   self.equip_emb(chest_ids) + self.equip_emb(leg_ids) + 
                   self.equip_emb(boot_ids))
        x_name = self.name_emb(name_ids)
        
        x_vox = x_block + x_light + x_equip + x_name
        x_vox = x_vox.permute(0, 2, 1)
        spatial_size = int(round(x_vox.shape[-1] ** (1/3)))
        x_vox = x_vox.reshape(B, self.dim, spatial_size, spatial_size, spatial_size)
        
        x_vox = self.spatial_conv(x_vox)
        x_vox = x_vox.mean(dim=[2, 3, 4])
        x_vox = self.spatial_norm(x_vox)
        x_vox = x_vox.unsqueeze(1)
        
        raw_status = status.squeeze(1)
        stats_and_home = raw_status[:, :13]
        coords = raw_status[:, 13:16]
        events_radar_fx = raw_status[:, 16:]
        
        x_coords = self.coord_proj(coords).unsqueeze(1)
        
        act_flat = action_history.reshape(B, -1)
        status_input = torch.cat([stats_and_home, events_radar_fx, act_flat], dim=-1)
        x_stat = self.status_emb(status_input).unsqueeze(1)
        
        inventory_ids = inventory.squeeze(1)
        x_inv = self.inventory_emb(inventory_ids)
        x_inv = x_inv.mean(dim=1).unsqueeze(1)
        
        if hx is None:
            hx = torch.zeros(B, self.dim, device=device)
        
        if hx.dim() == 2:
            hx_token = hx.unsqueeze(1)
        else:
            hx_token = hx
            
        sequence = torch.cat([hx_token, x_vox, x_stat, x_inv, x_coords], dim=1)
        
        x_out = self.transformer(sequence)
        
        next_hx = x_out[:, 0, :] 
        
        pred_next_states = self.predict_next_state(next_hx).reshape(B, self.num_actions, self.dim)
        pred_rewards = self.predict_reward(next_hx).unsqueeze(-1)
        
        imagined_context = pred_next_states.reshape(B, -1)
        imagined_emb = self.imagination_proj(imagined_context)
        
        policy_input = torch.cat([next_hx, imagined_emb], dim=-1)
        
        action_logits = self.action_head(policy_input)
        craft_logits = self.craft_head(next_hx)
        value = self.value_head(next_hx).squeeze(-1)
        
        return action_logits, craft_logits, value, next_hx, pred_next_states, pred_rewards