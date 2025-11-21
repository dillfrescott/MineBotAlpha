import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
from prodigyopt import Prodigy
import os
import time
from model import MinecraftAgent
from bot_env import MinecraftEnv
import asyncio
import sys
import numpy as np

class SPOAgent:
    def __init__(self, model, lr=1.0, value_coef=0.5, entropy_coef=0.03, dynamics_coef=0.5, d_max=0.01):
        self.model = model
        self.optimizer = Prodigy(model.parameters(), lr=lr)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.dynamics_coef = dynamics_coef
        self.d_max = d_max
    
    def act(self, voxels, status, inventory, action_history, hx):
        action_logits, craft_logits, value, next_hx, pred_next_states, _ = self.model(voxels, status, inventory, action_history, hx)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        craft_dist = torch.distributions.Categorical(logits=craft_logits)
        
        action = action_dist.sample()
        craft_action = craft_dist.sample()
        
        action_logp = action_dist.log_prob(action)
        craft_logp = craft_dist.log_prob(craft_action)
        
        entropy = action_dist.entropy().mean() + craft_dist.entropy().mean()
        
        return (action, craft_action), (action_logp, craft_logp), value, entropy, next_hx, pred_next_states
    
    def update(self, batch):
        vox, stat, inv, act_hist, hx, old_logps, actions, returns, next_hx_target = batch
        action_old, craft_old = actions
        action_logp_old, craft_logp_old = old_logps
        
        if action_old.dim() > 1:
            action_old = action_old.squeeze(-1)
        
        action_logits, craft_logits, values, current_hx, pred_next_states, pred_rewards = self.model(vox, stat, inv, act_hist, hx)
        
        device = action_logits.device
        action_old = action_old.to(device)
        craft_old = craft_old.to(device)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        craft_dist = torch.distributions.Categorical(logits=craft_logits)
        
        action_logp = action_dist.log_prob(action_old)
        craft_logp = craft_dist.log_prob(craft_old)
        
        entropy = action_dist.entropy().mean() + craft_dist.entropy().mean()
        
        ratio_action = torch.exp(action_logp - action_logp_old)
        ratio_craft = torch.exp(craft_logp - craft_logp_old)
        
        kl_action = action_logp_old - action_logp
        kl_craft = craft_logp_old - craft_logp
        
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        surr_action = ratio_action * advantages
        surr_craft = ratio_craft * advantages
        
        mask_action = (kl_action <= self.d_max).float().detach()
        mask_craft = (kl_craft <= self.d_max).float().detach()
        
        policy_loss_action = -(surr_action * mask_action).mean()
        policy_loss_craft = -(surr_craft * mask_craft).mean()
        
        policy_loss = policy_loss_action + policy_loss_craft
        
        value_loss = F.mse_loss(values, returns)
        
        batch_indices = torch.arange(pred_next_states.size(0), device=device)
        selected_next_states = pred_next_states[batch_indices, action_old]
        selected_rewards = pred_rewards[batch_indices, action_old].squeeze(-1)
        
        dynamics_loss = F.mse_loss(selected_next_states, next_hx_target.detach()) + \
                        F.mse_loss(selected_rewards, returns) 
        
        total_loss = policy_loss + \
                     self.value_coef * value_loss - \
                     self.entropy_coef * entropy + \
                     self.dynamics_coef * dynamics_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item(), entropy.item(), dynamics_loss.item()

async def train():
    cfg = json.load(open('config.json'))
    train_cfg = cfg['training']
    mc_cfg = cfg['minecraft']
    
    sys.setrecursionlimit(2000)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} using SPO + Intrinsic Curiosity")
    
    os.makedirs(train_cfg['checkpoint_dir'], exist_ok=True)

    def create_env():
        return MinecraftEnv(
            username=mc_cfg['username_train'],
            host=mc_cfg['host'],
            port=mc_cfg['port'],
            auth=mc_cfg['auth'],
            version=mc_cfg['version'],
            view_range=mc_cfg['view_range'],
            home_coords=mc_cfg.get('home_coords', [0, 64, 0]),
            enable_home=mc_cfg.get('enable_home_base', False)
        )
    
    env = create_env()
    
    model = MinecraftAgent(cfg['model']).to(device)
    
    agent = SPOAgent(model, lr=train_cfg['learning_rate'])
    
    checkpoint_path = os.path.join(train_cfg['checkpoint_dir'], "minecraft_agent.pth")
    start_step = 0
    hx = None
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint.get('global_step', 0) + 1
            
            if 'hx' in checkpoint:
                hx = checkpoint['hx'].to(device)
                
            print("Checkpoint loaded")
        except Exception as e:
            print(f"Load failed ({e}), starting fresh")

    save_interval = train_cfg.get('save_interval', 1000)
    update_batch_size = train_cfg.get('update_batch_size', 100)
    curiosity_weight = 0.02

    pbar = tqdm(initial=start_step, position=0, leave=True)
    
    global_step = start_step
    
    if not env.is_ready:
         while not env.is_ready:
             await asyncio.sleep(1)

    obs_vox, obs_stat, obs_inv, obs_hist = env.get_observation()
    
    if hx is None:
        hx = torch.zeros(1, model.dim, device=device)

    while True:
        pbar.set_description(f"Step {global_step}")

        buffer = {
            'vox': [], 'stat': [], 'inv': [], 'act_hist': [], 'hx': [],
            'actions': [], 'logps': [], 'rewards': [], 'values': [], 'next_hx': []
        }
        
        total_intrinsic = 0
        total_extrinsic = 0
        
        try:
            for _ in range(update_batch_size):
                if not env.is_ready:
                    print("Environment not ready, waiting...")
                    await asyncio.sleep(5)
                    obs_vox, obs_stat, obs_inv, obs_hist = env.get_observation()
                    continue
                
                with torch.no_grad():
                    actions, logps, value, entropy, next_hx_pred, pred_next_states = agent.act(
                        obs_vox.to(device),
                        obs_stat.to(device),
                        obs_inv.to(device),
                        obs_hist.to(device),
                        hx
                    )
                
                action, craft_action = actions
                action_logp, craft_logp = logps
                
                extrinsic_reward = await env.step_atomic(action, craft_action)
                
                next_vox, next_stat, next_inv, next_hist = env.get_observation()
                
                with torch.no_grad():
                    _, _, _, _, real_next_hx, _ = agent.act(
                        next_vox.to(device),
                        next_stat.to(device),
                        next_inv.to(device),
                        next_hist.to(device),
                        hx
                    )

                predicted_vector = pred_next_states[0, action.item()]
                surprise = F.mse_loss(predicted_vector, real_next_hx[0])
                intrinsic_reward = surprise.item() * curiosity_weight
                
                total_reward = extrinsic_reward + intrinsic_reward
                
                total_extrinsic += extrinsic_reward
                total_intrinsic += intrinsic_reward
                
                buffer['vox'].append(obs_vox)
                buffer['stat'].append(obs_stat)
                buffer['inv'].append(obs_inv)
                buffer['act_hist'].append(obs_hist)
                buffer['hx'].append(hx)
                buffer['actions'].append((action.cpu(), craft_action.cpu()))
                buffer['logps'].append((action_logp.cpu(), craft_logp.cpu()))
                buffer['rewards'].append(total_reward)
                buffer['values'].append(value.cpu())
                buffer['next_hx'].append(real_next_hx.detach().cpu()) 
                
                hx = real_next_hx.detach()
                
                obs_vox, obs_stat, obs_inv, obs_hist = next_vox, next_stat, next_inv, next_hist

                global_step += 1
                pbar.update(1)
                await asyncio.sleep(0.05)
            
            if len(buffer['rewards']) == 0:
                continue

            returns = []
            R = 0
            for r in reversed(buffer['rewards']):
                R = r + train_cfg['gamma'] * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            
            next_hx_target = torch.cat(buffer['next_hx']).to(device)
            if next_hx_target.dim() > 2: 
                 next_hx_target = next_hx_target.squeeze(1)

            batch = (
                torch.cat(buffer['vox']).to(device),
                torch.cat(buffer['stat']).to(device),
                torch.cat(buffer['inv']).to(device),
                torch.cat(buffer['act_hist']).to(device),
                torch.cat(buffer['hx']).to(device),
                (torch.stack([lp[0] for lp in buffer['logps']]).to(device),
                 torch.stack([lp[1] for lp in buffer['logps']]).to(device)),
                (torch.stack([a[0] for a in buffer['actions']]).to(device),
                 torch.stack([a[1] for a in buffer['actions']]).to(device)),
                returns,
                next_hx_target
            )
            
            loss, p_loss, v_loss, ent, dyn_loss = agent.update(batch)
            pbar.set_postfix({
                "Ext": f"{total_extrinsic:.1f}",
                "Int": f"{total_intrinsic:.2f}",
                "Loss": f"{loss:.2f}",
            })
            
            if global_step % save_interval < update_batch_size:
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'hx': hx
                }, checkpoint_path)

        except (RuntimeError, Exception) as e:
            if "BrokenBarrier" in str(e) or "daemon" in str(e):
                print(f"Bridge crashed ({e}). Restarting environment...")
                try:
                    env.bot.quit()
                except: pass
                env = create_env()
                await asyncio.sleep(5)
                obs_vox, obs_stat, obs_inv, obs_hist = env.get_observation()
            else:
                print(f"Training error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    env.close()

if __name__ == "__main__":
    asyncio.run(train())