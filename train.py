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

class A2CAgent:
    def __init__(self, model, lr=1.0, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = Prodigy(model.parameters(), lr=lr)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def act(self, voxels, status, inventory, action_history, hx):
        action_logits, craft_logits, value, next_hx = self.model(voxels, status, inventory, action_history, hx)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        craft_dist = torch.distributions.Categorical(logits=craft_logits)
        
        action = action_dist.sample()
        craft_action = craft_dist.sample()
        
        action_logp = action_dist.log_prob(action)
        craft_logp = craft_dist.log_prob(craft_action)
        
        entropy = action_dist.entropy().mean() + craft_dist.entropy().mean()
        
        return (action, craft_action), (action_logp, craft_logp), value, entropy, next_hx
    
    def update(self, batch):
        vox, stat, inv, act_hist, hx, old_logps, actions, returns = batch
        action_old, craft_old = actions
        action_logp_old, craft_logp_old = old_logps
        
        action_logits, craft_logits, values, _ = self.model(vox, stat, inv, act_hist, hx)
        
        device = action_logits.device
        action_old = action_old.to(device)
        craft_old = craft_old.to(device)
        
        action_dist = torch.distributions.Categorical(logits=action_logits)
        craft_dist = torch.distributions.Categorical(logits=craft_logits)
        
        action_logp = action_dist.log_prob(action_old)
        craft_logp = craft_dist.log_prob(craft_old)
        
        entropy = action_dist.entropy().mean() + craft_dist.entropy().mean()
        
        advantages = returns - values
        
        policy_loss = -(action_logp * advantages.detach()).mean() - (craft_logp * advantages.detach()).mean()
        
        value_loss = F.mse_loss(values, returns)
        
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

async def train():
    cfg = json.load(open('config.json'))
    train_cfg = cfg['training']
    mc_cfg = cfg['minecraft']
    
    sys.setrecursionlimit(2000)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    os.makedirs(train_cfg['checkpoint_dir'], exist_ok=True)

    def create_env():
        return MinecraftEnv(
            username=mc_cfg['username_train'],
            host=mc_cfg['host'],
            port=mc_cfg['port'],
            auth=mc_cfg['auth'],
            version=mc_cfg['version'],
            view_range=mc_cfg['view_range']
        )
    
    env = create_env()
    
    model = MinecraftAgent(cfg['model']).to(device)
    agent = A2CAgent(model, lr=train_cfg['learning_rate'])
    
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

    pbar = tqdm(initial=start_step, position=0, leave=True)
    
    global_step = start_step

    while True:
        pbar.set_description(f"Step {global_step}")

        buffer = {
            'vox': [], 'stat': [], 'inv': [], 'act_hist': [], 'hx': [],
            'actions': [], 'logps': [], 'rewards': [], 'values': []
        }
        
        try:
            for _ in range(update_batch_size):
                if not env.is_ready:
                    print("Environment not ready, waiting...")
                    await asyncio.sleep(5)
                    continue

                vox, stat, inv, act_hist = env.get_observation()
                
                with torch.no_grad():
                    if hx is None:
                        hx = torch.zeros(1, model.dim, device=device)
                        
                    actions, logps, value, entropy, next_hx = agent.act(
                        vox.to(device),
                        stat.to(device),
                        inv.to(device),
                        act_hist.to(device),
                        hx
                    )
                
                action, craft_action = actions
                action_logp, craft_logp = logps
                
                reward = await env.step_atomic(action, craft_action)
                
                buffer['vox'].append(vox)
                buffer['stat'].append(stat)
                buffer['inv'].append(inv)
                buffer['act_hist'].append(act_hist)
                buffer['hx'].append(hx)
                buffer['actions'].append((action.cpu(), craft_action.cpu()))
                buffer['logps'].append((action_logp.cpu(), craft_logp.cpu()))
                buffer['rewards'].append(reward)
                buffer['values'].append(value.cpu())
                
                hx = next_hx.detach()

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
                returns
            )
            
            loss, p_loss, v_loss, ent = agent.update(batch)
            pbar.set_postfix({
                "Reward": f"{sum(buffer['rewards']):.2f}",
                "Loss": f"{loss:.3f}",
                "Ent": f"{ent:.3f}"
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
            else:
                print(f"Training error: {e}")
                continue
    
    env.close()

if __name__ == "__main__":
    asyncio.run(train())