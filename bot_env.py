import sys
import time
import torch
import base64
import math
import numpy as np
from javascript import require, Once, On
from threading import BrokenBarrierError
import asyncio

Vec3 = require('vec3')

class MinecraftEnv:
    def __init__(self, username="Agent", host='127.0.0.1', port=25565, 
                 auth='offline', version='1.21.1', view_range=3):
        print(f"Connecting {username}...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.mineflayer = require('mineflayer')
        self.helper = require('./helper.js')
        self.pathfinder = require('mineflayer-pathfinder')
        
        self.view_range = view_range
        self.grid_len = (2 * view_range + 1) ** 3
        self.max_actions = 10
        self.inventory_len = 36
        self.last_health = 20.0
        self.last_food = 20.0
        self.current_health = 20.0
        self.current_food = 20.0
        self.last_wood_count = 0
        
        self.bot = self.mineflayer.createBot({
            'host': host, 'port': port, 'username': username, 'auth': auth,
            'hideErrors': False, 'version': version
        })
        
        self.bot.loadPlugin(self.pathfinder.pathfinder)
        self.tool = require('mineflayer-tool').Tool(self.bot)
        self.is_ready = False
        self.action_history = torch.zeros(self.max_actions, 12, device=device)
        
        self.wood_sources = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log']
        
        @On(self.bot, 'end')
        def on_end(this, reason):
            print(f"Bot disconnected: {reason}. Setting is_ready=False.")
            self.is_ready = False

        @On(self.bot, 'death')
        def on_death(this):
            self.bot.respawn() 

        @Once(self.bot, 'spawn')
        def on_spawn(this):
            print(f"{username} Spawned!")
            self.bot.chat("Agent Online.")
            time.sleep(1.0)
            self.is_ready = True
            
        start = time.time()
        while not self.is_ready and time.time() - start < 60:
            time.sleep(0.1)

    def craft_any_item(self):
        try:
            possible_recipes = self.bot.recipesFor(0, None, 1, True)
            if not possible_recipes or len(possible_recipes) == 0:
                return False
                
            best_recipe = None
            best_value = -1

            craftable_recipes = []
            
            for recipe in possible_recipes:
                if recipe.ingredients:
                    is_craftable = True
                    for ingredient in recipe.ingredients:
                        item_id = ingredient.id
                        count = ingredient.count
                        
                        item = self.bot.registry.items[item_id]
                        
                        if not item:
                            is_craftable = False
                            break
                        
                        inventory_count = self.bot.inventory.count(item_id)
                        if inventory_count < count:
                            is_craftable = False
                            break
                    if is_craftable:
                        craftable_recipes.append(recipe)
            
            if not craftable_recipes:
                return False
                
            for recipe in craftable_recipes:
                item_name = self.bot.registry.items[recipe.result.id].name
                value = 1.0 
                if 'sword' in item_name: value = 10.0
                elif 'axe' in item_name: value = 8.0
                elif 'pickaxe' in item_name: value = 8.0
                elif 'shovel' in item_name: value = 5.0
                elif 'shield' in item_name: value = 15.0
                elif 'planks' in item_name: value = 3.0
                elif 'stick' in item_name: value = 2.0
                elif 'ingot' in item_name: value = 5.0
                elif 'block' in item_name: value = 3.0
                elif 'cooked' in item_name: value = 7.0
                
                if value > best_value:
                    best_value = value
                    best_recipe = recipe

            if best_recipe:
                self.bot.craft(best_recipe, 1, None)
                return True
                
        except (BrokenBarrierError, Exception):
            pass
        return False

    def get_observation(self):
        dummy_vox = torch.zeros((1, 2, self.grid_len), dtype=torch.long)
        dummy_inv = torch.zeros((1, 1, self.inventory_len), dtype=torch.long)
        dummy_status = torch.zeros((1, 1, 9), dtype=torch.float)
        
        if not self.is_ready:
            return (dummy_vox,
                    dummy_status,
                    dummy_inv,
                    self.action_history.clone().unsqueeze(0))
        
        try:
            data = self.helper.getObservation(self.bot, self.view_range)
            if not data: raise ValueError("No observation data")
            
            raw_bytes = base64.b64decode(data['voxels'])
            raw_data = np.frombuffer(raw_bytes, dtype=np.uint8)
            light_levels = raw_data[2::3].astype(np.int64)
            
            light_levels = np.clip(light_levels, 0, 16)

            id_b1 = raw_data[0::3]
            id_b2 = raw_data[1::3]
            id_bytes_2d = np.stack([id_b1, id_b2], axis=-1).ravel()
            block_ids = np.frombuffer(id_bytes_2d.tobytes(), dtype='<u2').astype(np.int64)
            
            block_ids = np.clip(block_ids, 0, 4095) 
            
            voxels = torch.stack([
                torch.from_numpy(block_ids),
                torch.from_numpy(light_levels)
            ]).unsqueeze(0).long()
            
            status_list = list(data['status'])
            status = torch.tensor([status_list], dtype=torch.float).unsqueeze(0)
            
            inventory_list = list(data['inventory'])
            inventory = torch.tensor([inventory_list], dtype=torch.long).unsqueeze(0)
            
            self.current_health = status_list[0]
            self.current_food = status_list[1]
            self.last_wood_count = data['wood_count']
            
            return voxels, status, inventory, self.action_history.clone().unsqueeze(0)
        except (BrokenBarrierError, Exception):
            return dummy_vox, dummy_status, dummy_inv, self.action_history.clone().unsqueeze(0)

    async def execute_action(self, action_type):
        if not self.is_ready:
            return 0

        reward = 0
        bot = self.bot

        try:
            if action_type == 0:
                reward += await self.helper.attackMob(self.bot)
            
            elif action_type == 1:
                await self.helper.moveControl(bot, "back", 200)

            elif action_type == 2:
                if self.current_health < 10 and bot.food > 8:
                    if callable(bot.inventory.items):
                        inventory_items = bot.inventory.items()
                    elif isinstance(bot.inventory.items, list):
                        inventory_items = bot.inventory.items
                    else:
                        inventory_items = []

                    food_item = next((i for i in inventory_items
                        if i.name in ['cooked_beef', 'cooked_porkchop', 'bread']), None)
                        
                    if food_item:
                        bot.equip(food_item, 'hand')
                        bot.activateItem()
                        reward += 3.0

            elif action_type == 3:
                await self.helper.moveControl(bot, "forward", 200)

            elif action_type == 4:
                await self.helper.moveControl(bot, "back", 200)

            elif action_type == 5:
                await self.helper.moveControl(bot, "left", 200)

            elif action_type == 6:
                await self.helper.moveControl(bot, "right", 200)

            elif action_type == 7:
                await self.helper.moveControl(bot, "jump", 100)

            elif action_type == 8:
                yaw = bot.entity.yaw - 0.3
                bot.look(yaw, bot.entity.pitch)

            elif action_type == 9:
                yaw = bot.entity.yaw + 0.3
                bot.look(yaw, bot.entity.pitch)
            
            elif action_type == 10:
                reward += await self.helper.mineBlock(self.bot)
                
            elif action_type == 11:
                reward += await self.helper.attackMob(self.bot)
                
        except (BrokenBarrierError, Exception):
            return 0

        return reward

    async def step_atomic(self, action, craft_action):
        if not self.is_ready: 
            return 0
            
        reward = 0
        
        try:
            reward += await self.execute_action(action.item())
            
            if craft_action.item() == 0:
                if self.craft_any_item():
                    reward += 10.0
            
            if self.last_wood_count > 0:
                reward += 0.1
                
            try:
                if self.current_food < 6:
                    reward -= 0.5
                elif self.current_food == 0:
                    reward -= 2.0
                    
                if self.current_health < 5:
                    reward -= 1.0
                    
                damage_taken = max(0, self.last_health - self.current_health)
                if damage_taken > 0:
                    reward -= damage_taken * 5.0
                    
                health_gained = max(0, self.current_health - self.last_health)
                if health_gained > 0:
                    reward += health_gained * 1.5
                    
            except:
                pass
                
            self.last_health = self.current_health
            self.last_food = self.current_food
            
            device = self.action_history.device
            B = action.size(0)

            new_action_full = torch.cat([
                torch.nn.functional.one_hot(action, num_classes=12).float(),
                torch.nn.functional.one_hot(craft_action, num_classes=4).float(),
            ], dim=-1)

            new_action = new_action_full[:, :12]

            self.action_history = torch.cat([self.action_history[1:], new_action])
            
        except (BrokenBarrierError, Exception):
            pass
        
        return reward