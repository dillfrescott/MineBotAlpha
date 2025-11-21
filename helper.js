const { goals } = require('mineflayer-pathfinder');

function getStableId(name) {
    if (!name) return 0;
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
        hash = (hash << 5) - hash + name.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash) % 1024;
}

async function moveControl(bot, control, ms) {
    if (!bot || !bot.entity) return;
    try {
        if (bot.pathfinder) bot.pathfinder.stop();
    } catch (e) {}
    
    bot.setControlState(control, true);
    await new Promise(resolve => setTimeout(resolve, ms));
    bot.setControlState(control, false);
}

async function equipBestWeapon(bot) {
    if (!bot || !bot.inventory) return false;
    const items = bot.inventory.items();
    
    const weapons = ['iron_sword', 'stone_sword', 'wooden_sword'];
    let found = null;
    
    for (const w of weapons) {
        found = items.find(i => i.name === w);
        if (found) break;
    }
    
    if (!found) {
        found = items.find(i => i.name.includes('axe'));
    }

    if (found) {
        try {
            await bot.equip(found, 'hand');
            return true;
        } catch (e) { return false; }
    }
    return false;
}

function getWoodCounts(bot) {
    if (!bot || !bot.inventory) return 0;
    const items = bot.inventory.items();
    let count = 0;
    const woods = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log', 'planks'];
    
    for (const item of items) {
        if (woods.includes(item.name) || item.name.includes('planks')) {
            count += item.count;
        }
    }
    return count;
}

async function attackMob(bot) {
    if (!bot || !bot.entity || !bot.entity.position) return 0;
    
    try {
        const entity = bot.nearestEntity(e => 
            e && e.position && (e.type === 'mob' || e.type === 'player') && 
            e.name !== 'sheep' && e.name !== 'cow' && e.name !== 'pig'
        );

        if (!entity) return 0;

        const dist = bot.entity.position.distanceTo(entity.position);
        await equipBestWeapon(bot);

        if (dist < 3.5) {
            bot.lookAt(entity.position.offset(0, entity.height * 0.5, 0));
            bot.attack(entity);
            return 5.0;
        } else {
            if (bot.pathfinder) {
                try {
                    await Promise.race([
                        bot.pathfinder.goto(new goals.GoalFollow(entity, 3)),
                        new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 4000))
                    ]);
                    return 1.0;
                } catch (e) { 
                    bot.pathfinder.stop();
                    return 0; 
                }
            }
        }
    } catch (e) { return 0; }
    return 0;
}

async function mineBlock(bot) {
    if (!bot || !bot.entity || !bot.entity.position) return 0;
    
    const targets = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log', 'cobblestone', 'iron_ore', 'coal_ore'];
    
    try {
        const blockIds = targets.map(name => bot.registry.blocksByName[name]?.id).filter(id => id !== undefined);
        
        const targetBlock = bot.findBlock({
            matching: (block) => {
                return blockIds.includes(block.type) && bot.canSeeBlock(block);
            },
            maxDistance: 6,
            count: 1
        });

        if (targetBlock) {
            const dist = bot.entity.position.distanceTo(targetBlock.position);
            
            if (dist < 4) {
                try {
                    await bot.equip(bot.pathfinder.bestHarvestTool(targetBlock), 'hand');
                    await bot.dig(targetBlock);
                    return 10.0;
                } catch (e) { return -1.0; }
            } else {
                if (bot.pathfinder) {
                    try {
                        await Promise.race([
                            bot.pathfinder.goto(new goals.GoalBlock(targetBlock.position.x, targetBlock.position.y, targetBlock.position.z)),
                            new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 4000))
                        ]);
                        return 1.0;
                    } catch (e) { 
                        bot.pathfinder.stop();
                        return 0; 
                    }
                }
            }
        }
    } catch (e) { return 0; }
    return 0;
}

function getObservation(bot, viewRange) {
    if (!bot || !bot.entity || !bot.entity.position) return null;

    const center = bot.entity.position.floored();
    
    if (!center || typeof center.offset !== 'function') {
        return null;
    }

    const width = viewRange * 2 + 1;
    const totalSize = width * width * width;
    
    const buffer = Buffer.alloc(totalSize * 3);
    
    const getIndex = (x, y, z) => {
        const lx = x + viewRange;
        const ly = y + viewRange;
        const lz = z + viewRange;
        if (lx < 0 || lx >= width || ly < 0 || ly >= width || lz < 0 || lz >= width) return -1;
        return (ly * width * width) + (lz * width) + lx;
    }

    for (let y = -viewRange; y <= viewRange; y++) {
        for (let z = -viewRange; z <= viewRange; z++) {
            for (let x = -viewRange; x <= viewRange; x++) {
                const idx = getIndex(x, y, z);
                if (idx === -1) continue;
                const data_offset = idx * 3;
                
                const block = bot.blockAt(center.offset(x, y, z));
                
                const id = (block && block.name) ? getStableId(block.name) : 0;
                buffer.writeUInt16LE(id, data_offset);
                
                const light_level = (block && block.light !== undefined) ? block.light : 0;
                buffer.writeUInt8(light_level, data_offset + 2);
            }
        }
    }

    let nearest_mob = null;
    let nearest_distance = Infinity;
    
    for (const name in bot.entities) {
        const entity = bot.entities[name];
        if (entity === bot.entity || !entity.position) continue;
        
        if (entity.type === 'mob' || entity.type === 'player' || entity.type === 'projectile') {
            const dx = Math.floor(entity.position.x - center.x);
            const dy = Math.floor(entity.position.y - center.y);
            const dz = Math.floor(entity.position.z - center.z);

            const idx = getIndex(dx, dy, dz);
            if (idx !== -1) {
                const data_offset = idx * 3;
                let entityId = 2000;
                if (entity.type === 'player') entityId = 2001;
                if (entity.type === 'projectile') entityId = 2002;

                buffer.writeUInt16LE(entityId, data_offset);
                buffer.writeUInt8(Math.max(0, entity.health || 0), data_offset + 2);
            }
            
            const distance = entity.position.distanceTo(bot.entity.position);
            if (distance < nearest_distance) {
                nearest_distance = distance;
                nearest_mob = entity;
            }
        }
    }

    const bot_entity = bot.entity;
    const vel = bot_entity ? bot_entity.velocity : null;
    const pos = bot_entity ? bot_entity.position : {x:0, y:0, z:0};
    
    const yaw = bot_entity ? bot_entity.yaw : 0;
    const pitch = bot_entity ? bot_entity.pitch : 0;
    const vel_x = vel ? vel.x : 0;
    const vel_y = vel ? vel.y : 0;
    const vel_z = vel ? vel.z : 0;

    const status = [
        bot.health || 0,
        bot.food || 0,
        bot.foodSaturation || 0,
        yaw,
        pitch,
        vel_x,
        vel_y,
        vel_z,
        nearest_mob ? (16 - Math.min(nearest_distance, 16)) / 16 : 0,
        pos.x,
        pos.y,
        pos.z
    ];
    
    const inventoryIDs = new Array(36).fill(0);
    let hasWeapon = 0;
    
    const inventory_items = typeof bot.inventory.items === 'function' ? bot.inventory.items() : [];

    inventory_items.forEach(item => {
        if (!item) return; 

        const id = getStableId(item.name);
        
        if (item.name.includes('sword') || item.name.includes('axe')) {
            hasWeapon = 1;
        }
        
        if (item.slot >= 9 && item.slot <= 44) {
            inventoryIDs[item.slot - 9] = id;
        } else if (item.slot >= 36 && item.slot <= 44) {
            inventoryIDs[item.slot - 36] = id;
        }
    });

    return {
        voxels: buffer.toString('base64'),
        status: status,
        inventory: inventoryIDs,
        has_weapon: hasWeapon,
        nearest_mob_distance: nearest_distance,
        nearest_mob_type: nearest_mob ? nearest_mob.type : null,
        wood_count: getWoodCounts(bot)
    };
}

module.exports = { getObservation, moveControl, attackMob, mineBlock };