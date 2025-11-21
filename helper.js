const { goals } = require('mineflayer-pathfinder');
const Vec3 = require('vec3');

const hostile = new Set();
let lastBlockBreak = { type: 0, by: 0, time: 0 };

function getStableId(name) {
    if (!name) return 0;
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
        hash = (hash << 5) - hash + name.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash) % 4096;
}

function getLookVector(pitch, yaw) {
    const sinYaw = Math.sin(yaw);
    const cosYaw = Math.cos(yaw);
    const sinPitch = Math.sin(pitch);
    const cosPitch = Math.cos(pitch);
    return new Vec3(-cosPitch * sinYaw, sinPitch, -cosPitch * cosYaw);
}

function setupListeners(bot) {
    if (!bot._listenersAttached) {
        bot._listenersAttached = true;

        bot.on('entitySwing', (entity) => {
            entity.lastSwung = Date.now();
        });

        bot.on('blockUpdate', (oldBlock, newBlock) => {
            if (oldBlock && oldBlock.type !== 0 && newBlock.type === 0) {
                const blockPos = oldBlock.position.offset(0.5, 0.5, 0.5);
                let bestCandidate = 0;
                let minDot = 0.5;

                for (const name in bot.entities) {
                    const entity = bot.entities[name];
                    if (entity === bot.entity || !entity.position) continue;
                    if (entity.type !== 'player') continue;

                    const dist = entity.position.distanceTo(blockPos);
                    if (dist > 6) continue;

                    const look = getLookVector(entity.pitch, entity.yaw);
                    const toBlock = blockPos.minus(entity.position.offset(0, 1.62, 0)).normalize();
                    const dot = look.dot(toBlock);

                    if (dot > minDot) {
                        minDot = dot;
                        bestCandidate = getStableId(entity.username);
                    }
                }

                if (bestCandidate !== 0) {
                    lastBlockBreak = {
                        type: getStableId(oldBlock.name),
                        by: bestCandidate,
                        time: Date.now()
                    };
                }
            }
        });

        bot.on('entityHurt', (entity) => {
            if (entity !== bot.entity) return;
            
            const now = Date.now();
            const attacker = Object.values(bot.entities).find(e => {
                if (e === bot.entity) return false;
                if (e.type !== 'player') return false;
                
                const dist = e.position.distanceTo(bot.entity.position);
                const timeSinceSwing = now - (e.lastSwung || 0);
                
                return dist < 4 && timeSinceSwing < 800;
            });

            if (attacker) {
                hostile.add(attacker.username);
            }
        });

        bot.on('chat', async (username, message) => {
            if (username === bot.username) return;
            
            const target = bot.players[username] ? bot.players[username].entity : null;
            if (!target || !target.position) return;

            const dist = target.position.distanceTo(bot.entity.position);
            if (dist > 10) return;

            const msg = message.toLowerCase();
            if (msg.includes('hey') || msg.includes('hi') || msg.includes('hello')) {
                try {
                    await bot.lookAt(target.position.offset(0, target.height, 0));
                    bot.swingArm();
                } catch (e) {}
            }
        });
    }
}

function hasLineOfSight(bot, target) {
    if (!bot || !target || !target.position) return false;
    
    const eyePos = bot.entity.position.offset(0, bot.entity.height, 0);
    const targetPos = target.position.offset(0, target.height * 0.5, 0);
    
    const dist = eyePos.distanceTo(targetPos);
    const dir = targetPos.minus(eyePos).normalize();
    
    const step = 0.5;
    let currentDist = 0;
    
    while (currentDist < dist) {
        const checkPos = eyePos.plus(dir.scaled(currentDist));
        const block = bot.blockAt(checkPos);
        
        if (block && block.boundingBox === 'block') {
            return false;
        }
        currentDist += step;
    }
    return true;
}

async function moveControl(bot, control, ms) {
    if (!bot || !bot.entity) return;
    try {
        if (bot.pathfinder && bot.pathfinder.isMoving) {
            bot.pathfinder.stop();
        }
    } catch (e) {}
    
    try {
        bot.setControlState(control, true);

        if (['forward', 'left', 'right'].includes(control)) {
            const checkInterval = setInterval(() => {
                if (bot.entity.isCollidedHorizontally) {
                    bot.setControlState('jump', true);
                } else {
                    bot.setControlState('jump', false);
                }
            }, 50);

            await new Promise(resolve => setTimeout(resolve, ms));
            clearInterval(checkInterval);
        } else {
            await new Promise(resolve => setTimeout(resolve, ms));
        }

        bot.setControlState(control, false);
        bot.setControlState('jump', false);
    } catch (e) {}
}

async function sprintJump(bot, ms) {
    if (!bot || !bot.entity) return;
    try {
        if (bot.pathfinder && bot.pathfinder.isMoving) {
            bot.pathfinder.stop();
        }
    } catch (e) {}
    
    try {
        bot.setControlState('forward', true);
        bot.setControlState('sprint', true);
        bot.setControlState('jump', true);
        
        await new Promise(resolve => setTimeout(resolve, ms));
        
        bot.setControlState('forward', false);
        bot.setControlState('sprint', false);
        bot.setControlState('jump', false);
    } catch (e) {}
}

async function randomWander(bot, radius) {
    if (!bot || !bot.entity || !bot.pathfinder) return false;
    
    const angle = Math.random() * Math.PI * 2;
    const dist = Math.random() * (radius - 5) + 5; 
    
    const dx = Math.cos(angle) * dist;
    const dz = Math.sin(angle) * dist;
    
    const targetPos = bot.entity.position.offset(dx, 0, dz);
    
    try {
        await Promise.race([
            bot.pathfinder.goto(new goals.GoalNear(targetPos.x, targetPos.y, targetPos.z, 1)),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 5000))
        ]);
        return true;
    } catch (e) {
        try { bot.pathfinder.stop(); } catch(err) {}
        return false;
    }
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

async function equipBestGear(bot) {
    if (!bot || !bot.inventory) return false;
    const items = bot.inventory.items();
    const armorSlots = ['head', 'torso', 'legs', 'feet'];
    const armorKeywords = ['helmet', 'chestplate', 'leggings', 'boots'];
    
    await equipBestWeapon(bot);
    
    for (let i = 0; i < 4; i++) {
        const keyword = armorKeywords[i];
        const slot = armorSlots[i];
        
        let bestItem = null;
        let bestRank = -1;
        
        items.forEach(item => {
            if (item.name.includes(keyword)) {
                let rank = 0;
                if (item.name.includes('leather')) rank = 1;
                if (item.name.includes('gold')) rank = 2;
                if (item.name.includes('chainmail')) rank = 3;
                if (item.name.includes('iron')) rank = 4;
                if (item.name.includes('diamond')) rank = 5;
                if (item.name.includes('netherite')) rank = 6;
                
                if (rank > bestRank) {
                    bestRank = rank;
                    bestItem = item;
                }
            }
        });
        
        if (bestItem) {
            try {
                await bot.equip(bestItem, slot);
            } catch (e) {}
        }
    }
}

function getResourceSummary(bot) {
    if (!bot || !bot.inventory) return { wood: 0, stone: 0, iron_ore: 0, iron_ingot: 0, diamond: 0, raw_food: 0, cooked_food: 0 };
    
    const items = bot.inventory.items();
    let wood = 0;
    let stone = 0;
    let iron_ore = 0;
    let iron_ingot = 0;
    let diamond = 0;
    let raw_food = 0;
    let cooked_food = 0;
    
    for (const item of items) {
        if (item.name.includes('log') || item.name.includes('planks')) wood += item.count;
        if (item.name.includes('cobblestone') || item.name.includes('stone')) stone += item.count;
        
        if (item.name.includes('iron_ore') || item.name.includes('raw_iron')) iron_ore += item.count;
        if (item.name.includes('iron_ingot')) iron_ingot += item.count;
        
        if (item.name.includes('diamond')) diamond += item.count;
        
        if (item.name.includes('beef') || item.name.includes('pork') || item.name.includes('chicken') || item.name.includes('mutton') || item.name.includes('rabbit')) {
            if (item.name.includes('cooked')) cooked_food += item.count;
            else raw_food += item.count;
        }
        if (item.name.includes('bread') || item.name.includes('apple')) cooked_food += item.count;
    }
    
    return { wood, stone, iron_ore, iron_ingot, diamond, raw_food, cooked_food };
}

async function placeBlock(bot) {
    if (!bot || !bot.entity) return false;
    try {
        const heldItem = bot.inventory.slots[bot.getEquipmentDestSlot('hand')];
        if (!heldItem || heldItem.type === 0) return false;
        
        const refBlock = bot.findBlock({
            matching: block => block && block.type !== 0 && block.boundingBox === 'block',
            maxDistance: 3.5
        });
        
        if (refBlock) {
            const face = new Vec3(0, 1, 0);
            try {
                await bot.placeBlock(refBlock, face);
                return true;
            } catch (e) {}
        }
    } catch (e) {}
    return false;
}

async function smeltItem(bot) {
    if (!bot || !bot.entity) return false;
    
    const items = bot.inventory.items();
    const smeltables = items.filter(i => i.name.includes('raw') || i.name.includes('ore') || i.name === 'potato' || i.name === 'kelp');
    const fuels = items.filter(i => i.name.includes('coal') || i.name.includes('plank') || i.name.includes('log') || i.name === 'charcoal');
    
    let furnaceBlock = bot.findBlock({ matching: bot.registry.blocksByName.furnace.id, maxDistance: 4 });
    
    if (!furnaceBlock || smeltables.length === 0 || fuels.length === 0) return false;
    
    try {
        const furnace = await bot.openFurnace(furnaceBlock);
        
        if (furnace.outputItem) {
            await furnace.takeOutput();
        }
        
        if (smeltables.length > 0 && !furnace.inputItem) {
            await furnace.putInput(smeltables[0]);
        }
        
        if (fuels.length > 0 && !furnace.fuelItem) {
             await furnace.putFuel(fuels[0]);
        }
        
        await new Promise(resolve => setTimeout(resolve, 200));
        furnace.close();
        return true;
    } catch (e) {
        return false;
    }
}

function getActiveEffects(bot) {
    let badFx = 0;
    let goodFx = 0;
    
    if (bot.entity && bot.entity.effects) {
        const effects = Object.values(bot.entity.effects);
        for (const eff of effects) {
             const id = eff.id;
             if ([19, 20, 2, 18, 7, 15, 9, 17, 27].includes(id)) badFx = 1;
             if ([10, 1, 5, 11, 12, 22, 8, 16, 26].includes(id)) goodFx = 1;
        }
    }
    return { badFx, goodFx };
}

async function attackMob(bot) {
    if (!bot || !bot.entity || !bot.entity.position) return 0;
    setupListeners(bot);
    
    try {
        const entity = bot.nearestEntity(e => 
            e && e.position && 
            (
                (e.type === 'mob' && e.name !== 'sheep' && e.name !== 'cow' && e.name !== 'pig') ||
                (e.type === 'player' && hostile.has(e.username))
            )
        );

        if (!entity) return 0;

        const dist = bot.entity.position.distanceTo(entity.position);
        
        if (dist < 3.5) {
            if (!hasLineOfSight(bot, entity)) return 0;

            await equipBestWeapon(bot);
            await bot.lookAt(entity.position.offset(0, entity.height * 0.5, 0));
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
                    try { bot.pathfinder.stop(); } catch(err) {}
                    return 0; 
                }
            }
        }
    } catch (e) { return 0; }
    return 0;
}

async function mineBlock(bot) {
    if (!bot || !bot.entity || !bot.entity.position) return 0;
    
    const targets = ['diamond_ore', 'deepslate_diamond_ore', 'iron_ore', 'deepslate_iron_ore', 'gold_ore', 'coal_ore', 'oak_log', 'cobblestone'];
    
    try {
        const blockIds = targets.map(name => bot.registry.blocksByName[name]?.id).filter(id => id !== undefined);
        
        const targetBlock = bot.findBlock({
            matching: (block) => {
                return blockIds.includes(block.type); 
            },
            maxDistance: 6,
            count: 1
        });

        if (targetBlock) {
            const dist = bot.entity.position.distanceTo(targetBlock.position);
            
            if (dist < 4) {
                await bot.lookAt(targetBlock.position.offset(0.5, 0.5, 0.5));
                
                const blockCursor = bot.blockAtCursor(5);
                
                if (blockCursor && !blockCursor.position.equals(targetBlock.position) && blockCursor.type !== 0) {
                     try {
                        await bot.equip(bot.pathfinder.bestHarvestTool(blockCursor), 'hand');
                        await bot.dig(blockCursor);
                        return 1.0;
                     } catch (e) { return -0.5; }
                }
                
                try {
                    await bot.equip(bot.pathfinder.bestHarvestTool(targetBlock), 'hand');
                    await bot.dig(targetBlock);
                    
                    if (targetBlock.name.includes('diamond')) return 50.0;
                    if (targetBlock.name.includes('iron')) return 20.0;
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
                        try { bot.pathfinder.stop(); } catch(err) {}
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
    setupListeners(bot);

    const center = bot.entity.position.floored();
    
    if (!center || typeof center.offset !== 'function') {
        return null;
    }

    const width = viewRange * 2 + 1;
    const totalSize = width * width * width;
    
    const buffer = Buffer.alloc(totalSize * 15);
    
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
                const data_offset = idx * 15;
                
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
        
        if (entity.type === 'mob' || entity.type === 'player' || entity.type === 'projectile' || entity.type === 'object') {
            const dx = Math.floor(entity.position.x - center.x);
            const dy = Math.floor(entity.position.y - center.y);
            const dz = Math.floor(entity.position.z - center.z);

            const idx = getIndex(dx, dy, dz);
            if (idx !== -1) {
                const data_offset = idx * 15;
                let entityId = 2000;
                if (entity.type === 'player') entityId = 2001;
                if (entity.type === 'projectile') {
                    entityId = 2002;
                    if (entity.name && (entity.name.includes('potion') || entity.metadata[8] === 1)) {
                        entityId = 2005;
                    }
                }
                if (entity.type === 'object') {
                    entityId = 2003;
                    if (entity.objectType === 'Thrown Potion' || entity.name === 'thrown_potion') {
                        entityId = 2005;
                    } else {
                        const item = entity.getDroppedItem ? entity.getDroppedItem() : null;
                        if (item) {
                            entityId = getStableId(item.name);
                        }
                    }
                }

                buffer.writeUInt16LE(entityId, data_offset);
                buffer.writeUInt8(Math.max(0, entity.health || 0), data_offset + 2);
                
                const equipment = entity.equipment || [];
                const hand = equipment[0] ? getStableId(equipment[0].name) : 0;
                const head = equipment[5] ? getStableId(equipment[5].name) : 0;
                const chest = equipment[4] ? getStableId(equipment[4].name) : 0;
                const legs = equipment[3] ? getStableId(equipment[3].name) : 0;
                const feet = equipment[2] ? getStableId(equipment[2].name) : 0;
                
                buffer.writeUInt16LE(hand, data_offset + 3);
                buffer.writeUInt16LE(head, data_offset + 5);
                buffer.writeUInt16LE(chest, data_offset + 7);
                buffer.writeUInt16LE(legs, data_offset + 9);
                buffer.writeUInt16LE(feet, data_offset + 11);
                
                let nameToHash = entity.username || entity.name || "";
                if (entity.type === 'object') {
                    const item = entity.getDroppedItem ? entity.getDroppedItem() : null;
                    if (item) nameToHash = item.name;
                }
                const nameId = getStableId(nameToHash);
                buffer.writeUInt16LE(nameId, data_offset + 13);
            }
            
            const distance = entity.position.distanceTo(bot.entity.position);
            if (distance < nearest_distance) {
                nearest_distance = distance;
                nearest_mob = entity;
            }
        }
    }

    let globalNearestPlayer = null;
    let globalNearestDist = Infinity;

    for (const name in bot.entities) {
        const entity = bot.entities[name];
        if (entity === bot.entity || !entity.position) continue;
        if (entity.type === 'player') {
             const d = entity.position.distanceTo(bot.entity.position);
             if (d < globalNearestDist) {
                 globalNearestDist = d;
                 globalNearestPlayer = entity.position;
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
    
    let brokenType = 0;
    let breakerId = 0;
    
    if (Date.now() - lastBlockBreak.time < 1000) {
        brokenType = lastBlockBreak.type;
        breakerId = lastBlockBreak.by;
    }

    const resources = getResourceSummary(bot);
    const effects = getActiveEffects(bot);

    const status = [
        bot.health || 0,
        bot.food || 0,
        bot.foodSaturation || 0,
        bot.oxygenLevel || 0,
        yaw,
        pitch,
        vel_x,
        vel_y,
        vel_z,
        nearest_mob ? (16 - Math.min(nearest_distance, 16)) / 16 : 0,
        pos.x,
        pos.y,
        pos.z,
        brokenType,
        breakerId,
        globalNearestPlayer ? (globalNearestPlayer.x - pos.x) : 0,
        globalNearestPlayer ? (globalNearestPlayer.y - pos.y) : 0,
        globalNearestPlayer ? (globalNearestPlayer.z - pos.z) : 0,
        globalNearestDist !== Infinity ? globalNearestDist : -1,
        resources.wood,
        resources.stone,
        resources.iron_ingot + resources.iron_ore,
        resources.cooked_food + resources.raw_food,
        effects.badFx,
        effects.goodFx,
        resources.diamond
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
        wood_count: resources.wood,
        stone_count: resources.stone,
        iron_ore_count: resources.iron_ore,
        iron_ingot_count: resources.iron_ingot,
        diamond_count: resources.diamond,
        cooked_food_count: resources.cooked_food
    };
}

module.exports = { getObservation, moveControl, sprintJump, randomWander, attackMob, mineBlock, equipBestGear, placeBlock, smeltItem };