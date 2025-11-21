function getStableId(name) {
    if (!name) return 0;
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
        hash = (hash << 5) - hash + name.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash) % 1024;
}

function getObservation(bot, viewRange) {
    if (!bot || !bot.entity || !bot.entity.position) return null;

    const center = bot.entity.position.floored();
    
    if (!center || typeof center.offset !== 'function') {
        console.log("Transient state: Center object invalid.");
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
        
        if (entity.type === 'mob' || entity.type === 'player') {
            const dx = Math.floor(entity.position.x - center.x);
            const dy = Math.floor(entity.position.y - center.y);
            const dz = Math.floor(entity.position.z - center.z);

            const idx = getIndex(dx, dy, dz);
            if (idx !== -1) {
                const data_offset = idx * 3;
                buffer.writeUInt16LE(2000 + (entity.type === 'player' ? 1 : 0), data_offset);
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
    
    const yaw = bot_entity ? bot_entity.yaw : 0;
    const pitch = bot_entity ? bot_entity.pitch : 0;
    const vel_x = vel ? vel.x : 0;
    const vel_y = vel ? vel.y : 0;
    const vel_z = vel ? vel.z : 0;

    const heldItem = bot.heldItem;
    
    const status = [
        bot.health || 0,
        bot.food || 0,
        bot.foodSaturation || 0,
        yaw,
        pitch,
        vel_x,
        vel_y,
        vel_z,
        nearest_mob ? (16 - Math.min(nearest_distance, 16)) / 16 : 0
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
        nearest_mob_type: nearest_mob ? nearest_mob.type : null
    };
}

module.exports = { getObservation };