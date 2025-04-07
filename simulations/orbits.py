import numpy as np
import matplotlib.pyplot as plt
import math

WORLD_SIZE = 100
MIN_DISTANCE = 1.0  # Minimum distance to prevent extreme forces
DAMPING = 0.9       # Damping factor to prevent oscillations

def gravitational_force(m1, m2, r):
    G = 1  # Gravitational constant
    if r < MIN_DISTANCE:
        r = MIN_DISTANCE
    return G * (m1 * m2) / (r**2)

def move(current_x, current_y, speed_x, speed_y):
    new_x = current_x + speed_x
    new_y = current_y + speed_y
    return new_x, new_y

def scan_all_1s(matrix, size):
    ones_coords = []
    for x in range(size):
        for y in range(size):
            if matrix[x,y] == 1:
                ones_coords.append((x,y))
    return ones_coords

def distance(m1x, m1y, m2x, m2y):
    return max(MIN_DISTANCE, np.sqrt((m2x - m1x)**2 + (m2y - m1y)**2))

def direction_obj(x1, y1, x2, y2):
    dx = x2 - x1  # Direction from 1 to 2
    dy = y2 - y1
    return math.atan2(dy, dx)

def init_matrix(size):
    matrix = np.zeros((size, size))
    matrix[size//2, size//2] = 1  # Central mass
    matrix[5, 5] = 1              # Another mass
    matrix[15, 15] = 1            # Third mass
    return matrix

def mod_matrix(matrix, size):
    # Create velocity dictionary if it doesn't exist
    if not hasattr(mod_matrix, "velocities"):
        mod_matrix.velocities = {}
        coords = scan_all_1s(matrix, size)
        for coord in coords:
            mod_matrix.velocities[coord] = [0, 0]  # [vx, vy]
    
    new_matrix = np.zeros_like(matrix)
    new_velocities = {}
    processed = set()
    
    all_objects = scan_all_1s(matrix, size)
    
    for i, (x1, y1) in enumerate(all_objects):
        if (x1, y1) in processed:
            continue
            
        vx, vy = mod_matrix.velocities.get((x1, y1), [0, 0])
        total_fx, total_fy = 0, 0
        
        for j, (x2, y2) in enumerate(all_objects):
            if i == j:
                continue
                
            dist = distance(x1, y1, x2, y2)
            force = gravitational_force(100, 100, dist)
            angle = direction_obj(x1, y1, x2, y2)
            
            fx = force * math.cos(angle)
            fy = force * math.sin(angle)
            
            total_fx += fx
            total_fy += fy
        
        # Update velocity with damping
        vx = (vx + total_fx) * DAMPING
        vy = (vy + total_fy) * DAMPING
        
        # Calculate new position
        new_x, new_y = move(x1, y1, vx, vy)
        new_x = int(round(new_x)) % size
        new_y = int(round(new_y)) % size
        
        # Check for collisions/merging
        if (new_x, new_y) in new_velocities:
            # Merge objects (just keep one)
            processed.add((new_x, new_y))
            # Combine velocities (momentum conservation)
            existing_vx, existing_vy = new_velocities[(new_x, new_y)]
            new_velocities[(new_x, new_y)] = [(existing_vx + vx)/2, (existing_vy + vy)/2]
        else:
            new_matrix[new_x, new_y] = 1
            new_velocities[(new_x, new_y)] = [vx, vy]
            processed.add((new_x, new_y))
    
    mod_matrix.velocities = new_velocities
    return new_matrix

fig, ax = plt.subplots()
matrix = init_matrix(WORLD_SIZE)
img = ax.imshow(matrix, cmap="gray", vmin=0, vmax=1)

try:
    for _ in range(100):  # Run for 100 frames
        matrix = mod_matrix(matrix, WORLD_SIZE)
        img.set_data(matrix)
        plt.draw()
        plt.pause(0.1)
except KeyboardInterrupt:
    print("Simulation stopped")