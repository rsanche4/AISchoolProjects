# Author: Rafael Sanchez
# This program simulates an invasive entity known as "White Death,"  
# a virus-like force that lurks in dormancy, waiting for its moment to strike.  
#  
# - Healthy cells (red) grow and reproduce naturally.  
# - White Death remains inactive at first, scattered in small numbers.  
# - Once a critical threshold is met, it infects a cell, rapidly replicating  
#   and bursting forth, consuming the system in a wave of destruction.  
# - Most of the infection dies off, but a small search squad remains.  
# - The cycle repeats, ensuring the White Death never fully disappears.  
#  
# A self-sustaining cycle of annihilation and rebirth.  

import numpy as np
import matplotlib.pyplot as plt
import random

WORLD_SIZE = 200
def init_matrix(size):
    probability = 0.01
    matrix = np.random.choice([0.0, 1.0], size=(size, size), p=[1-probability, probability])
    matrix[size//2, size//2] = 0.5
    return matrix

def mod_matrix(matrix, size):
    moved_process = []
    number_sick_people = np.count_nonzero(matrix == 0.5)
    number_healthy = np.count_nonzero(matrix == 1)
    number_people = number_healthy + number_sick_people
    for x in range(size):
        for y in range(size):
            if matrix[x,y]==1 and (x,y) not in moved_process:
                killlife = random.choice(range(int((size*size)-number_people)))
                if killlife<number_people//4 and number_healthy>size:
                    matrix[x,y] = 0
                else:
                    movex = random.choice([-1, 1])
                    movey = random.choice([-1, 1])
                    new_x = (x + movex) % size
                    new_y = (y + movey) % size
                    if matrix[new_x][new_y]==0.5 and (number_healthy>size):
                        matrix[x,y] = 0.5
                        moved_process.append((x, y))
                    elif matrix[new_x,new_y]==1.0:
                        if random.randint(0, 1)==0:
                            matrix[new_x, (new_y+1)%size] = 1.0
                        moved_process.append((x, y))
                    elif (matrix[(x + 1) % size, (y + 1) % size]==0.5 or matrix[(x - 1) % size, (y + 1) % size]==0.5 or matrix[(x + 1) % size, (y - 1) % size]==0.5 or matrix[(x - 1) % size, (y - 1) % size]==0.5 or matrix[x, (y + 1) % size]==0.5 or matrix[x, (y - 1) % size]==0.5 or matrix[(x + 1) % size, y]==0.5 or matrix[(x - 1) % size, y]==0.5) and (number_healthy>size):
                        matrix[x,y] = 0.5
                        moved_process.append((x, y))
                    elif (matrix[(x + 1) % size, (y + 1) % size]==0.5 or matrix[(x - 1) % size, (y + 1) % size]==0.5 or matrix[(x + 1) % size, (y - 1) % size]==0.5 or matrix[(x - 1) % size, (y - 1) % size]==0.5 or matrix[x, (y + 1) % size]==0.5 or matrix[x, (y - 1) % size]==0.5 or matrix[(x + 1) % size, y]==0.5 or matrix[(x - 1) % size, y]==0.5):
                        moved_process.append((x, y))
                    else:
                        matrix[new_x,new_y] = 1
                        matrix[x,y] = 0
                        moved_process.append((new_x, new_y))

            if matrix[x,y]==0.5 and (x,y) not in moved_process:
                killvirus = random.choice(range(5))
                if killvirus==0 and number_sick_people>3:
                    matrix[x,y] = 0
                elif number_healthy==0:
                    matrix[x,y] = 0
                else:
                    movex = random.choice([-1, 1])
                    movey = random.choice([-1, 1])
                    new_x = (x + movex) % size
                    new_y = (y + movey) % size
                    if matrix[new_x][new_y]==1:
                        moved_process.append((x, y))
                    elif matrix[new_x,new_y]==0.5:
                        moved_process.append((x, y))
                    else:
                        matrix[new_x,new_y] = 0.5
                        matrix[x,y] = 0
                        moved_process.append((new_x, new_y))
    return matrix

fig, ax = plt.subplots()

matrix = init_matrix(WORLD_SIZE)

img = ax.imshow(matrix, cmap="coolwarm", vmin=0, vmax=1)
#somedata = []
#somedata_2 = []
while True:
    
    matrix = mod_matrix(matrix, WORLD_SIZE)      
    #somedata.append(number_sick_people)
    #somedata_2.append(number_healthy)

    img.set_data(matrix)
    plt.draw() 
    plt.pause(0.1)  

# data = somedata


# time_steps = list(range(len(data)))

# plt.figure(figsize=(8, 5)) 
# plt.plot(time_steps, data, marker='o', linestyle='-', color='b', label="Data Over Time")

# plt.xlabel("Time Steps")
# plt.ylabel("Data Values")
# plt.title("Number of sick people over time")
# plt.legend()  
# plt.grid(True) 

# plt.show()