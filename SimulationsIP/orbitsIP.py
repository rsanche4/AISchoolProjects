import numpy as np
import matplotlib.pyplot as plt
import math

WORLD_SIZE = 20

# Mass in Kilograms, r is distance in meters
def gravitational_force(m1, m2, r):
    G = 1  # Gravitational constant 
    return G * (m1 * m2) / r**2

def acceleration_obj(F, m):
    return F/m

def move(direction, speed, current_x, current_y):
    new_x = current_x
    new_y = current_y
    if direction=="X":
        new_x = current_x + speed
    if direction=="Y":
        new_y = current_y + speed
    return new_x, new_y

def scan_all_1s(matrix, size, objx, objy):

    allonescoords = []
    for x in range(size):
        for y in range(size):
            if matrix[x,y]==1 and x!=objx and y!=objy:
                allonescoords.append((x,y))
    
    return allonescoords

def distance(m1x, m2x, m1y, m2y):
    return np.sqrt(((m2x - m1x)**2)+((m2y - m1y)**2))

def direction_obj(x1, x2, y1, y2):
    # Relative position vector
    dx = x1 - x2
    dy = y1 - y2

    # Calculate angle in radians
    theta_radians = math.atan2(dy, dx)

    # Convert to degrees
    theta_degrees = math.degrees(theta_radians)
    return theta_degrees

def init_matrix(size):
    matrix = np.zeros((size, size))
    matrix[size//2, size//2] = 1
    matrix[5, 5] = 1
    return matrix

def mod_matrix(matrix, size):
    moved_process = []
    for x in range(size):
        for y in range(size):
            if matrix[x,y]==1 and (x,y) not in moved_process:
                
                
                # Check the locations of all the ones around us
                allaround_scan = scan_all_1s(matrix, size, x, y)
                
                acc_vectors = []
                for obj in allaround_scan:
                    acceleration = acceleration_obj(gravitational_force(100, 100, distance(y, obj[1], x, obj[0])), 100)
                    direction = direction_obj(y, obj[1], x, obj[0])
                    
                    # great! Now that we have our degrees direction, and our acceleration, we need to break acceleration into ax and ay components to determine net vector acceleration. We make hypo = accelration, and using trig, we get the corresponding opp and adj
                    accx = acceleration*math.cos(direction)
                    accy = acceleration*math.sin(direction)
                    accvector = np.array([
                        [accx],
                        [accy]
                    ])
                    acc_vectors.append(accvector)

                
                accnet = np.sum(acc_vectors, axis=0)
                
                

                _, new_y = move("X", int(1*accnet[0,0]), y, x)
                #print(new_x)
                new_x, _ = move("Y", int(1*accnet[1,0]), y, x)
                matrix[new_x%size,new_y%size] = 1
                if new_x!=x or new_y!=y:
                    matrix[x,y]=0
                #print(matrix)
                moved_process.append((new_x,new_y))
    return matrix

fig, ax = plt.subplots()

matrix = init_matrix(WORLD_SIZE)

img = ax.imshow(matrix, cmap="gray", vmin=0, vmax=1)

while True:
    
    matrix = mod_matrix(matrix, WORLD_SIZE)      
    #somedata.append(number_sick_people)
    #somedata_2.append(number_healthy)

    img.set_data(matrix)
    plt.draw() 
    plt.pause(0.1)  

