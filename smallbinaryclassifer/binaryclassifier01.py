import cv2
import numpy as np
import glob
import os
import math

# lets read our data!

# Folder containing PNG images
folder_path = "TrainingData"

# Dictionary to store images as numpy arrays
image_arrays0 = {}

# Get all matching PNG files
image_files = glob.glob(os.path.join(folder_path, "t0_*.png"))

# Loop through each image
for image_file in image_files:
    # Extract the number from the filename
    filename = os.path.basename(image_file)  # Get "t1_numberhere.png"
    number = filename.split("_")[1].split(".")[0]  # Extract "numberhere"

    # Read the image in grayscale mode (0-255 values)
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Convert to binary: 255 (white) -> 1, 0 (black) remains 0
    binary_image = (image == 255).astype(np.uint8)  # Convert True/False to 1/0

    # Reshape into a column vector (1D but vertical)
    binary_image = binary_image.reshape(-1, 1)
    # Store in dictionary
    image_arrays0[number] = binary_image

image_arrays1 = {}

# Get all matching PNG files
image_files = glob.glob(os.path.join(folder_path, "t1_*.png"))

# Loop through each image
for image_file in image_files:
    # Extract the number from the filename
    filename = os.path.basename(image_file)  # Get "t1_numberhere.png"
    number = filename.split("_")[1].split(".")[0]  # Extract "numberhere"

    # Read the image in grayscale mode (0-255 values)
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Convert to binary: 255 (white) -> 1, 0 (black) remains 0
    binary_image = (image == 255).astype(np.uint8)  # Convert True/False to 1/0

    # Reshape into a column vector (1D but vertical)
    binary_image = binary_image.reshape(-1, 1)

    # Store in dictionary
    image_arrays1[number] = binary_image

# ourimagearrays. So basically each index corresponds to its number label!
image_labels_training_data = [image_arrays0, image_arrays1]

# Awesome!! so now that we can read our data! Now that we have that, we need to kinda create a linear combination of that data.
# Training happens here

image_width = 8
image_height = 8
n = 2
learningrate = 0.1
weights = np.zeros((image_width * image_height, 1))
b = 0 # the b term
number_of_iterations = 2000

for iteration in range(number_of_iterations):

    # loss function i still need to add it here for monitoring
    # lets add the loss function and add each result of it to a list so that we can then plot it so we can see how we get better and better with each gradient

    for j in range(len(weights)):

        ys_predictes = []
        for ytrue in range(2):
            
            for data in image_labels_training_data[ytrue]: 
                x = image_labels_training_data[ytrue][data]
                z = sum(weights*x)[0]+b
                
                ypredicted = 1/(1+math.e**(-z))
                
                ys_predictes.append((ypredicted-ytrue)*x[j,0])
                #print(x[j,0])
        
        gradient = sum(ys_predictes)/len(ys_predictes)
        weights[j,0] = weights[j,0] - learningrate*gradient

    ys_predictes = []
    for ytrue in range(2):
        
        for data in image_labels_training_data[ytrue]: 
            x = image_labels_training_data[ytrue][data]
            z = sum(weights*x)[0]+b
            ypredicted = 1/(1+math.e**(-z))
            ys_predictes.append((ypredicted-ytrue))
        
    gradient = sum(ys_predictes)/len(ys_predictes)
    
    b = b - learningrate*gradient

#print(weights)
#print(b)

# so let's test it!
# lets add an example number that is exactly in the data set. We should guess it right.

# we should output the percentage that we are sure. So we output 2 prob, 1 that we are sure if its 1 percent, and the other the opposite for 0 percent

sampletest = image_labels_training_data[1]['0'] # so this is just like the first image we fed to the model to train on, we said it was 0 so
z = sum(weights*sampletest)[0]+b
hypothesisfunctionresult = 1/(1+math.e**(-z))

print("Just rerunning witht the sample data to make sure we did a good job!")
print("possibility that its one: ")
print(hypothesisfunctionresult)
print("possibility that its 0: ")
print(1-hypothesisfunctionresult)

print("Actual Test!!")
# lets do a quick test!
# Read the image in grayscale mode (0-255 values)
image = cv2.imread("TestData\\test0.png", cv2.IMREAD_GRAYSCALE)

# Convert to binary: 255 (white) -> 1, 0 (black) remains 0
binary_image = (image == 255).astype(np.uint8)  # Convert True/False to 1/0

# Reshape into a column vector (1D but vertical)
binary_image = binary_image.reshape(-1, 1)

z = sum(weights*binary_image)[0]+b
hypothesisfunctionresult = 1/(1+math.e**(-z))

print("possibility that its one: ")
print(hypothesisfunctionresult)
print("possibility that its 0: ")
print(1-hypothesisfunctionresult)


import matplotlib.pyplot as plt

# Assuming 'weights' is your learned weights vector of shape (64, 1)
weights = weights.reshape(8, 8)  # Reshape to 8x8 grid

# Normalize the weights for better visualization (optional)
weights_normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

# Plot the weights using the 'coolwarm' colormap
plt.figure(figsize=(6, 6))
plt.imshow(weights, cmap='coolwarm', vmin=-np.max(np.abs(weights)), vmax=np.max(np.abs(weights)))
plt.colorbar(label='Weight Value')
plt.title("Learned Weights (Coolwarm Colormap)")
plt.show()

print("Here are the updated weights for reference!")
print(weights)
print(b)