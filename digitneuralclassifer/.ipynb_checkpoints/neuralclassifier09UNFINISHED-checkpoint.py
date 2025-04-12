import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Load the digits dataset
digits = datasets.load_digits()

digits.images

# Display the last digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()



# import cv2
# import numpy as np
# import glob
# import os
# import math
# import tqdm

# def compute_loss(image_labels_training_data, weights, b):
#     loss = 0
#     for digit in range(10):
#         for data in image_labels_training_data[digit]:
#             x = image_labels_training_data[digit][data]
#             z = sum(weights[:, digit].reshape(-1, 1)*x)[0] + b[digit]
#             y_pred = 1 / (1 + np.exp(-z))
#             loss += -np.log(y_pred)  # Binary cross-entropy for positive class
#         for other_digit in range(10):
#             if other_digit == digit:
#                 continue
#             for data in image_labels_training_data[other_digit]:
#                 x = image_labels_training_data[other_digit][data]
#                 z = sum(weights[:, digit].reshape(-1, 1)*x)[0] + b[digit]
#                 y_pred = 1 / (1 + np.exp(-z))
#                 loss += -np.log(1 - y_pred)  # Binary cross-entropy for negative class
#     return loss / len(image_labels_training_data)

# # lets read our data!
# def train_model():

#     # Folder containing PNG images
#     folder_path = "TrainingData09"

#     image_labels_training_data = []
#     for digit in range(10):

#         # Dictionary to store images as numpy arrays
#         image_arrays = {}

#         # Get all matching PNG files
#         image_files = glob.glob(os.path.join(folder_path, f"t{digit}_*.png"))

#         # Loop through each image
#         for image_file in image_files:
#             # Extract the number from the filename
#             filename = os.path.basename(image_file)  # Get "t1_numberhere.png"
#             number = filename.split("_")[1].split(".")[0]  # Extract "numberhere"

#             # Read the image in grayscale mode (0-255 values)
#             image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

#             # Convert to binary: 255 (white) -> 1, 0 (black) remains 0
#             binary_image = (image == 255).astype(np.uint8)  # Convert True/False to 1/0

#             # Reshape into a column vector (1D but vertical)
#             binary_image = binary_image.reshape(-1, 1)
#             # Store in dictionary
#             image_arrays[number] = binary_image

#             image_labels_training_data.append(image_arrays)


#     #print(image_labels_training_data) # The index is where we are going to bet at the true label basically


#     # Awesome!! so now that we can read our data! Now that we have that, we need to kinda create a linear combination of that data.
#     # Training happens here

#     image_width = 8
#     image_height = 8
#     learningrate = 0.05
#     number_of_digits = 10
#     weights = np.random.randn(image_width * image_height, number_of_digits)
#     b = np.random.randn(number_of_digits) # the b term
#     number_of_iterations = 3000

#     for iteration in tqdm.tqdm(range(number_of_iterations)):

#         # loss function i still need to add it here for monitoring
#         # lets add the loss function and add each result of it to a list so that we can then plot it so we can see how we get better and better with each gradient
#         if iteration%100==0:
#             loss = compute_loss(image_labels_training_data, weights, b)
#             print("Current Loss: "+str(loss))
#         # So before i do anything... i should check my loss function and check if the model is in fact improving at all during each iteration, if it's not in any way we are messing up here. CHanging the data, or anytthing else might not be good enough

#         for num in range(number_of_digits): # cuz there is 10 digits, thus 10 sets of weights we have to do calculations for

#             for j in range(weights.shape[0]):

#                 ys_predictes = []
                

#                 for data in image_labels_training_data[num]: # this is the one that we have chosen to train for (we are selecting hey this is our digit so rn we are looking for example at digit 0, which means y true here has to be 1), 
                    
#                     x = image_labels_training_data[num][data]
#                     z = sum(weights[:, num].reshape(-1, 1)*x)[0]+b[num]
                    
#                     ypredicted = 1/(1+math.e**(-z))
                    
#                     ys_predictes.append((1-ypredicted)*x[j,0])
#                     #print(x[j,0])
                
#                 # now though, we still have to add the rest! Which should be all zero! We should traverse basically any other values in the data but that are not the current digit and set all those to 0 as ytrue
#                 for rest in range(number_of_digits):
#                     if rest==num:
#                         continue # because we already added that one anyways as 1
                    
#                     for data in image_labels_training_data[rest]:
#                         x = image_labels_training_data[rest][data]
#                         z = sum(weights[:, num].reshape(-1, 1)*x)[0]+b[num]
                        
#                         ypredicted = 1/(1+math.e**(-z))
                        
#                         ys_predictes.append((0-ypredicted)*x[j,0])

#                 # Now we have all we need
#                 gradient = sum(ys_predictes)/len(ys_predictes)
#                 weights[j,num] = weights[j,num] - learningrate*gradient

#             ys_predictes = []
#             # Now do the same for b!
#             for data in image_labels_training_data[num]: 
#                 x = image_labels_training_data[num][data]
#                 z = sum(weights[:, num].reshape(-1, 1)*x)[0]+b[num]
#                 ypredicted = 1/(1+math.e**(-z))
#                 ys_predictes.append(1-ypredicted)

#             # and now again we need to do the rest with just zeros
#             for rest in range(number_of_digits):
#                 if rest==num:
#                     continue
#                 for data in image_labels_training_data[rest]:
#                     x = image_labels_training_data[rest][data]
#                     z = sum(weights[:, num].reshape(-1, 1)*x)[0]+b[num]
                    
#                     ypredicted = 1/(1+math.e**(-z))
                    
#                     ys_predictes.append(0-ypredicted)
                
#             gradient = sum(ys_predictes)/len(ys_predictes)
            
#             b[num] = b[num] - learningrate*gradient

#     #print(weights)
#     #print(b)

#     # now that it's all done, we need to save these weights and biases so we dont have to calulcate it again
#     np.savez('neuralclassifier09modelparams.npz', weights=weights, biases=b)
#     return "SUCCESS"

# # so let's test it!
# def run_model(imagelocation):
#     # make sure the image is the right size! 8x8
#     image = cv2.imread(imagelocation, cv2.IMREAD_GRAYSCALE)

#     # Convert to binary: 255 (white) -> 1, 0 (black) remains 0
#     binary_image = (image == 255).astype(np.uint8)  # Convert True/False to 1/0

#     # Reshape into a column vector (1D but vertical)
#     binary_image = binary_image.reshape(-1, 1)
#     # now we have to run all the weights and pick the most probable one:

#     hypothesisfunctionresult = []
#     for classifer in range(10):
#         z = sum(weights[:, classifer]*binary_image)[0]+biases[classifer]
#         hypothesisfunctionresult.append(1/(1+math.e**(-z)))

#     print("Full list of digit probabilities:")
#     print(hypothesisfunctionresult )
#     digitguess = hypothesisfunctionresult.index(max(hypothesisfunctionresult))
#     return digitguess, max(hypothesisfunctionresult)

# print("Welcome to neural classifier 0-9, where we look at small images and check what's their digit using a very simple neural network, please select an option below:")
# print("1: I already trained the model, and now I want to run it! The image to test is in TestData/ and its called digit.png and its 8x8 pixels with black background and white pixels drawn for the number")
# print("2: I want to train the model on the dataset given in TrainingData09/ and the files are all correctly named for successful reading")
# print("3: Give me a visual representation, a heatmap, of pixels of what the model is seeing.")
# print("4: Check the model accuracy by running it on the training data again! Ez check just to see if it's doing a somewhat ok job of labeling the data it was originally trained on.")
# selectedoption = input("Type the option number you want to select and hit Enter (if unsure hit number 2 as this trains from scratch): ")

# if selectedoption=="1":
#     # that means we can read the weights and biases to run!
#  # we are ready to read the weights and biases
#         # Load the .npz file
#     data = np.load('neuralclassifier09modelparams.npz')

#     # Access the arrays by their names
#     weights = data['weights']
#     biases = data['biases']

#     digitguess, confidence = run_model("TestData\\digit.png")

#     print("The Model believes the most likely digit is: "+str(digitguess))
#     print("Confidence: "+ str(confidence))

# elif selectedoption=="2":
#     # that means we have to train from scratch
#     print("Training in progress...")
#     returncode = train_model()
#     print("Training done successfully!")
#     if returncode=="SUCCESS":
#         # we are ready to read the weights and biases
#         # Load the .npz file
#         data = np.load('neuralclassifier09modelparams.npz')

#         # Access the arrays by their names
#         weights = data['weights']
#         biases = data['biases']

#         print(weights)
#         print(biases)

# elif selectedoption=="3":

#     import matplotlib.pyplot as plt

#     data = np.load('neuralclassifier09modelparams.npz')

#     # Access the arrays by their names
#     weights = data['weights']
#     biases = data['biases']

#     for i in range(10):

#         # HERE I NEED U TO DISPLAY 10 different heat maps for eahc classifier plz

#         # Assuming 'weights' is your learned weights vector of shape (64, 1)
#         weights_to_dispplay = weights[:, i].reshape(8, 8)  # Reshape to 8x8 grid

#         # Normalize the weights for better visualization (optional)
#         weights_normalized = (weights_to_dispplay - np.min(weights_to_dispplay)) / (np.max(weights_to_dispplay) - np.min(weights_to_dispplay))

#         # Plot the weights using the 'coolwarm' colormap
#         plt.figure(figsize=(6, 6))
#         plt.imshow(weights_to_dispplay, cmap='coolwarm', vmin=-np.max(np.abs(weights_to_dispplay)), vmax=np.max(np.abs(weights_to_dispplay)))
#         plt.colorbar(label='Weight Value')
#         plt.title(f"Learned Weights (Coolwarm Colormap) for digit: {i}")
#         plt.show()

# elif selectedoption=="4":
#     data = np.load('neuralclassifier09modelparams.npz')

#     # Access the arrays by their names
#     weights = data['weights']
#     biases = data['biases']
#     import os

#     folder_path = "TrainingData09"

#     alltests = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".png"):
#             alltests.append(os.path.join(folder_path, filename))
    
#     number_correct = 0
#     for test in alltests:
#         digitguess, confidence = run_model(test)
#         if "t0" in test and digitguess==0:
#             number_correct+=1
#         if "t1" in test and digitguess==1:
#             number_correct+=1
#         if "t2" in test and digitguess==2:
#             number_correct+=1
#         if "t3" in test and digitguess==3:
#             number_correct+=1
#         if "t4" in test and digitguess==4:
#             number_correct+=1
#         if "t5" in test and digitguess==5:
#             number_correct+=1
#         if "t6" in test and digitguess==6:
#             number_correct+=1
#         if "t7" in test and digitguess==7:
#             number_correct+=1
#         if "t8" in test and digitguess==8:
#             number_correct+=1
#         if "t9" in test and digitguess==9:
#             number_correct+=1
        
#     print("Number of correct guesses / total guesses:")
#     print(number_correct/len(alltests))