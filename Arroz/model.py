import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import os

class Model:
     def predict():
          pass

class Image_recognition(nn.Module):
     #Generating the architecture of the network
     def __init__(self, in_features = 28*28, h1 = 512, h2 = 512, h3 = 512, out_features = 5):
          super().__init__()
          self.fc1 = nn.Linear(in_features, h1)
          self.fc2 = nn.Linear(h1, h2)
          self.fc3 = nn.Linear(h2, h3)
          self.out = nn.Linear(h3, out_features)    
     #Move forward in the network
     def forward(self, input_data):
          data = F.relu(self.fc1(input_data))
          data = F.relu(self.fc2(data))
          data = F.relu(self.fc3(data))
          output_data = self.out(data)
          return output_data

#Convert image to an array of neurons
     
def bitmap(image_path):
    #Open image
    rgb = Image.open(image_path, mode = "r")
    #Converting to grayscale
    grayscale = rgb.convert(mode = "L")

    b_map = np.asarray(grayscale)
    #Reshaping array to 1D and normalicing it
    initial_neurons = np.reshape(b_map, (1,len(b_map)*len(b_map[0]))) / 255

    return initial_neurons

model = Image_recognition()