import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import torch
import os

class Image_recognition(nn.Module):

     #Generating the architecture of the network

     def __init__(self, in_features = 62500, h1 = 512, h2 = 512, out_features = 5):
          super().__init__()
          self.fc1 = nn.Linear(in_features, h1)
          self.fc2 = nn.Linear(h1, h2)
          self.out = nn.Linear(h2, out_features)    

     #Move forward in the network
          
     def forward(self, input_data):
          data = F.relu(self.fc1(input_data))
          data = F.relu(self.fc2(data))
          output_data = self.out(data)
          return output_data

#class Model:
   #  def predict(file_path):
        #  X = bitmap(file_path)
     
#Convert image to an array of neurons
     
def bitmap(image_path):
    #Open image
    rgb = Image.open(image_path, mode = "r")
    #Converting to grayscale
    grayscale = rgb.convert(mode = "L")

    b_map = np.asarray(grayscale)
    #Reshaping array to 1D and normalicing it
    initial_neurons = np.reshape(b_map, len(b_map)*len(b_map[0])) / 255

    return initial_neurons

#Generate train data from one random image from each folder

def get_train_data(): 
     path = './Rice_Image_Dataset/Train'
     folders = os.listdir(path)
     i = 0

     y_train = np.zeros(5)
     x_train = np.zeros((5,62500))
     while i < 5:
          folder = folders[i]
          random_image = np.random.choice(os.listdir(path+'/'+folder))
          try:
               x_train[i] = bitmap(path+'/'+folder+'/'+random_image)
          except PIL.UnidentifiedImageError:
               continue
          y_train[i] = i
          i+=1
     return x_train,y_train

def gradient_descent(epochs = 1000):
     
     model = Image_recognition()

     #Cost function

     criterion = nn.CrossEntropyLoss()

     #Creating the gradient descent

     optimizer = SGD(model.parameters(), lr = 0.01)


     #x_train,y_train = get_train_data()

     losses = []
     for i in range(epochs):

          x_train, y_train = get_train_data()

          #Converting arrays into torch tensors

          x_train = torch.FloatTensor(x_train)
          y_train = torch.LongTensor(y_train)

          #Moving forward in the network

          y_pred = model.forward(x_train)
          loss = criterion(y_pred, y_train)
          losses.append(loss.detach().numpy())

          #Gradient descent

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if loss < 0.01:
               break

     return model

model = gradient_descent(epochs = 1000)
X = bitmap('/home/cod3_breaker/portafolio/Arroz/Rice_Image_Dataset/Test/Karacadag (11990).jpg')
X = torch.FloatTensor(X)
print(model.forward(X).detach().numpy())
