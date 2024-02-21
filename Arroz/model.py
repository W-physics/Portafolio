import numpy as np
from PIL import Image
imp ort os

class Model:
     def train():
          pass
     def predict(filepath):
          pass

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

def random_weights(initial_neurons, horizontal_depht, vertical_depht, final_depht = 5):
     
     neurons = initial_neurons
     weights = np.random.rand((0,vertical_depht,len(neurons)))
     bias = np.random.rand((0,vertical_depht))

     for i in range(0,len(horizontal_depht))

          nn_neurons = weights[i] @ neurons + bias[i]
          neurons = nn_neurons / np.sum(nn_neurons)

          if i != horizontal_depht:
               np.append(weights, np.random.rand((vertical_depht,vertical_depht)))
               np.append(bias, np.random.rand((vertical_depht)))
          else:
               np.append(weights, np.random.rand((final_depht,vertical_depht)))
               np.append(bias, np.random.rand(final_depht))

     return (weights, bias)