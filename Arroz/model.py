import numpy as np
from PIL import Image

#Convert image to an array of neurons
def bitmap(image_path):

    #Open image
    rgb = Image.open(image_path, mode = "r")
    #Converting to grayscale
    grayscale = rgb.convert(mode = "L")

    b_map = np.asarray(grayscale)
    #Reshaping array to 1D and normalicing it
    rb_map = np.reshape(b_map, (1,len(b_map)*len(b_map[0]))) / 255

    return rb_map


    
#    grayscale.show()

try:
     print(bitmap("/home/cod3_breaker/portafolio/Arroz/Rice_Image_Dataset/Train/Arborio/Arborio (1).jpg"))
except FileNotFoundError:
     print("No se encontró el archivo")

