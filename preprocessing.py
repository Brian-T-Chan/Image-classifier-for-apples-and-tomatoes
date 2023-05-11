
# Compress the images in the data folder.

import cv2
import os

from compress import image_compress
from tqdm import tqdm


# Get the file paths of all the images.

cwd = os.getcwd()

dataset = os.path.join(cwd, 'data')
apples = os.path.join(dataset, 'apples')
tomatoes = os.path.join(dataset, 'tomatoes')
paths = [apples, tomatoes]
folders = [os.listdir(apples), os.listdir(tomatoes)]


# Get the file paths for the folders that will contain
# the compressed images.

cdataset = os.path.join(cwd, 'compressed data')
capples = os.path.join(cdataset, 'compressed apples')
ctomatoes = os.path.join(cdataset, 'compressed tomatoes')
cpaths = [capples, ctomatoes]


# Set the width of every compressed image 
# to be 150 pixels by 150 pixels.

new_width = 150


# Get the images, use the image_compress function to compress
# them, then save those compressed images.

for i in range(2):
    for file in tqdm(folders[i]):
        image = cv2.imread(os.path.join(paths[i], file))
        compressed = image_compress(image, new_width)
        cv2.imwrite(os.path.join(cpaths[i], file), compressed)


