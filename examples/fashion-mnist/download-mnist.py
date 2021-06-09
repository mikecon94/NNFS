import os
import urllib
import urllib.request
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt, cv2

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)
    print('Unzipping images...')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)
    print('Fashion MNIST Dataset Saved!')

labels = os.listdir(FOLDER + '/train')
labels.sort()
print(labels)
files = os.listdir(FOLDER + '/train/0')
files.sort()
print(files[:10])
print(len(files))

# Second argument says to read in these images in the same format they were saved.
# OpenCV will convert to use all 3 colour channels otherwise.
image_data = cv2.imread(FOLDER + '/train/4/0011.png', cv2.IMREAD_UNCHANGED)
np.set_printoptions(linewidth=200)
plt.imshow(image_data, cmap='gray')
# plt.show()
