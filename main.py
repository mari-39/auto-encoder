import numpy as np
import tqdm as tqdm
import matplotlib as mpl
import utils
from auto_encoder import AutoEncoder as AE

# LOAD DATA
train_data, train_labels = utils.get_dataset("data/mnist_train.csv")
test_data, test_labels = utils.get_dataset("data/mnist_test.csv")

# CREATE AE INSTANCE
model = AE(784, 500, 0.01) # Note: mu = 0.5 was WAY too high, init settings: 500, 0.01

# RUN TRAIN
model.train(train_data)

# VISUALIZE
example_img = test_data[0] # training img
utils.plot_image(example_img) # initial image
utils.plot_image(model.forward(example_img)) # encoded-decoded image

