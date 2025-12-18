#%%
import numpy as np
import tqdm as tqdm
import matplotlib as mpl
import utils
from auto_encoder import AutoEncoder as AE

# LOAD DATA
train_data, train_labels = utils.get_dataset("data/mnist_train.csv")
test_data, test_labels = utils.get_dataset("data/mnist_test.csv")


# CREATE AE INSTANCE
model = AE(784, 500, 0.005) # Note: mu = 0.5 was WAY too high, init settings: 500, 0.01

# RUN TRAIN
model.train(train_data)

# %%
# RUN TEST
test_pred = model.forward(test_data)
print(f"Loss for one epoch of test data prediction: ", model.loss(test_pred, test_data))

#%%
# VISUALIZE
example_img = test_data[0]
input_batch = example_img.reshape(1, -1)

batch_pred = model.forward(input_batch)
reconstruction_vector = batch_pred.flatten()

print("Original image:")
utils.plot_image(example_img)

print("Reconstructed image:")
utils.plot_image(reconstruction_vector)
# %%
