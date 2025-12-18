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

#%%
# VISUALIZE
# 1. Pick a single test image (Shape is likely (784,))
example_img = test_data[1]

# 2. Add a batch dimension to make it (1, 784) for the model
input_batch = example_img.reshape(1, -1)

# 3. Get the reconstruction (Shape will be (1, 784))
reconstructed_batch = model.forward(input_batch)

# 4. CRITICAL STEP: Convert back to 1D (784,) for the assignment's function
# .flatten() or [0] removes the batch dimension so the square root math works
reconstruction_vector = reconstructed_batch.flatten()

# 5. Render
print("Original:")
utils.plot_image(example_img)

print("Reconstructed:")
utils.plot_image(reconstruction_vector)




# %%
