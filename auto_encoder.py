from typing import List
import numpy as np
from tqdm import tqdm
import utils


class AutoEncoder:
    """
    Represents a 4-layers auto-encoder.
    """

    def __init__(self, input_dim, encoded_dim, learning_rate):
        """
        Initialisation function.
        Parameters:
            - input_dim (int): the shape of the input (flattened)
            - hidden_dim (int): the shape of the encoded vector
            - learning_rate (float): the learning rate factor
        """
        self.mu = learning_rate
        self.W1 = (np.random.random((input_dim, input_dim // 2)) - 0.5) * 0.001
        self.W2 = (np.random.random((input_dim // 2, encoded_dim)) - 0.5) * 0.001
        self.W3 = (np.random.random((encoded_dim, input_dim // 2)) - 0.5) * 0.001
        self.W4 = (np.random.random((input_dim // 2, input_dim)) - 0.5) * 0.001

    def loss(self, x: np.ndarray, y: np.ndarray) -> float: 
        """MSE(𝑥, 𝑦) = 1/𝑛 Σ (𝑥 − 𝑦)^2"""
        n = x.size
        return 1/n * np.sum((x - y) ** 2)

# x = input vec => uncompressed
# x1 = downsized vec first stage => extraction of broader ft
# x_hat = encoded vec => fully downsized
# x2 = upsized vector first stage => starting to re-draw
# y = decoded vec => full reconstruction

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encodes an input vector x.
        """
        self.downsized = utils.activation(x @ self.W1)
        self.encoded = utils.activation(self.downsized @ self.W2)
        return np.asarray(self.encoded)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Decodes an encoded vector x.
        """
        self.upsized = utils.activation(x @ self.W3)
        self.decoded = utils.activation(self.upsized @ self.W4)
        return np.asarray(self.decoded)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass.
        """
        encoded_vec = self.encode(x)
        decoded_vec = self.decode(encoded_vec)
        return decoded_vec

    def backward(self, x: np.ndarray) -> None:
        """
        Updates the weights of the network using the backpropagation.
        """
        e4 = 2 * (self.decoded - x)
        d4 = e4 * utils.derivative(self.decoded)

        e3 = d4 @ np.transpose(self.W4)
        d3 = e3 * utils.derivative(self.upsized)

        e2 = d3 @ np.transpose(self.W3)
        d2 = e2 * utils.derivative(self.encoded)

        e1 = d2 @ np.transpose(self.W2)
        d1 = e1 * utils.derivative(self.downsized)

        self.W4 -= self.mu * (np.transpose(self.upsized) @ d4)
        self.W3 -= self.mu * (np.transpose(self.encoded) @ d3)
        self.W2 -= self.mu * (np.transpose(self.downsized) @ d2)
        self.W1 -= self.mu * (np.transpose(x) @ d1)

    def train(self, x_train: np.ndarray, epochs: int = 10, batch_size: int = 16) -> List[float]:
        """
        Trains the auto-encoder on the given dataset.
        Parameters:
            - x_train (np.ndarray): the dataset containing the input vectors.
            - epochs (int): the number of epochs to train the auto-encoder with.
            - batch_size (int): the size of each training batch.
        Returns:
            - losses (List[int]): the training loss of each epoch.
        """
        losses = []
        for epoch in range(epochs): # means one full pass through data set
            for i in tqdm(range(0, x_train.shape[0], batch_size)): # do this in batches => SLICE OF WHOLE MATRIX
                self.forward(x_train[i : i + batch_size]) # predict slice
                self.backward(x_train[i : i + batch_size]) # update (all) weights using that slice

            output = self.forward(x_train) # PREDICTION of whole set after having LEARNED a lot
            loss = self.loss(output, x_train) # How good was the prediction compared to the label (x_train) ?
            losses.append(loss) # add most recent MSE of this epoch to list of losses
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.7f}")
        return losses
