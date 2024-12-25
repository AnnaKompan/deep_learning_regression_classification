# train.py
import numpy as np
from layers import Linear, ReLU, Sigmoid
from losses import MSE, BinaryCrossEntropy
from model import Model
from utils import split_batch, train_step, test_step

if __name__ == "__main__":
    # Define your model architecture
    model = Model([
        Linear(2, 4),   # Example: Linear layer with 2 inputs, 4 outputs
        ReLU(),
        Linear(4, 1),   # Example: Linear layer with 4 inputs, 1 output
        Sigmoid()
    ])

    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR input
    target = np.array([[0], [1], [1], [0]])            # XOR output

    # Set loss function and learning rate
    loss_fn = BinaryCrossEntropy()
    learning_rate = 0.01

    # Training loop
    for epoch in range(1000):  # Train for 1000 epochs
        loss = 0
        for batch_data, batch_target in split_batch(data, target, batch_size=2):
            loss += train_step(batch_data, batch_target, model, loss_fn, learning_rate)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
