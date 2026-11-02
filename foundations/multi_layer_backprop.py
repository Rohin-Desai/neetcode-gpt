import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)


        l1 = np.dot(x, W1.T) + b1

        l2 = np.maximum(0, l1)

        l3 = np.dot(l2, W2.T) + b2

        y_hat = l3

        loss = np.mean((y_hat - y_true)**2)

        dict1 = {}

        dL_dy_hat = 2/len(y_true) * (y_hat - y_true)

        dW2 = np.outer(dL_dy_hat, l2)

        db2 = dL_dy_hat

        next1 = dL_dy_hat @ W2

        next2 = next1 * (l1 > 0)

        dW1 = np.outer(next2, x)

        dB1 = next2

        dict1["loss"] = round(float(loss), 4)
        dict1["dW1"] = np.round(dW1, 4).tolist()
        dict1["db1"] = np.round(dB1, 4).tolist()
        dict1["dW2"] = np.round(dW2, 4).tolist()
        dict1["db2"] = np.round(db2, 4).tolist()

        return dict1

       



