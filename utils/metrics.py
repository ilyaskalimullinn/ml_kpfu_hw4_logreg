import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    return ((predictions - targets) ** 2).mean()


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return (predictions == targets).mean()

def confusion_matrix(*args, **kwargs):
    # TODO build confusion matrix
    pass
