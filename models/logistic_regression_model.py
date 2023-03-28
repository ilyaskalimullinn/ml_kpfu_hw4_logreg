import numpy as np
from easydict import EasyDict
from typing import Union

from datasets.base_dataset_classes import BaseClassificationDataset
from utils.metrics import accuracy, confusion_matrix


class LogReg:

    def __init__(self, cfg: EasyDict, number_classes: int, input_vector_dimension: int, reg_coeff: float = 0):
        self.k = number_classes
        self.d = input_vector_dimension
        self.cfg = cfg
        getattr(self, f'weights_init_{cfg.weights_init_type.name}')(**cfg.weights_init_kwargs)

        self.accuracy_train = []
        self.accuracy_valid = []
        self.target_func_values = []
        self.reg_coeff = 0

    def weights_init_normal(self, sigma, *args, **kwargs):
        # init weights with values from normal distribution
        self.W = np.random.normal(0, sigma, size=(self.k, self.d))
        self.bias = np.random.normal(0, sigma, size=(self.k, 1))

    def weights_init_uniform(self, epsilon, *args, **kwargs):
        # init weights with values from uniform distribution BONUS TASK
        self.W = np.random.uniform(-epsilon, epsilon, size=(self.k, self.d))
        self.bias = np.random.uniform(-epsilon, epsilon, size=(self.k, 1))

    def weights_init_xavier(self, n_in, n_out):
        # TODO Xavier weights initialisation BONUS TASK
        pass

    def weights_init_he(self, n_in):
        # TODO He weights initialisation BONUS TASK
        pass

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        # softmax function realisation
        # subtract max value of the model_output for numerical stability
        y = model_output - np.max(model_output)
        y = np.exp(y)
        y = y / y.sum(axis=1).reshape(-1, 1)
        return y

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        # calculate model confidence (y in lecture)
        z = self.__get_model_output(inputs)
        y = self.__softmax(z)
        return y

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        # calculate model output (z in lecture) using matrix multiplication DONT USE LOOPS
        return inputs @ self.W.T + self.bias.T

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        #  calculate gradient for w
        #  slide 10 in presentation
        return (model_confidence - targets).T @ inputs + self.reg_coeff * self.W

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        #  calculate gradient for b
        #  slide 10 in presentation
        return (model_confidence - targets).sum(axis=0).reshape(-1, 1)

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        #  update model weights
        #  slide 8, item 2 in presentation for updating weights
        self.W = self.W - self.cfg.gamma * self.__get_gradient_w(inputs, targets, model_confidence)
        self.bias = self.bias - self.cfg.gamma * self.__get_gradient_b(targets, model_confidence)

    def __gradient_descent_step(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                targets_train_encoded: np.ndarray,
                                epoch: int, inputs_valid: Union[np.ndarray, None] = None,
                                targets_valid: Union[np.ndarray, None] = None,
                                targets_valid_encoded: Union[np.ndarray, None] = None):
        #  one step in Gradient descent:
        #  calculate model confidence;
        #  target function value calculation;
        #  update weights
        #  calculate accuracy and confusion matrix on train and valid sets
        #  save calculated metrics
        """
        :param targets_train: onehot-encoding
        :param epoch: number of loop iteration
        """

        model_confidence_train = self.get_model_confidence(inputs_train)

        target_func_value = self.__target_function_value(inputs_train, targets_train_encoded,
                                                         model_confidence_train)
        self.target_func_values.append(target_func_value)

        accuracy_train, confusion_matrix_train = self.__validate(inputs_train, targets_train, model_confidence_train)
        self.accuracy_train.append(accuracy_train)

        accuracy_valid, confusion_matrix_valid = None, None
        if inputs_valid is not None:
            accuracy_valid, confusion_matrix_valid = self.__validate(inputs_valid, targets_valid,
                                                                 self.get_model_confidence(inputs_valid))
            self.accuracy_valid.append(accuracy_valid)

        self.__log_metrics(target_func_value, accuracy_train, confusion_matrix_train, accuracy_valid, confusion_matrix_valid)

        self.__weights_update(inputs_train, targets_train_encoded, model_confidence_train)

    def gradient_descent_epoch(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               targets_train_encoded: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None,
                               targets_valid_encoded: Union[np.ndarray, None] = None):
        # loop stopping criteria - number of iterations of gradient_descent
        for epoch in range(self.cfg.nb_epoch):
            self.__gradient_descent_step(inputs_train, targets_train, targets_train_encoded, epoch, inputs_valid,
                                         targets_valid, targets_valid_encoded)

    def gradient_descent_gradient_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                       inputs_valid: Union[np.ndarray, None] = None,
                                       targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with gradient norm stopping criteria BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def gradient_descent_difference_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               targets_train_encoded: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None,
                               targets_valid_encoded: Union[np.ndarray, None] = None):
        # gradient_descent with stopping criteria - norm of difference between ￼w_k-1 and w_k;￼BONUS TASK
        W_old = np.zeros_like(self.W)

        epoch = 1
        while np.linalg.norm(self.W - W_old) > self.cfg.min_difference_norm:
            W_old = self.W
            self.__gradient_descent_step(inputs_train, targets_train, targets_train_encoded, epoch, inputs_valid,
                                         targets_valid, targets_valid_encoded)
            epoch += 1

    def gradient_descent_metric_value(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               targets_train_encoded: np.ndarray,
                               inputs_valid: np.ndarray,
                               targets_valid: np.ndarray,
                               targets_valid_encoded: np.ndarray):
        #  gradient_descent with stopping criteria - metric (accuracy, f1 score or other) value on validation set is not growing;￼
        #  BONUS TASK
        predictions = np.argmax(self.get_model_confidence(inputs_valid), axis=1)

        # first make it learn itself
        epoch = 1
        while epoch < 10:
            self.__gradient_descent_step(inputs_train, targets_train, targets_train_encoded, epoch, inputs_valid,
                                         targets_valid, targets_valid_encoded)
            epoch += 1

        # now learn until accuracy falls
        accuracy_old = 0
        accuracy_new = accuracy(predictions, targets_valid)
        while accuracy_new - accuracy_old > -1e-8:
            self.__gradient_descent_step(inputs_train, targets_train, targets_train_encoded, epoch, inputs_valid,
                                         targets_valid, targets_valid_encoded)
            predictions = np.argmax(self.get_model_confidence(inputs_valid), axis=1)
            accuracy_old = accuracy_new
            accuracy_new = accuracy(predictions, targets_valid)
            epoch += 1

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray,
              inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None):
        targets_train_encoded = BaseClassificationDataset.onehotencoding(targets_train, self.k)
        targets_valid_encoded = None
        if targets_valid is not None:
            targets_valid_encoded = BaseClassificationDataset.onehotencoding(targets_valid, self.k)
        getattr(self, f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')(inputs_train, targets_train,
                                                                                targets_train_encoded,
                                                                                inputs_valid,
                                                                                targets_valid,
                                                                                targets_valid_encoded)

    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                model_confidence: Union[np.ndarray, None] = None) -> float:
        #  target function value calculation
        #  use formula from slide 6 for computational stability
        #  could add more optimization with usage of model output
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        return -np.log(model_confidence[targets.astype(bool)]).sum()

    def __validate(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: Union[np.ndarray, None] = None):
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        model_predictions = np.argmax(model_confidence, axis=1)
        acc = accuracy(model_predictions, targets)
        matrix = confusion_matrix(model_predictions, targets, self.k)
        return acc, matrix

    def __log_metrics(self, target_func_value, accuracy_train, confusion_matrix_train, accuracy_valid, confusion_matrix_valid):
        print("*" * 50)

        print(f"Target func value on train set: {target_func_value}, train accuracy: {accuracy_train}")
        print("Confusion matrix on train set:")
        print(confusion_matrix_train)

        if accuracy_valid is not None:
            print(f"Validation accuracy: {accuracy_valid}")

        if confusion_matrix_valid is not None:
            print("Confusion matrix on validation set: ")
            print(confusion_matrix_valid)

        print("*" * 50)

    def __call__(self, inputs: np.ndarray):
        model_confidence = self.get_model_confidence(inputs.reshape(-1, self.d))
        predictions = np.argmax(model_confidence, axis=1)
        return predictions
