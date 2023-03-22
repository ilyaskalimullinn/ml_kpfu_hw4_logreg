from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import train_test_split


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    @property
    @abstractmethod
    def d(self):
        # inputs variables
        pass

    def divide_into_sets(self):
        #  define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test; you can use your code from previous homework
        self.inputs_train, self.inputs_test, self.targets_train, self.targets_test = train_test_split(self.inputs,
                                                                                                      self.targets,
                                                                                                      train_size=self.train_set_percent)

        test_size = 1 - self.train_set_percent - self.valid_set_percent
        test_size = test_size / (test_size + self.valid_set_percent)

        self.inputs_valid, self.inputs_test, self.targets_valid, self.targets_test = train_test_split(self.inputs_test,
                                                                                                      self.targets_test,
                                                                                                      test_size=test_size)

    def normalization(self):
        # TODO write normalization method BONUS TASK
        pass

    def get_data_stats(self):
        # TODO calculate mean and std of inputs vectors of training set by each dimension
        pass

    def standartization(self):
        # TODO write standardization method (use stats from __get_data_stats)
        #   DON'T USE LOOP
        pass


class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        # TODO create matrix of onehot encoding vactors for input targets
        #  it is possible to do it without loop
        pass
