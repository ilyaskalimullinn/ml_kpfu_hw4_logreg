import os
import pickle

import numpy as np

from config.logistic_regression_config import cfg
from datasets.digits_dataset import Digits
from models.logistic_regression_model import LogReg
from utils.metrics import accuracy, confusion_matrix
from utils.visualisation import Visualisation


def main_task(logreg: LogReg, digits: Digits):
    predictions = logreg(digits.inputs_test)

    test_accuracy = accuracy(predictions, digits.targets_test)
    test_confusion_matrix = confusion_matrix(predictions, digits.targets_test, digits.k)

    print(f"Test accuracy: {test_accuracy}, confusion matrix: ")
    print(test_confusion_matrix)

    visualisation = Visualisation()
    visualisation.plot_accuracy(np.array(logreg.accuracy_valid), np.array(logreg.accuracy_train), cfg.nb_epoch,
                                f"Accuracy on train set and validation set; testing accuracy is {round(test_accuracy, 2)}",
                                save_path=os.path.join(ROOT_DIR, 'graphs/accuracy.html'))
    visualisation.plot_target_function(np.array(logreg.target_func_values), cfg.nb_epoch,
                                       save_path=os.path.join(ROOT_DIR, 'graphs/target_function.html'))

    DIGITS_AMOUNT = 3

    model_confidence = logreg.get_model_confidence(digits.inputs_test)
    model_confidence_idx = model_confidence.max(axis=1).argsort()[::-1]

    model_confidence_sorted = model_confidence[model_confidence_idx]
    predictions_sorted = predictions[model_confidence_idx]
    targets_sorted = digits.targets_test[model_confidence_idx]
    inputs_sorted = digits.inputs_test[model_confidence_idx]

    predictions_matched_mask = (predictions_sorted == targets_sorted)

    predictions_matched_max = predictions_sorted[predictions_matched_mask][:DIGITS_AMOUNT]
    inputs_matched_max = inputs_sorted[predictions_matched_mask][:DIGITS_AMOUNT]
    targets_matched_max = targets_sorted[predictions_matched_mask][:DIGITS_AMOUNT]

    predictions_not_matched_max = predictions_sorted[~predictions_matched_mask][:DIGITS_AMOUNT]
    inputs_not_matched_max = inputs_sorted[~predictions_matched_mask][:DIGITS_AMOUNT]
    targets_not_matched_max = targets_sorted[~predictions_matched_mask][:DIGITS_AMOUNT]

    visualisation.plot_matched_and_unmatched(predictions_matched_max, inputs_matched_max, targets_matched_max,
                                             predictions_not_matched_max, inputs_not_matched_max,
                                             targets_not_matched_max, save_path='graphs/best_and_worst.html')

    print(model_confidence[model_confidence_idx])


def save_model(logreg: LogReg, path: str):
    with open(path, 'wb+') as f:
        pickle.dump(logreg, f)


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    ROOT_DIR = os.path.abspath(os.curdir)
    np.random.seed(200)
    digits = Digits(cfg)

    model_path = input("Do you want to get a model from a file? Please, specify the file name or leave it empty: ").strip()
    if len(model_path) != 0:
        model_path = os.path.join(ROOT_DIR, "saves", model_path)
        logreg = load_model(model_path)
    else:
        logreg = LogReg(cfg, digits.k, digits.d, reg_coeff=0.005)
        logreg.train(digits.inputs_train, digits.targets_train, digits.inputs_valid, digits.targets_valid)

    main_task(logreg, digits)

    filename = input("Save model's params in the file? Enter the file name (leave empty if you don't want to save): ").strip()
    if len(filename) != 0:
        path = os.path.join(ROOT_DIR, "saves", filename)
        save_model(logreg, path)
