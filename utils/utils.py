import glob

import numpy as np

from utils.enums import *

from typing import Type, Union
import pandas as pd
from scipy.special import expit
import tensorflow as tf

from utils.plotter import plot_multi_model_predictions
from utils.enums import ClassValues


def make_data_path(
    class_type: Type[ClassValues],
    nm: int,
    num_outliers: Union[float, Type[DirValues]],
    noise: Union[float, Type[DirValues]],
    npps: int,
    num_samples: int,
    train_or_test: Options,
):
    """
    builds path like "/CLASS_TYPE/NM/no_NO_noise_NOISE/npps_NPPS/ns_NUM_SAMPLES/[TRAIN|TEST]

    :param class_type: ClassValues.CIRCLES | ClassValues.LINES | ClassValues.ELLIPSES | ClassValues.CONICS
    :param nm: number of models
    :param num_outliers:
    :param noise:
    :param npps: num points per sample
    :param num_samples:
    :param train_or_test: Options enum value, "Options.TRAIN" or "Options.TEST"
    :return:
    """
    if type(num_outliers) == DirValues:
        num_outliers = num_outliers.value
    if type(noise) == DirValues:
        noise = noise.value

    path = (
        "/"
        + class_type.value
        + "/nm_"
        + str(nm)
        + "/no_"
        + str(num_outliers)
        + "_noise_"
        + str(noise)
        + "/npps_"
        + str(npps)
        + "/ns_"
        + str(num_samples)
        + "/"
        + train_or_test.value
    )
    return path


def find_best_weights(path):
    """
    :path: folder into which weights are to be looked for
    :return: name of the h5 file that contains the best weights found so far.
             (assumes the files are within the folder in which you are calling the method)
    """
    best_weight_fn_list = glob.glob(path + "/best_weights_*.h5")
    best_epoch = max(
        [
            int(best_weight_fn.split("_")[2].split(".")[0])
            for best_weight_fn in best_weight_fn_list
        ]
    )
    return path + "/best_weights_" + str(best_epoch) + ".h5"


def split_train_valid(data, labels, train_ratio):
    """
    split data into train and validation set.

    :param data: numpy array, contains data
    :param labels: numpy array, contains labels of data
    :param train_ratio: percentage of data to be used in training
    """
    n_train_data = int(data.shape[0] * train_ratio)
    mask = np.ones(shape=(data.shape[0],))
    mask[n_train_data:] = 0
    np.random.shuffle(mask)
    mask = mask.astype(bool)
    train_data = data[mask]
    train_labels = labels[mask]
    valid_data = data[~mask]
    valid_labels = labels[~mask]
    return train_data, train_labels, valid_data, valid_labels


def save_perfs_and_plot_preds(
        data: Type[np.ndarray],
        labels: Type[np.ndarray],
        model: tf.keras.models.Model,
        path: str,
        is_test: bool = True,
        plot: bool = True,
        auto_find_weights: bool = True
):
    """
    saves performance of data in a csv file.
    plots predictions for randomly chosen data in the test set.
    assuming you are in the same folder where weights are saved.

    :param data: np.array, (ns, npps, 4), used for predictions
    :param labels: np.array, (ns, npps, nm), used for metrics
    :param model: tf.keras.model
    :param path: folder into which results are to be saved
    :param is_test: bool, if true saves performance in test_performance.csv
                          if false saves performance in valid_performance.csv
    :param auto_find_weights: if True retrieves the best weights
    :param plot: bool, if true plots, otherwise not
    """
    bs = model.input.shape[0]
    if auto_find_weights:
        best_weights_path = find_best_weights(path=path)
        print('###### best weights ######\n"{}"'.format(best_weights_path))
        model.load_weights(best_weights_path)

    score = model.evaluate(data,
                           labels,
                           verbose=0,
                           batch_size=bs)

    print("###### test metrics ######")
    print("accuracy: ", score[1])
    print("idr: ", score[3])
    print("odr: ", score[4])
    print("f1-score: ", score[5])

    predictions = model.predict(data, batch_size=64)
    predictions = expit(predictions)

    IMGS_FOLDER_NAME = path + "/imgs/final"
    plot_multi_model_predictions(predictions, data, IMGS_FOLDER_NAME, plot, plot_predicted_models=False)

    performance = {
        "loss": score[0],
        "accuracy": score[1],
        "number of samples predicted inliers": score[2],
        "inliers detection rate": score[3],
        "outliers detection rate": score[4],
        "f1score": score[5],
    }

    performance_name = path + "/test_performance.csv" if is_test else "valid_performance.csv"
    pd.DataFrame([performance]).to_csv(performance_name)

