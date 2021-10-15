import glob
import os
import pickle
import re
import pandas as pd
import keras
import tensorflow.keras.backend as K
from scipy.special import expit

from utils.plotter import plot_multi_model_predictions


def delete_old_weights_and_opt(is_best: bool, path: str):
    """
    deletes the weights of the weights and optimizers of the last but one saved epoch.
    (lbo stands for last but one).
    :param is_best: if true deletes best_weights/op, otherwise deletes weights/opt
    :param path: str, folder in which the deletion takes place
    """

    # define name of files
    weights_re = (
        "weights_*.h5"
        if not is_best
        else "best_weights_*.h5"
    )
    weights_re = path + "/" + weights_re
    optimizer_re = (
        "optimizer_*.pkl"
        if not is_best
        else "best_optimizer_*.pkl"
    )
    optimizer_re = path + "/" + optimizer_re

    weight_fn_list = glob.glob(weights_re)

    if len(weight_fn_list) < 2:
        return

    weight_fn_list.sort(key=os.path.getctime)
    lbo_weight_file = weight_fn_list[-2]
    os.remove(lbo_weight_file)

    opt_fn_list = glob.glob(optimizer_re)

    if len(opt_fn_list) < 2:
        return
    opt_fn_list.sort(key=os.path.getctime)
    lbo_opt_file = opt_fn_list[-2]
    os.remove(lbo_opt_file)


def save_weights_and_opt(model, epoch: int, is_best: bool, path: str):
    """
    save weights and optimizer state of the model at a certain epoch
    :param model: tf.keras model
    :param epoch: int, epoch
    :param is_best: bool, pass True if you wanna save files as "best_weights_EPOCH.h5", "best_optimizer_EPOCH.pkl"
    :param path: str, folder in which weights and opt are to be saved
    :return:
    """
    # define name of files
    weights_name = "weights_" + str(epoch + 1) + ".h5"
    if is_best:
        weights_name = "best_" + weights_name

    optimizer_name = "optimizer_" + str(epoch + 1) + ".pkl"
    if is_best:
        optimizer_name = "best_" + optimizer_name

    # save weights
    model.save_weights(path + "/" + weights_name)
    # save optimizer state
    symbolic_weights = getattr(model.optimizer, "weights")
    weights_values = K.batch_get_value(symbolic_weights)
    with open(path + "/" + optimizer_name, "wb") as f:
        pickle.dump(weights_values, f)


class EpochsSaviourCallback(keras.callbacks.Callback):
    """
    callback class to save epoch metrics in a csv file
    """
    def __init__(self, filename):
        super(EpochsSaviourCallback, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, "a") as f:
            if epoch == 0:
                f.write('sep=,\n')
                f.write("epoch,")
                for key in list(logs.keys())[: len(logs.keys()) - 1]:
                    f.write("%s," % key)
                f.write("%s\n" % (list(logs.keys())[-1]))

            f.write("%s," % (epoch + 1))
            for key in list(logs.keys())[: len(logs.keys()) - 1]:
                f.write("%s," % (logs[key]))
            f.write("%s\n" % (logs[list(logs.keys())[-1]]))


class StateCallback(keras.callbacks.Callback):
    """
    callback class to save weights and optimizer state
    """
    def __init__(self, draw, valid_d, path):
        super(StateCallback, self).__init__()
        self.draw = draw
        self.valid_d = valid_d
        self.path = path
        self.best_epochs_filename = path + '/best_epoch.csv'

    def on_epoch_end(self, epoch, logs=None):
        # save weights and optimizer every x epochs
        if (epoch + 1) % 5 == 0:
            save_weights_and_opt(model=self.model, epoch=epoch, is_best=False, path=self.path)
            # delete old ones
            delete_old_weights_and_opt(path=self.path, is_best=False)

        # save imgs every y epochs
        if self.draw and ((epoch + 1) % 5 == 0 or epoch == 0):

            predictions = self.model.predict(self.valid_d)
            predictions = expit(predictions)
            path = self.path + "/imgs/" + str(epoch+1)
            plot_multi_model_predictions(predictions, self.valid_d, path, plot=False, plot_predicted_models=False)

        # save best epoch if it's the case
        val_keys = []
        for key in logs.keys():
            matched = re.search(r"val_*", key)
            if matched:
                val_keys.append(key)

        val_values = [str(logs[key]) for key in val_keys]

        first_line = ['best_epoch']
        first_line.extend(val_keys)
        first_line = ",".join(first_line) + "\n"

        second_line = [str(epoch + 1)]
        second_line.extend(val_values)
        second_line = ",".join(second_line)

        curr_perf = [first_line, second_line]

        if not os.path.exists(self.best_epochs_filename):

            file1 = open(self.best_epochs_filename, "w")
            file1.writelines(curr_perf)
            file1.close()

        else:
            old_perfs = pd.read_csv(self.best_epochs_filename)
            best_loss_so_far = float(old_perfs['val_loss'][0])
            best_acc_so_far = float(old_perfs['val_acc'][0])

            if float(logs["val_loss"]) < best_loss_so_far:
                # writing to file
                file1 = open(self.best_epochs_filename, "w")
                file1.writelines(curr_perf)
                file1.close()

                # save best weights and optimizer state so far
                save_weights_and_opt(model=self.model, epoch=epoch, is_best=True, path=self.path)

                # delete old best weights if new accuracy is higher than previous one
                if float(logs["val_acc"]) > best_acc_so_far:
                    delete_old_weights_and_opt(is_best=True, path=self.path)