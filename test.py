import os

import pandas as pd
from scipy.special import expit

from utils.loader import get_data_and_labels_from_file
from model.mmpnet import get_mmpn
from utils.enums import ClassValues, DirValues
import tensorflow as tf
import numpy as np
import warnings

from utils.plotter import plot_multi_model_predictions

tf.compat.v1.enable_eager_execution()

warnings.filterwarnings("ignore")

EXPS_MAPPING = {
    0: "stop execution",
    1: "2 circles with increasing outlier rate and fixed noise stddev at 0.01",
    2: "2 circles with outlier rate fixed at 25% and increasing noise stddev",
    3: "2 circles with outlier rate fixed at 50% and increasing noise stddev",
    4: "2 circles with outlier rate fixed at 60% and increasing noise stddev",
    5: "2 homographies with increasing outlier rate and noise stddev fixed at 0.01",
    6: "3 homographies with increasing outlier rate and noise stddev fixed at 0.01"
}


class ExperimentSetUp:
    """
       Circles
       1 -- 2 circles with increasing outlier rate and fixed noise stddev at 0.01
       2 -- 2 circles with fixed outlier rate at 25% and increasing noise stddev
       3 -- 2 circles with fixed outlier rate at 50% and increasing noise stddev
       4 -- 2 circles with fixed outlier rate at 60% and increasing noise stddev
       Homographies
       5 -- 2 homographies with increasing outlier rate and fixed noise stddev at 0.01
       6 -- 3 homographies with increasing outlier rate and fixed noise stddev at 0.01
    """

    def set_ID(self, expID):
        self.expID = expID
        if self.expID == 1:
            self.class_type = ClassValues.CIRCLES
            self.nm = 2
            self.npps = 256
            self.dir_params = (DirValues.VARIANT, 0.01)
            self.outliers_rate_range = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
            self.noise_range = None
            self.ns = 4096
            self.weights = "weights_2_circles.h5"

        if self.expID == 2:
            self.class_type = ClassValues.CIRCLES
            self.nm = 2
            self.npps = 256
            self.dir_params = (0.25, DirValues.VARIANT)
            self.outliers_rate_range = None
            self.noise_range = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            self.ns = 1024
            self.weights = "weights_2_circles.h5"

        if self.expID == 3:
            self.class_type = ClassValues.CIRCLES
            self.nm = 2
            self.npps = 256
            self.dir_params = (0.50, DirValues.VARIANT)
            self.outliers_rate_range = None
            self.noise_range = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            self.ns = 1024
            self.weights = "weights_2_circles.h5"

        if self.expID == 4:
            self.class_type = ClassValues.CIRCLES
            self.nm = 2
            self.npps = 256
            self.dir_params = (0.60, DirValues.VARIANT)
            self.outliers_rate_range = None
            self.noise_range = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            self.ns = 1024
            self.weights = "weights_2_circles.h5"

        if self.expID == 5:
            self.class_type = ClassValues.HOMOGRAPHIES
            self.nm = 2
            self.dir_params = (DirValues.VARIANT, 0.01)
            self.outliers_rate_range = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
            self.noise_range = None
            self.npps = 512
            self.ns = 4096
            self.weights = "weights_2_homographies.h5"

        if self.expID == 6:
            self.class_type = ClassValues.HOMOGRAPHIES
            self.nm = 3
            self.dir_params = (DirValues.VARIANT, 0.01)
            self.outliers_rate_range = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
            self.noise_range = None
            self.npps = 512
            self.ns = 4096
            self.weights = "weights_3_homographies.h5"

    def setup_model(self):
        """

        :return: a keras model
        """

        model = get_mmpn(bs=64,
                         npps=self.npps,
                         hps=(0, 0, 0, 0),
                         nm=self.nm,
                         is_loss_v1=True,
                         class_type=self.class_type,
                         is_supervised=False,
                         lr=0.001,
                         n_gf=1024)
        model.load_weights('data/weights/' + self.weights)
        self.model = model

    def setup_data(self, filename):
        """

        :return:
        """
        path = "data/"
        path += "circles" if self.class_type == ClassValues.CIRCLES else "homographies"
        path += "/nm_{}/npps_{}".format(self.nm, self.npps)
        if self.expID == 1 or self.expID == 5 or self.expID == 6:
            path += "/noise_0.01"
        elif self.expID == 2:
            path += "/no_0.25"
        elif self.expID == 3:
            path += "/no_0.5"
        elif self.expID == 4:
            path += "/no_0.6"
        path += "/test/data/" + filename + ".pkl"
        self.data, self.labels = get_data_and_labels_from_file(path,
                                                               class_type=self.class_type,
                                                               nm=self.nm)

    def predict(self):
        predictions = self.model.predict(self.data[:, :, np.nmewaxis, :])
        return predictions

    def evaluate(self):
        score = self.model.evaluate(self.data[:, :, np.newaxis, :],
                                    self.labels,
                                    verbose=0,
                                    batch_size=64)

        acc = "{:.3f}".format(score[1])
        idr = "{:.3f}".format(score[3])
        odr = "{:.3f}".format(score[4])
        f1 = "{:.3f}".format(score[5])

        # save metrics
        header = True if self.first_call else False
        pd.DataFrame(data={"Accuracy": [acc],
                           "Inliers Detection Rate": [idr],
                           "Outliers Detection Rate": [odr],
                           "F1-Score": [f1]}).to_csv(path_or_buf=self.results_path + "/metrics.csv",
                                                     mode='a',
                                                     header=header)
        self.first_call=False
        # print metrics
        print("accuracy: ", acc)
        print("idr: ", idr)
        print("odr: ", odr)
        print("f1-score: ", f1)

    def run_experiment(self, expID):
        print("Running: ", EXPS_MAPPING[expID])
        self.results_path = "results/test/" + EXPS_MAPPING[expID]
        self.first_call = True
        os.makedirs(self.results_path, exist_ok=True)

        self.set_ID(expID=expID)
        if self.expID == 1 or self.expID == 5 or self.expID == 6:
            self.setup_model()
            for outlier_rate in self.outliers_rate_range:
                print("******\noutlier rate: ",outlier_rate)
                self.setup_data(filename=str(outlier_rate))
                self.evaluate()
                self.save_imgs(additional_info=outlier_rate)

        if self.expID == 2 or self.expID == 3 or self.expID == 4:
            self.setup_model()
            for noise in self.noise_range:
                print("******\nnoise: ", noise)
                self.setup_data(filename=str(noise))
                self.evaluate()
                self.save_imgs(additional_info=noise)

    def save_imgs(self, additional_info):
        imgs_folder_name = self.results_path + "/imgs/" + str(additional_info)
        predictions = self.model.predict(self.data[:, :, np.newaxis, :], batch_size=64)
        predictions = expit(predictions)
        plot_multi_model_predictions(predictions, self.data, imgs_folder_name, plot=False, plot_predicted_models=False)


if __name__ == "__main__":

    exp = ExperimentSetUp()

    while True:
        for key, value in EXPS_MAPPING.items():
            print("{} - {}".format(key, value))

        id = int(input("ID: "))
        if id != 0:
            exp.run_experiment(id)
        else:
            break
