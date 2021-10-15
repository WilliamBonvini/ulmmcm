import os
from random import randrange
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.vis_utils import plot_model
import syndalib.drawer as sydraw
import syndalib.linalg as syla
from utils.enums import ClassValues
from utils.vandermonde import get_vandermonde_matrix


def get_rc(model_index):
    """
    0 --> 0,1
    1 --> 1,0
    2 --> 1,1
    3 --> 2,1

    :param model_index: index of the model in the pic ( from 0 t0 num_models -1)
    :return: row and column index of axs that refer to the plotting of that particular model
    """
    num = model_index + 1
    row = int(np.floor(num / 2.0))
    col = num - 2 * row
    return row, col


def plot_multi_model_sample(i_sample,
                            sample_xy,
                            sample_preds: np.ndarray,
                            path: str,
                            plot: bool,
                            plot_predicted_models: bool,
                            vander,
                            heatmaps=True):
    """
    :param sample_xy: (num points per sample, num coordinates=2)
    :param sample_preds: (num points per sample, num models)
    :param path: name of img folder
    :param plot: bool, if true plots otherwise not
    :param plot_predicted_models: bool, if true plots the predicted models along with the heatmaps
    :param vander: IT  SHOULD BE (npps, num monomials), but it's (ns, npps, num monomials) np.array vandermonde matrix of the sample
    :param heatmaps:

    :return:
    """

    plt.clf()
    color_outliers = "tab:grey"
    colors = [
        "tab:orange",
        "tab:green",
        "tab:blue",
        "tab:pink",
        "tab:red",
        "tab:black",
    ]
    assert len(sample_xy) == len(sample_preds)
    num_points_per_sample = sample_preds.shape[0]
    num_models = sample_preds.shape[1]

    if not heatmaps:
        for i in range(num_models):
            inliers = [
                sample_xy[j]
                for j in range(num_points_per_sample)
                if sample_preds[j, i] < 0.5
            ]
            if len(inliers) > 0:
                plt.scatter(*zip(*inliers), s=10)
        # compute outliers as points that are classified as outliers for all models
        outliers = [
            sample_xy[j]
            for j in range(num_points_per_sample)
            if all(sample_preds[j, :] > 0.5)
        ]
        plt.scatter(*zip(*outliers), s=10)

    if heatmaps:
        ### first ax for all models together, the remaining for single ones.
        fig, axs = plt.subplots(
            int(np.ceil((num_models + 1) / 2.0)), 2, figsize=(10, 10), squeeze=False
        )
        fig.suptitle("Sample's Plot", fontsize="xx-large")
        for ax in fig.get_axes():
            ax.label_outer()

        # plot entire sample (every model + outliers)
        axs[0][0].set_aspect("equal", "datalim")
        axs[0][0].set_title("all models and outliers")
        model_markers = []
        for i in range(num_models):
            model_marker = mlines.Line2D(
                [],
                [],
                color=colors[i],
                marker=".",
                linestyle="None",
                markersize=10,
                label="model" + str(i + 1),
            )
            model_markers.append(model_marker)
            inliers = [sample_xy[j] for j in range(num_points_per_sample) if sample_preds[j, i] < 0.5]
            # inliers = [sample_xy[j] for j in range(num_points_per_sample) if tf.less(sample_preds[j, i], 0.5)]
            if len(inliers) > 0:
                axs[0][0].scatter(*zip(*inliers), c=colors[i], s=10)
        # compute outliers as points that are classified as outliers for all models
        outliers = [
            sample_xy[j]
            for j in range(num_points_per_sample)
            if all(sample_preds[j, :] >= 0.5)
        ]
        if len(outliers) > 0:
            axs[0][0].scatter(*zip(*outliers), c=color_outliers, s=10)

        outliers_marker = mlines.Line2D([], [], color=color_outliers, marker=".", linestyle="None", markersize=10, label="outliers")

        model_markers.append(outliers_marker)
        axs[0][0].legend(handles=model_markers, loc="upper right")

        # plot each model separately
        for i in range(num_models):
            r, c = get_rc(i)
            inliers_prob = 1.0 - sample_preds[:, i]
            axs[r][c].set_title("model " + str(r * 2 + c))
            axs[r][c].set_aspect("equal", "datalim")
            sc = axs[r][c].scatter(*zip(*sample_xy), c=inliers_prob, s=10, vmin=0, vmax=1)

            if plot_predicted_models:
                predicted_coefs = syla.dlt_coefs(vander, inliers_prob, returned_type="numpy")

                # hard coding case circle case!
                a = predicted_coefs[0]
                b = 0
                cc = predicted_coefs[0]
                d = predicted_coefs[1]
                e = predicted_coefs[2]
                f = predicted_coefs[3]
                coefs = [a, b, cc, d, e, f]
                cx, cy = sydraw.conic_points(coefs=coefs,
                                             x_range=(-2, 2),
                                             y_range=(-2, 2),
                                             resolution=2000)  # maybe it's useless to specify ranges
                axs[r][c].scatter(cx, cy, s=1, c="tab:purple")

            cbar = plt.colorbar(sc, ax=axs[r][c])
            cbar.set_label("inlier probability")

    # save image
    os.makedirs(path, exist_ok=True)
    title = path + "/" + str(i_sample) + ".png"
    plt.savefig(title)

    if plot is True:
        plt.show()


def plot_multi_model_predictions(predictions, xy, path, plot: bool, plot_predicted_models: bool):
    """

    :param predictions: (bs, num points per sample, num models) predictions
    :param xy: (bs, num points per sample, 1, num coords) points coordinates
    :param path: path of the folder in which to save imgs
    :param plot: bool, if true plots and saves img otherwise just saves img
    :param plot_predicted_models: bool, if true plots along with the inliers probability the predicted model
    :return:
    """
    if plot_predicted_models:
        nm = predictions.shape[-1]
        n_coords = xy.shape[-1]
        vanders = get_vandermonde_matrix(segmentation_inputs=xy,
                                            nm=nm,
                                            is_loss_v1=True,
                                            class_type=ClassValues.CIRCLES,
                                            n_coords=n_coords)

    i_sample = 0
    while i_sample < predictions.shape[0]:
        if plot_predicted_models:
            sample_vander = vanders[i_sample]
        else:
            sample_vander = None

        # retrieve coordinates of circle
        if len(xy.shape) == 4:
          sample_xy = xy[i_sample, :, 0, 0:2]
        else:
          sample_xy = xy[i_sample, :, 0:2]

        sample_preds = predictions[i_sample]
        plot_multi_model_sample(i_sample, sample_xy, sample_preds, path, plot, plot_predicted_models=plot_predicted_models, vander=sample_vander)
        i_sample += 100
