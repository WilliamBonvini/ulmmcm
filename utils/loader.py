import os
import pickle

import numpy as np

from utils.enums import ClassValues, TrainParams
from utils.vandermonde import get_vandermonde_matrix


def config_data(class_type: ClassValues,
                nm: int,
                noise_range: TrainParams,
                outliers_range: TrainParams,
                ns: int,
                npps: int,
                train_or_test: str,
                is_loss_v1: bool = True,
                shuffle: bool = True):
    """
    Used to configure data for training. It puts in the same structure data with different settings
    (i.e. different outliers rate) in such a way the neural network is fed heterogeneous data


    :param class_type: ClassValues.CIRCLES or ClassValues.HOMOGRAPHIES
    :param nm: number of models in each data sample
    :param noise_range: choose among values in TrainParams
    :param outliers_range: choose among values in TrainParams
    :param ns: number of samples to be retrieved for each (noise percentage, outlier rate) pair
    :param npps: number of points per sample
    :param train_or_test: specify whether it's "train" or "test" data
    :param is_loss_v1: default is True. Setting it to False will transform the data in order for the 2nd loss formulation to be used
    :param shuffle:
    :return:
    """

    # retrieve number of point coordinates
    if class_type == ClassValues.HOMOGRAPHIES:
        n_coords = 4
    elif class_type == ClassValues.CIRCLES:
        n_coords = 2
    else:
        raise Exception("Invalid class type")

    # create data skeleton
    num_files = len(outliers_range.value) if type(outliers_range.value) == list else len(noise_range.value)
    alldata = np.zeros((ns * num_files, npps, n_coords))
    alllabels = np.zeros((ns * num_files, npps, nm))

    # build directory
    data_dir = "data/{}/nm_{}/npps_{}/noise_{}/{}/data".format(class_type.value, nm, npps, repr(noise_range.value),
                                                               train_or_test) if type(
        noise_range.value) == float else "data/{}/nm_{}/npps_{}/no_{}/{}/data".format(class_type.value, nm, npps,
                                                                                      repr(outliers_range.value),
                                                                                      train_or_test)

    # retrieve files we are interested in
    files = os.listdir(data_dir)

    # retrieve actual number of points per sample (anpps)
    with open(data_dir + "/" + files[0], 'rb') as handle:
        b = pickle.load(handle)
    data_sample = b['data'][0]['x1s']
    anpps = data_sample.shape[0] if class_type == ClassValues.CIRCLES else data_sample.shape[1]

    # populate alldata and alllabels
    for i_file, file in enumerate(files):
        # retrieve data and labels
        data, labels = get_data_and_labels_from_file(data_dir + "/" + file, class_type=class_type, nm=nm)

        # select only npps from anpps
        if shuffle:
            selection_mask = np.random.permutation(anpps)[:npps]
            data = data[:, selection_mask]
            labels = labels[:, selection_mask]

        # populate final data structure
        alldata[i_file * ns: (i_file + 1) * ns] = data[:ns]
        alllabels[i_file * ns: (i_file + 1) * ns] = labels[:ns]

    # add vandermonde to labels (needed for convenience in the loss/metrics computation)
    vander = get_vandermonde_matrix(segmentation_inputs=alldata,
                                    nm=nm,
                                    is_loss_v1=is_loss_v1,
                                    class_type=class_type,
                                    n_coords=n_coords)
    alllabels = np.concatenate((alllabels, vander), axis=-1)

    # shuffle data
    if shuffle:
        shuffled_indexes = np.random.permutation(alldata.shape[0])
        alldata = alldata[shuffled_indexes]
        alllabels = alllabels[shuffled_indexes]

    # the convolutions in pointnet require to have an additional dimension in the input
    alldata = alldata[:, :, np.newaxis, :]

    return alldata, alllabels


def get_data_and_labels_from_file(path: str,
                                  class_type: ClassValues,
                                  nm: int):
    """

    :param path:
    :param class_type:
    :param nm:
    :return:
    """
    # retrieve number of coordinates
    if class_type == ClassValues.CIRCLES:
        n_coords = 2
    elif class_type == ClassValues.HOMOGRAPHIES:
        n_coords = 4
    else:
        raise Exception("Invalid class type")

    # retrieve data from file specified by path
    with open(path, 'rb') as handle:
        myfile = pickle.load(handle)

    # retrieve actual number of samples per file (a sample is data points belonging to models and outliers)
    nspf = len(myfile['data'].keys())
    # retrieve correct number of data points per sample (circles and homographies have been saved with different shape)
    if n_coords == 2:
        anpps = myfile['data'][0]['x1s'].shape[0]
    elif n_coords == 4:
        anpps = myfile['data'][0]['x1s'].shape[1]
    else:
        raise Exception("invalid number of coordinates")

    # declare empty structures for data and labels
    data = np.zeros((nspf, anpps, n_coords))
    labels = np.zeros((nspf, anpps, nm))

    # populate each sample with its data (shape (npps, n_coords)) and its labels (shape (npps, nm))
    for i_sample in myfile['data'].keys():
        x1s = myfile['data'][i_sample]['x1s']
        x2s = myfile['data'][i_sample]['x2s']

        if n_coords == 2:
            x1s = x1s.reshape(-1, 1)  # i.e. for circle it has shape 256, 1
            x2s = x2s.reshape(-1, 1)
            sample_labels = myfile['data'][i_sample]['labels']
        if n_coords == 4:
            x1s = np.transpose(x1s)[...,
                  0:2]  # original x1.shape is (n_coords = 3, npps = 512), with 3rd coord = 1. -> transpose it and remove 3rd component.
            x2s = np.transpose(x2s)[...,
                  0:2]  # original x2.shape is (n_coords = 3, npps = 512), with 3rd coord = 1. -> transpose it and remove 3rd component.
            sample_labels = np.transpose(myfile['data'][i_sample]['labels'])

        data[i_sample] = np.concatenate((x1s, x2s), axis=-1)
        labels[i_sample] = sample_labels
    return data, labels
