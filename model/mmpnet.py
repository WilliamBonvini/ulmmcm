from __future__ import print_function

from typing import Tuple

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import RepeatVector, Reshape
from tensorflow.keras.models import Model

from losses import mm_loss, il_outer, vl_outer, sl, variance_outer, crossentropy_loss
from metrics import *
from model.transform_net import feature_transform_net
from utils.enums import ClassValues

tf.compat.v1.enable_eager_execution()

def get_mmpn(bs: int,
             npps: int,
             hps: Tuple,
             nm: int,
             is_loss_v1: bool,
             class_type: ClassValues,
             is_supervised: bool = False,
             lr: float = 0.001,
             n_gf: int = 1024):
    """

    :param bs: batch size
    :param npps: number of points per sample
    :param hps: hyperparameters
    :param nm: number of models
    :param is_loss_v1: True: default loss. False: union loss
    :param classtype:
    :param lr: learning rate
    :param is_supervised: valid for single model. True: crossentropy loss. False: default loss
    :param n_gf: number of global features in pointnet
    :return:
    """

    inliers_lambda, vander_lambda, sim_lambda, var_lambda = hps

    if not is_supervised:
        if is_loss_v1:
            output_cols = nm
        else:
            output_cols = 1
    if is_supervised:
        if nm == 1:
            output_cols = nm
        else:
            output_cols = nm + 1

    if class_type == ClassValues.CIRCLES:
        n_coords = 2
    elif class_type == ClassValues.HOMOGRAPHIES:
        n_coords = 4
    else:
        raise Exception("Invalid class type")

    segmentation_inputs = Input(shape=(npps, 1, n_coords), batch_size=bs)

    # n_coords X n_coords TNET
    transform_net = feature_transform_net(segmentation_inputs, n_coords)
    input_mat = Reshape((npps, n_coords))(segmentation_inputs)
    transformed_input = tf.matmul(input_mat, transform_net)
    transformed_input = K.expand_dims(transformed_input, axis=[2])

    # FIRST MLPs here
    normalized_inputs = BatchNormalization()(transformed_input)
    segLayer1 = Conv2D(64, (1, 1), strides=1, activation="relu")(normalized_inputs)
    segLayer1_n = BatchNormalization()(segLayer1)
    localFeatures = Conv2D(64, (1, 1), strides=1, activation="relu")(segLayer1_n)
    localFeatures_n = BatchNormalization()(localFeatures)

    # 64X64 TNET
    featureT = feature_transform_net(localFeatures_n, 64)
    mat = Reshape((npps, 64))(localFeatures_n)
    localFeatures_n = tf.matmul(mat, featureT, name="local_features_not_expanded")
    localFeatures_n_ = K.expand_dims(localFeatures_n, axis=[2])

    # SECOND MLPs here
    segLayer3 = Conv2D(64, (1, 1), strides=1, activation="relu")(localFeatures_n_)
    segLayer3_n = BatchNormalization()(segLayer3)
    segLayer4 = Conv2D(128, (1, 1), strides=1, activation="relu")(segLayer3_n)
    segLayer4_n = BatchNormalization()(segLayer4)
    segLayer5 = Conv2D(n_gf, (1, 1), strides=1, activation="relu")(segLayer4_n)

    # max pooling layer
    segLayer5_n = BatchNormalization()(segLayer5)
    segLayer6 = MaxPooling2D((npps, 1), strides=1, padding="valid")(segLayer5_n)
    segLayer7 = Flatten()(
        segLayer6
    )

    # merge local and global features into one vector
    segLayer8 = RepeatVector(npps)(segLayer7)
    globalFeatures = Reshape((npps, 1, n_gf))(segLayer8)
    mixedFeatures = tf.keras.layers.concatenate([localFeatures_n_, globalFeatures])

    # third MLP
    mixedFeatures_n = BatchNormalization()(mixedFeatures)  # forse non serve a niente, non penso penalizzi, ma maybe...
    segLayer11 = Conv2D(512, (1, 1), strides=1, activation="relu")(mixedFeatures_n)
    segLayer11_n = BatchNormalization()(segLayer11)
    segLayer11_n = Dropout(0.25)(segLayer11_n)  # -> appena aggiunto
    segLayer12 = Conv2D(256, (1, 1), strides=1, activation="relu")(segLayer11_n)
    segLayer12_n = BatchNormalization()(segLayer12)
    pointFeatures = Conv2D(128, (1, 1), strides=1, activation="relu")(segLayer12_n)
    pointFeatures_n = BatchNormalization()(pointFeatures)

    # third MLP
    segLayer14 = Conv2D(128, (1, 1), strides=1, activation="relu")(pointFeatures_n)
    segLayer14_n = BatchNormalization()(segLayer14)
    if nm > 10:
        segLayer14_n = Conv2D(64, (1, 1), strides=1, activation="relu")(segLayer14_n)
        segLayer14_n = BatchNormalization()(segLayer14_n)
        segLayer14_n = Conv2D(32, (1, 1), strides=1, activation="relu")(segLayer14_n)
        segLayer14_n = BatchNormalization()(segLayer14_n)

    predictions = Conv2D(output_cols, (1, 1), strides=1, activation=None)(segLayer14_n)
    predictions = Reshape((npps, output_cols))(predictions)

    # model creation
    segmentation_model = Model(inputs=segmentation_inputs, outputs=predictions)

    metrics = [
        acc,
        nspi,
        idr,
        odr,
        f1,
        il_outer(i_model=-1),
        vl_outer(i_model=-1, classtype=class_type),
        variance_outer(i_model=-1),
        sl,
    ]

    if is_loss_v1:
        single_ils = [il_outer(i) for i in range(nm)]
        single_vls = [vl_outer(i, classtype=class_type) for i in range(nm)]

        metrics.extend(single_ils)
        metrics.extend(single_vls)

    # compile
    if is_supervised and nm == 1:
            loss = crossentropy_loss
    else:
        loss = mm_loss(inliers_lambda=inliers_lambda,
                       vander_lambda=vander_lambda,
                       sim_lambda=sim_lambda,
                       var_lambda=var_lambda,
                       classtype=class_type)

    segmentation_model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=metrics,
        run_eagerly=True
    )

    return segmentation_model
