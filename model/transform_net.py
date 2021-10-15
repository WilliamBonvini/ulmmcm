from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Reshape


def feature_transform_net(inputs, k=64):
    """Feature Transform Net, input is Bx1xNxK
    Return:
        Transformation matrix of size KxK"""

    tnetLayer1 = Conv2D(64, (1, 1), strides=1, activation="relu")(inputs)
    tnetLayer1_n = BatchNormalization()(tnetLayer1)

    tnetLayer2 = Conv2D(128, (1, 1), strides=1, activation="relu")(tnetLayer1_n)
    tnetLayer2_n = BatchNormalization()(tnetLayer2)

    tnetLayer2_n = Conv2D(64, (1, 1), strides=1, activation="relu")(
        tnetLayer2_n
    )
    tnetLayer2_n = BatchNormalization()(tnetLayer2_n)

    tnetLayer3 = Conv2D(1024, (1, 1), strides=1, activation="relu")(tnetLayer2_n)
    tnetLayer3_n = BatchNormalization()(tnetLayer3)

    tnetLayer4 = tf.math.reduce_max(
        tnetLayer3_n, axis=1, keepdims=True
    )

    tnetLayer4 = Flatten()(tnetLayer4)

    tnetLayer5 = Dense(512, activation="relu")(tnetLayer4)
    tnetLayer5_n = BatchNormalization()(tnetLayer5)

    tnetLayer6 = Dense(256, activation="relu")(tnetLayer5_n)
    tnetLayer6 = BatchNormalization()(tnetLayer6)

    net = tnetLayer6
    net = Dense(k * k, weights=[np.zeros([256, k * k]).astype(np.float32), np.eye(k).flatten().astype(np.float32), ],)(net)
    T = Reshape((k, k))(net)
    return T

