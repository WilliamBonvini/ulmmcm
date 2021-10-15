from utils.enums import ClassValues
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def veronese_map(seg_input, n):
    cols = []
    i = n
    while i >= 0:
        j = n - i
        while j >= 0:
            k = n - i - j
            cols.append(K.expand_dims(K.pow(seg_input[:, :, 0], i) * K.pow(seg_input[:, :, 1], j) * K.pow(seg_input[:, :, 2],k)))  # x_1^i * x_2^j * x_3^k
            j -= 1
        i -= 1
    return cols


def get_vandermonde_matrix(segmentation_inputs, nm, is_loss_v1, class_type, n_coords):
    """
    multimodel vandermonde loss v2, for GPCA
    :param segmentation_inputs: tensor, (bs, npps, 1, n_coords)
    :param nm:
    :param is_loss_v1
    :return:
    """
    bs = segmentation_inputs.shape[0]

    seg_input = segmentation_inputs
    print("seg_input shape: {}".format(seg_input.shape))
    if seg_input.shape[-2] == 1:
        seg_input = K.squeeze(segmentation_inputs, axis=-2)  # (bs, npps, 1, n_coords) -> (bs, npps, n_coords)

    if class_type == ClassValues.HOMOGRAPHIES:
        if n_coords == 6:
            seg_input_ = tf.gather(seg_input, tf.constant([0, 1, 3, 4]), axis=-1)
        else:
            seg_input_ = seg_input
        print("vandermonde for homography")
        vandermondes = []
        for i in range(bs):
            X = seg_input_[i, ...]
            X = K.transpose(X)
            N = X.shape[1]

            combinations = np.array(
                ((1, 0, 1, 0),  # xu
                 (1, 0, 0, 1),  # xv
                 (1, 0, 0, 0),  # x
                 (0, 1, 1, 0),  # yu
                 (0, 1, 0, 1),  # yv
                 (0, 1, 0, 0),  # y
                 (0, 0, 1, 0),  # u
                 (0, 0, 0, 1)))  # v

            combinationsRow = combinations > 0

            nTerms = combinationsRow.shape[0] + 1
            vandermonde = np.zeros((N, nTerms))

            for ti in range(0, nTerms - 1):
                #dimIdx = np.where(combinationsRow[ti, :])[0]
                dimIdx = tf.constant(np.where(combinationsRow[ti, :])[0])
                # tval = X[dimIdx, :]  # (dimIdx.shape[0], 512)
                tval = tf.gather(X, dimIdx, axis=0)

                """
                degrees = combinations[ti, dimIdx]
                for r in range(1, tval.shape[0]):
                    tval[r, :] = np.power(tval[r, :], degrees[r])
                """

                # multiply together the variables x and u:
                tval_final = np.prod(tval, axis=0)
                vandermonde[:, ti] = tval_final

            vandermonde[:, nTerms - 1] = 1
            # vandermonde[:, nTerms - 1] = np.ones((vandermonde.shape[0]))

            vandermonde = np.divide(vandermonde, np.linalg.norm(vandermonde, axis=1).reshape((-1, 1)))
            # vandermonde = tf.reshape(vandermonde, (1, vandermonde.shape[0], vandermonde.shape[1]))
            vandermonde = tf.constant(vandermonde)
            vandermondes.append(vandermonde)

        vandermondes = tf.stack(vandermondes)
        print("vandermondes shape: {}".format(vandermondes.shape))
        return vandermondes

    # 1 circle
    if class_type == ClassValues.CIRCLES:
        if is_loss_v1:
            # circle fitting
            if len(segmentation_inputs.shape) == 4:
                i = K.squeeze(segmentation_inputs, axis=2)
            # no squeezing for pointnet
            else:
                i = segmentation_inputs
                print("fitting per pointnet")
            one = K.expand_dims(K.sum(K.square(i), axis=2))  # x^2 + y^2
            two = K.expand_dims(i[:, :, 0], axis=2)  # x
            three = K.expand_dims(i[:, :, 1], axis=2)  # y
            four = K.ones_like(one)  # 1
            cols = [one, two, three, four]

        if not is_loss_v1:
            if nm == 1:
                if n_coords == 3:
                    base_degree = 2
                    print("vandermonde matrix for 1 circle")
                    cols = veronese_map(seg_input, base_degree)
                elif n_coords == 2:
                        i = K.squeeze(segmentation_inputs, axis=2)
                        one = K.expand_dims(K.sum(K.square(i), axis=2))  # x^2 +y^2
                        two = K.expand_dims(i[:, :, 0], axis=2)  # x
                        three = K.expand_dims(i[:, :, 1], axis=2)  # y
                        four = K.ones_like(one)  # 1
                        cols = [one, two, three, four]

            if nm == 2:
                print("vandermonde matrix for 2 circles")
                i = K.squeeze(segmentation_inputs, axis=2)
                one = K.expand_dims(K.sum(K.pow(i, 4), axis=2))  # x^4 + y^4
                two = K.expand_dims(K.prod(K.pow(i, 2), axis=2))  # x^2y^2
                three = K.expand_dims(K.pow(i[:, :, 0], 3) + i[:, :, 0] * K.square(i[:, :, 1]))  # x^3 + xy^2
                four = K.expand_dims(K.pow(i[:, :, 1], 3) + K.square(i[:, :, 0]) * i[:, :, 1])  # y^3 + x^2y
                five = K.expand_dims(K.square(i[:, :, 0]))  # x^2
                six = K.expand_dims(K.prod(i, axis=2))  # xy
                seven = K.expand_dims(K.square(i[:, :, 1]))  # y ^2
                eight = K.expand_dims(i[:, :, 0])  # x
                nine = K.expand_dims(i[:, :, 1])  # y
                ten = K.ones_like(one)  # 1
                cols = [one, two, three, four, five, six, seven, eight, nine, ten]

            if nm == 3:
                print("vandermonde matrix for 3 circles")
                cols = veronese_map(seg_input, 6)

    vander = K.concatenate(cols)

    sample_vanders = []
    for i in range(segmentation_inputs.shape[0]):
        sample_vander = tf.math.divide(vander[i], K.reshape(tf.linalg.norm(vander[i], axis=1), (-1, 1)))
        sample_vanders.append(sample_vander)

    pp_vander = tf.stack(sample_vanders)

    return pp_vander
