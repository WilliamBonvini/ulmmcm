import itertools
import tensorflow as tf
import tensorflow.keras.backend as K
tf.compat.v1.enable_eager_execution()


def create_permutations(num_models):
    """
    create_permutations(3) returns [[0,1,2],[0,2,1],[1,0,2],[1,2,0],...]

    :param num_models:
    :return: list of lists of permutations
    """
    indexes = [i for i in range(0, num_models)]
    perms = list(itertools.permutations(indexes))
    perms = [list(perm) for perm in perms]
    return perms


def find_best_perm_MM(y_true, y_pred):
    """
    accuracy with variable models

    :param y_true: (BS, NPOINTS, NMODELS)
    :param y_pred: (BS, NPOINTS, NMODELS)
    :return: best y pred permutation
    """

    BS = y_pred.shape[0]
    num_models = y_pred.shape[2]
    inlierTresh = 0.5

    y_pred_out = []
    for sample in range(BS):
        best_acc = 0.0
        pred_with_best_permutation = tf.gather(y_pred[sample], [i for i in range(num_models)], axis=-1)
        for perm in create_permutations(num_models):

            pred = tf.gather(y_pred[sample], perm, axis=-1)
            totAcc = 0.0
            for model in range(num_models):
                inliersProb = 1.0 - K.sigmoid(pred[:, model])
                predictedInliers = K.cast(
                    K.greater(inliersProb, inlierTresh), "float32"
                )
                acc = K.mean(
                    K.cast(
                        K.not_equal(predictedInliers, y_true[sample, :, model]),
                        "float32",
                    )
                )
                totAcc += acc

            totAcc = totAcc / num_models

            if totAcc > best_acc:
                pred_with_best_permutation = pred
                best_acc = totAcc
        y_pred_out.append(
            K.reshape(
                pred_with_best_permutation,
                shape=(
                    1,
                    pred_with_best_permutation.shape[0],
                    pred_with_best_permutation.shape[1],
                ),
            )
        )

    y_pred_out = tf.concat(y_pred_out, axis=0)

    return y_pred_out


def precision(y_true, y_pred):
    # TP / TP + FP
    inliersGT = 1.0 - y_true
    inlierTresh = 0.5
    inliersProb = 1.0 - K.sigmoid(y_pred)
    predictedInliers = K.cast(K.greater(inliersProb, inlierTresh), "float32")
    trueInliers = predictedInliers * inliersGT
    trueInliersCount = K.sum(trueInliers)
    falseInliers = predictedInliers * (1.0 - inliersGT)
    falseInliersCount = K.sum(falseInliers)
    precision = trueInliersCount / (trueInliersCount + falseInliersCount)
    return precision


def recall(y_true, y_pred):
    # TP / TP + FN
    inliersGT = 1.0 - y_true
    inlierTresh = 0.5
    inliersProb = 1.0 - K.sigmoid(y_pred)
    predictedInliers = K.cast(K.greater(inliersProb, inlierTresh), "float32")
    trueInliers = predictedInliers * inliersGT
    trueInliersCount = K.sum(trueInliers)
    falseOutliers = (1.0 - predictedInliers) * inliersGT
    falseOutliersCount = K.sum(falseOutliers)
    recall = trueInliersCount / (trueInliersCount + falseOutliersCount)
    return recall


def acc(y_true, y_pred):
    """
    accuracy
    :param y_true:
    :param y_pred:
    :return:
    """
    num_models = y_pred.shape[2]
    y_true = y_true[:, :, 0:num_models]
    y_p = find_best_perm_MM(y_true, y_pred)
    inliersTresh = 0.5
    totAcc = 0
    for model in range(num_models):
        inliersProb = 1.0 - K.sigmoid(y_p[..., model])
        predictedInliers = K.cast(K.greater(inliersProb, inliersTresh), "float32")
        temp = y_true[..., model]
        acc = K.mean(K.cast(K.not_equal(predictedInliers, temp), "float32"))
        totAcc += acc

    totAcc /= num_models
    return totAcc


def nspi(y_true, y_pred, avg=False):
    """
    number of samples predicted inliers
    :param y_true:
    :param y_pred:
    :param avg: explained below
    :return:
        avg = True -> average num of inliers for each model in the sample
        avg = False -> total num of samples predicted inliers, no matter what model they are in
    """
    n_coords = y_pred.shape[2]
    y_true = y_true[:, :, 0:n_coords]

    y_p = find_best_perm_MM(y_true, y_pred)
    inliersTresh = 0.5
    num_models = y_pred.shape[2]
    tot_num_inliers = 0.0
    for model in range(num_models):
        inliersProb = 1.0 - K.sigmoid(y_p[:, :, model])
        n_inliers = K.sum(K.cast(K.greater_equal(inliersProb, inliersTresh), "float32"), axis=1)
        tot_num_inliers += n_inliers

    if avg:
        tot_num_inliers = tot_num_inliers / num_models

    return tot_num_inliers


def idr(y_true, y_pred):
    """
    inliers detection rate
    :param y_true:
    :param y_pred:
    :return:
    """
    nm = y_pred.shape[2]
    y_true = y_true[:, :, 0:nm]

    y_p = find_best_perm_MM(y_true, y_pred)

    tot_inliers_det_rate = 0
    for model in range(nm):
        # model
        correctPred = K.equal(
            K.cast(K.greater_equal(K.sigmoid(y_p[:, :, model]), 0.5), "int64"),
            K.cast(y_true[:, :, model], "int64"),
        )
        correctPred = K.cast(correctPred, "float32")
        inliersGT = K.cast(K.less(y_true[:, :, model], 0.5), "float32")
        correctInliers = correctPred * inliersGT
        correctInliers = K.sum(K.cast(correctInliers, "float32")) / (K.sum(inliersGT))
        inliersDetectionRate = K.mean(correctInliers)
        tot_inliers_det_rate += inliersDetectionRate

    tot_inliers_det_rate = tot_inliers_det_rate / nm

    return tot_inliers_det_rate


def odr(y_true, y_pred):
    """
    outliers detection rate
    :param y_true:
    :param y_pred:
    :return:
    """
    n_coords = y_pred.shape[2]
    y_true = y_true[:, :, 0:n_coords]

    y_p = find_best_perm_MM(y_true, y_pred)
    num_models = y_pred.shape[2]

    tot_outliers_det_rate_MM = 0
    for model in range(num_models):
        correctPred = K.equal(
            K.cast(K.greater_equal(K.sigmoid(y_p[:, :, model]), 0.5), "int64"),
            K.cast(y_true[:, :, model], "int64"),
        )
        correctPred = K.cast(correctPred, "float32")
        outliersGT = K.cast(K.greater(y_true[:, :, model], 0.5), "float32")
        correctOutliers = correctPred * outliersGT
        correctOutliers = K.sum(K.cast(correctOutliers, "float32")) / (K.sum(outliersGT))
        outliersDetectionRate = K.mean(correctOutliers)
        tot_outliers_det_rate_MM += outliersDetectionRate

    tot_outliers_det_rate_MM = tot_outliers_det_rate_MM / num_models

    return tot_outliers_det_rate_MM


def f1(y_true, y_pred):
    """
    f1 score
    :param y_true:
    :param y_pred:
    :return:
    """
    n_coords = y_pred.shape[2]
    y_true = y_true[:, :, 0:n_coords]

    y_pred_good_perm = find_best_perm_MM(y_true, y_pred)
    num_models = y_pred.shape[2]

    tot_f1_score_MM = 0
    for model in range(num_models):
        yt = y_true[..., model]
        yp = y_pred_good_perm[..., model]
        p = precision(yt, yp)
        r = recall(yt, yp)
        tot_f1_score_MM += 2 * (r * p) / (r + p)

    f1_score_MM = tot_f1_score_MM / num_models

    return f1_score_MM
