import tensorflow as tf
from tensorflow.keras import backend as K
from utils.enums import ClassValues

# inliers loss


def il_outer(i_model):

    # for all models
    if i_model == -1:
        def il(y_true, y_pred):

            inliers_prob = 1.0 - K.sigmoid(y_pred)

            # inliers loss
            inliers_mean_per_model = -K.mean(inliers_prob, axis=1)  # (bs, npps, nm) -> (bs, nm)

            # 21/10/20: cambio K.mean in K.sum qui sotto:
            inliers_loss = K.sum(inliers_mean_per_model, axis=1)  # (bs, nm) ->  (bs,)

            # inliers_loss = K.mean(inliers_loss)  # (bs,) -> (1,)

            return inliers_loss

        return il

    # for first model
    if i_model == 0:
        def il_0(y_true, y_pred):
            inliers_prob = 1.0 - K.sigmoid(y_pred)

            # inliers loss
            inliers_mean_per_model = -K.mean(inliers_prob, axis=1)  # (bs, npps, nm) -> (bs, nm)

            # inliers loss
            inliers_loss = inliers_mean_per_model[..., i_model]

            return inliers_loss

        return il_0

    # for second model
    if i_model == 1:
        def il_1(y_true, y_pred):
            inliers_prob = 1.0 - K.sigmoid(y_pred)

            # inliers loss
            inliers_mean_per_model = -K.mean(inliers_prob, axis=1)  # (bs, npps, nm) -> (bs, nm)

            # inliers loss
            inliers_loss = inliers_mean_per_model[..., i_model]

            return inliers_loss

        return il_1

    # for third model
    if i_model == 2:
        def il_2(y_true, y_pred):
            inliers_prob = 1.0 - K.sigmoid(y_pred)

            # inliers loss
            inliers_mean_per_model = -K.mean(inliers_prob, axis=1)  # (bs, npps, nm) -> (bs, nm)

            # inliers loss
            inliers_loss = inliers_mean_per_model[..., i_model]

            return inliers_loss

        return il_2

    # for fourth model
    if i_model == 3:
        def il_3(y_true, y_pred):
            inliers_prob = 1.0 - K.sigmoid(y_pred)

            # inliers loss
            inliers_mean_per_model = -K.mean(inliers_prob, axis=1)  # (bs, npps, nm) -> (bs, nm)

            # inliers loss
            inliers_loss = inliers_mean_per_model[..., i_model]

            return inliers_loss

        return il_3


# vander loss
def vl_outer(i_model: int,
             classtype: ClassValues):
    """

    :param i_model:
    :param classtype:
    :return:
    """

    if classtype == ClassValues.HOMOGRAPHIES:
        num_sing_vals = 3
    else:
        num_sing_vals = 1

    if i_model == -1:
        def vl(y_true, y_pred):
            nm = y_pred.shape[-1]
            vander = y_true[:, :, nm:]
            inliers_prob = 1.0 - K.sigmoid(y_pred)

            bs = inliers_prob.shape[0]
            vander_dim = vander.shape[-1]
            npps = vander.shape[1]
            nm = inliers_prob.shape[-1]

            weights = inliers_prob  # --> (bs, npps, nm)

            # add small random values to produce high vander loss in case of too few inliers
            weights = weights + K.random_uniform((bs, npps, nm), 0, 1e-9)  # --> (bs, npps, nm)
            # normalize weights: norm = 1 for all models
            weights = tf.linalg.normalize(weights, axis=1)[0]
            vander_loss = 0

            indexes = [i for i in range(nm)]

            for i in indexes:
                weights_model = tf.slice(weights, [0, 0, i], [bs, npps, 1])
                weights_model = K.reshape(weights_model, shape=(bs, npps))
                weights_model = tf.linalg.diag(weights_model)
                weighted_vander = K.batch_dot(weights_model, vander)

                s = tf.linalg.svd(weighted_vander, compute_uv=False)
                last_svs = s[:, vander_dim - num_sing_vals: vander_dim]
                vander_loss_model = K.mean(last_svs, axis=1)  # (bs,1) -> (bs,)
                vander_loss_model = K.mean(vander_loss_model)  # (bs,) -> (,) compute average over mini batch
                vander_loss = vander_loss + vander_loss_model

            vander_loss = vander_loss / len(indexes)
            return vander_loss

        return vl

    # for first model
    if i_model == 0:
        def vl_0(y_true, y_pred):
            nm = y_pred.shape[-1]
            vander = y_true[:, :, nm:]
            inliers_prob = 1.0 - K.sigmoid(y_pred)
            bs = inliers_prob.shape[0]
            vander_dim = vander.shape[-1]
            npps = vander.shape[1]
            nm = inliers_prob.shape[-1]

            weights = inliers_prob  # --> (bs, npps, nm)

            # add small random values to produce high vander loss in case of too few inliers
            weights = weights + K.random_uniform((bs, npps, nm), 0, 1e-9)  # --> (bs, npps, nm)
            # normalize weights: norm = 1 for all models
            weights = tf.linalg.normalize(weights, axis=1)[0]

            weights_model = tf.slice(weights, [0, 0, i_model], [bs, npps, 1])
            weights_model = K.reshape(weights_model, shape=(bs, npps))
            weights_model = tf.linalg.diag(weights_model)
            weighted_vander = K.batch_dot(weights_model, vander)

            s = tf.linalg.svd(weighted_vander, compute_uv=False)
            last_svs = s[:, vander_dim - num_sing_vals: vander_dim]

            vander_loss = K.mean(last_svs, axis=1)  # (bs,1) -> (bs,) (penso)
            vander_loss = K.mean(vander_loss)  # (bs,) -> (,) compute average over mini batch

            return vander_loss

        return vl_0

    # for second model
    if i_model == 1:
        def vl_1(y_true, y_pred):
            nm = y_pred.shape[-1]
            vander = y_true[:, :, nm:]
            inliers_prob = 1.0 - K.sigmoid(y_pred)
            bs = inliers_prob.shape[0]
            vander_dim = vander.shape[-1]
            npps = vander.shape[1]
            nm = inliers_prob.shape[-1]

            weights = inliers_prob  # --> (bs, npps, nm)

            # add small random values to produce high vander loss in case of too few inliers
            weights = weights + K.random_uniform((bs, npps, nm), 0, 1e-9)  # --> (bs, npps, nm)
            # normalize weights: norm = 1 for all models
            weights = tf.linalg.normalize(weights, axis=1)[0]

            weights_model = tf.slice(weights, [0, 0, i_model], [bs, npps, 1])
            weights_model = K.reshape(weights_model, shape=(bs, npps))
            weights_model = tf.linalg.diag(weights_model)
            weighted_vander = K.batch_dot(weights_model, vander)

            s = tf.linalg.svd(weighted_vander, compute_uv=False)
            last_svs = s[:, vander_dim - num_sing_vals: vander_dim]

            vander_loss = K.mean(last_svs, axis=1)  # (bs,1) -> (bs,) (penso)
            vander_loss = K.mean(vander_loss)  # (bs,) -> (,) compute average over mini batch
            return vander_loss

        return vl_1

    # for third model
    if i_model == 2:
        def vl_2(y_true, y_pred):
            nm = y_pred.shape[-1]
            vander = y_true[:, :, nm:]
            inliers_prob = 1.0 - K.sigmoid(y_pred)
            bs = inliers_prob.shape[0]
            vander_dim = vander.shape[-1]
            npps = vander.shape[1]
            nm = inliers_prob.shape[-1]

            weights = inliers_prob  # --> (bs, npps, nm)

            # add small random values to produce high vander loss in case of too few inliers
            weights = weights + K.random_uniform((bs, npps, nm), 0, 1e-9)  # --> (bs, npps, nm)
            # normalize weights: norm = 1 for all models
            weights = tf.linalg.normalize(weights, axis=1)[0]

            weights_model = tf.slice(weights, [0, 0, i_model], [bs, npps, 1])
            weights_model = K.reshape(weights_model, shape=(bs, npps))
            weights_model = tf.linalg.diag(weights_model)
            weighted_vander = K.batch_dot(weights_model, vander)

            s = tf.linalg.svd(weighted_vander, compute_uv=False)
            last_svs = s[:, vander_dim - num_sing_vals: vander_dim]

            vander_loss = K.mean(last_svs, axis=1)  # (bs,1) -> (bs,) (penso)
            vander_loss = K.mean(vander_loss)  # (bs,) -> (,) compute average over mini batch
            return vander_loss

        return vl_2

    # for fourth model
    if i_model == 3:
        def vl_3(y_true, y_pred):
            nm = y_pred.shape[-1]
            vander = y_true[:, :, nm:]
            inliers_prob = 1.0 - K.sigmoid(y_pred)
            bs = inliers_prob.shape[0]
            vander_dim = vander.shape[-1]
            npps = vander.shape[1]
            nm = inliers_prob.shape[-1]

            weights = inliers_prob  # --> (bs, npps, nm)

            # add small random values to produce high vander loss in case of too few inliers
            weights = weights + K.random_uniform((bs, npps, nm), 0, 1e-9)  # --> (bs, npps, nm)
            # normalize weights: norm = 1 for all models
            weights = tf.linalg.normalize(weights, axis=1)[0]

            weights_model = tf.slice(weights, [0, 0, i_model], [bs, npps, 1])
            weights_model = K.reshape(weights_model, shape=(bs, npps))
            weights_model = tf.linalg.diag(weights_model)
            weighted_vander = K.batch_dot(weights_model, vander)

            s = tf.linalg.svd(weighted_vander, compute_uv=False)
            last_svs = s[:, vander_dim - num_sing_vals: vander_dim]

            vander_loss = K.mean(last_svs, axis=1)  # (bs,1) -> (bs,) (penso)
            vander_loss = K.mean(vander_loss)  # (bs,) -> (,) compute average over mini batch
            return vander_loss

        return vl_3


# similarity loss
def sl(y_true, y_pred):
    """
    inliers prob is W
    sim_loss = log(1 + ||W^TW-I||)   (nm,npps)*(npps,nm) = (nm,nm)
    where I is an nm*nm matrix
    """
    inliers_prob = 1.0 - K.sigmoid(y_pred)

    nm = inliers_prob.shape[2]

    w = inliers_prob
    # add small random values to produce high sim loss in case of too few inliers
    w = w + K.random_uniform((w.shape[0], w.shape[1], nm), 0, 1e-9)  # --> (bs, 256, number_of_models)
    w = tf.linalg.normalize(w, axis=1)[0]

    wt = tf.transpose(w, perm=[0, 2, 1])
    wtw = K.batch_dot(wt, w)
    ones = [1 for _ in range(nm)]
    i = tf.linalg.diag(K.constant(ones))
    sim_matrix = wtw - i
    sim_loss = tf.linalg.norm(sim_matrix, ord='fro', axis=[-2, -1])
    log_sim_loss = tf.math.log(1 + sim_loss)
    return log_sim_loss


# variance loss
def variance_outer(i_model):
    if i_model == -1:
        def variance(y_true, y_pred):
            inliers_prob = 1.0 - K.sigmoid(y_pred)

            # inliers loss
            inliers_mean_per_model = - K.mean(inliers_prob, axis=1)   # (bs, npps, nm) -> (bs, nm)

            # compute average inliers mean
            avg_inliers_mean = K.mean(inliers_mean_per_model, axis=-1)  # (bs,)

            # compute variance as sum(||w_i|| - avg_inliers_mean)^2 / (nm -1)
            nm = inliers_prob.shape[-1]
            sum = 0
            for i in range(nm):
                sum += K.pow(inliers_mean_per_model[:, i] - avg_inliers_mean, 2)

            var = sum / (nm-1)
            return var
        return variance


# complete multimodel loss
def mm_loss(inliers_lambda: float,
            vander_lambda: float,
            sim_lambda: float,
            var_lambda: float,
            classtype: ClassValues):
    """
    This function groups together both the 1° and 2° formulation of the multimodel loss
    1° version of multimodel loss:
        1. compute for each model: inliers_lambda * inliers_loss + vander_lambda * vander_loss
        2. compute mean performance over models of step 1.
        3. sum sim_lambda*sim_loss to reward orthogonality

    2° version of multimodel loss:
        1. use Probst formulation (loss = inliers_lambda * inliers_loss + vander_lambda * vander_loss)
        (this is because the monomial basis used in the vandermonde matrix has been computed by multiplying
        all models together -> the problem becomes consensus maximization)

    :param vander: K.tensor, (bs, npps, monomial basis dim).
                    monomial basis dim is, for example, 4 if you're fitting a single circle
                     --> len([x^2+y^2, x, y, 1]) = 4
    :param inliers_lambda: float, weight of inliers loss
    :param vander_lambda: float, weight of vander loss
    :param sim_lambda: float, weight of similarity loss
    :param var_lambda: float, weights of variance loss
    :param classtype:
    :return: unsupervised loss for multimodel fitting
    """

    def keras_loss(y_true, y_pred):
        """

        :param y_true:
        :param y_pred: (bs,npps,nm)
        :return:
        """

        # inliers loss
        inliers_loss = il_outer(-1)(y_true, y_pred)

        # variance loss
        if y_pred.shape[-1] != 1 and var_lambda != 0:
            variance_loss = variance_outer(-1)(y_true, y_pred)
        else:
            variance_loss = 0

        # vander loss
        vander_loss = vl_outer(-1, classtype=classtype)(y_true, y_pred)

        # similarity loss
        if y_pred.shape[-1] != 1 and sim_lambda != 0:
            sim_loss = sl(y_true, y_pred)
        else:
            sim_loss = 0

        return inliers_lambda * inliers_loss + vander_lambda * vander_loss + sim_lambda * sim_loss + var_lambda * variance_loss
    return keras_loss


def crossentropy_loss(y_true, y_pred):
    """
    crossentropy single model
    dice loss multi model
    :param y_true:
    :param y_pred: (bs, npps, nm)
    :return:
    """
    print("crossentropy loss")

    bs = y_pred.shape[0]
    npps = y_pred.shape[1]

    nm = y_pred.shape[2]
    logits = 1 - K.sigmoid(y_pred)
    labels = 1 - y_true[..., 0:nm]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)

    return loss




