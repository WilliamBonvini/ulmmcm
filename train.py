from metrics import *
from model.mmpnet import get_mmpn
from utils.custom_callbacks import *
from utils.enums import TrainParams, ClassValues, Models
from utils.utils import split_train_valid
from utils.loader import config_data
from datetime import datetime

"""
In this script you'll be able to set hyperparameters (user defined variables) freely and train the model.
Files generated:
- results/DATETIME/imgs folder:
        contains images of the predictions, computed each 5 epochs and when training is over.
- results/DATETIME/best_weights_*.pkl & results/DATETIME/best_optimizer_*.pkl:
        checkpoints of the optimizer and the weights whenever they provide a better performance wrt the last one.
- results/DATETIME/best_epoch.csv:
        contains best computed metrics and loss terms values during training.
- results/DATETIME/test_performance.csv:
        contains performance on test dataset if you choose to test directly after training (testAtEndOfTraining = True)
"""


if __name__ == "__main__":

    # user defined variables
    noise_range = TrainParams.NOISE_PERCENTAGE_1  # choose noise percentage among values in TrainParams
    outliers_range = TrainParams.ALL_OUTLIERS_RATES   # choose outliers range among values in TrainParams
    inliers_lambda, vander_lambda, sim_lambda, var_lambda = hps = (0.1, 1.0, 1.0, 0.1)  # hyper-parameters of model
    bs = 64  # batch size
    lr = 0.001  # learning rate
    class_type = ClassValues.CIRCLES  # choose among ClassValues.CIRCLES and ClassValues.HOMOGRAPHIES
    nm = 2
    # number of models to fit
    epochs = 6  # number of epochs to run
    testAtEndOfTraining = True  # True if you want to test the model at the end of training with unseen data

    # fixed value variables, do not modify
    model = Models.MMPN
    is_loss_v1 = True
    ns = 1024 if class_type == ClassValues.CIRCLES else 2048  # number of samples
    npps = 256 if class_type == ClassValues.CIRCLES else 512  # number of points per sample
    n_coords = 2 if class_type == ClassValues.CIRCLES else 4  # number of coordinates of point cloud

    data, labels = config_data(class_type=class_type,
                               nm=nm,
                               noise_range=noise_range,
                               outliers_range=outliers_range,
                               ns=ns,
                               npps=npps,
                               train_or_test="train")

    # split train from validation
    train_data, train_labels, valid_data, valid_labels = split_train_valid(data, labels, train_ratio=0.75)

    nn = get_mmpn(bs=bs,
                  npps=npps,
                  hps=hps,
                  nm=nm,
                  is_loss_v1=is_loss_v1,
                  class_type=class_type,
                  lr=lr)

    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y-%H-%M")
    results_path = "results/train/" + now_str
    os.makedirs(results_path)

    a = EpochsSaviourCallback(filename=results_path + "/epochs.csv")
    b = StateCallback(valid_d=valid_data, draw=True, path=results_path)
    c = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    nn.fit(x=train_data[:256], y=train_labels[:256],
           validation_data=(valid_data[:256], valid_labels[:256]),
           batch_size=bs,
           initial_epoch=0,
           epochs=epochs,
           verbose=1,
           callbacks=[a, b, c])

    if testAtEndOfTraining:
        tdata, tlabels = config_data(class_type=class_type,
                                     nm=nm,
                                     noise_range=noise_range,
                                     outliers_range=outliers_range,
                                     ns=ns,
                                     npps=npps,
                                     train_or_test="test")

        from utils.utils import save_perfs_and_plot_preds

        save_perfs_and_plot_preds(data=tdata[:512],
                                  labels=tlabels[:512],
                                  model=nn,
                                  plot=False,
                                  path=results_path)