import numpy as np
import tensorflow as tf
from keras.api.losses import MeanSquaredError
from keras.api.optimizers import Adam, RMSprop, SGD
from keras.api.activations import sigmoid
from keras.api.layers import Dense, BatchNormalization, ReLU, concatenate, Add, LeakyReLU, ELU, Activation, \
    LayerNormalization, Dropout
from keras import Input, Model, callbacks
from keras.api.initializers import RandomNormal, GlorotNormal, HeUniform
import keras_tuner
from neural_networks.lstm.lstm_model import create_lstm_model
from neural_networks.lstm.default_blocks import *

# data_path = "/home/workpc/datasets/stocks/stocks_agg"
data_path = "/home/ubuntu/datasets/stocks_new/train_data/"


# tf.keras.backend.set_floatx('float16')


def lr_schedule(epoch):
    if epoch > 100:
        learning_rate = 1e-6
    elif epoch > 50:
        learning_rate = 1e-5
    elif epoch > 25:
        learning_rate = 1e-4
    else:
        learning_rate = 1e-3
    return learning_rate


def build_lstm_model(hp):
    layer_size = hp.Choice("hidden_layer_size", [32, 64, 128, 256])
    # regularization_coef = hp.Float("std", min_value=1e-8, max_value=0.01, step=10, sampling="log")
    regularization_coef = 1e-4
    initializer_type = hp.Choice("initializer_type", ["normal", "glorot", "he_uni"])
    # random_seed = hp.Int("rand_seed", min_value=1, max_value=1000)
    random_seed = 42

    if initializer_type == "glorot":
        initializer = GlorotNormal(seed=random_seed)
    elif initializer_type == "he_uni":
        initializer = HeUniform(seed=random_seed)
    else:
        std = hp.Float("std", min_value=0.0001, max_value=10, step=10, sampling="log")
        initializer = RandomNormal(mean=0., stddev=std, seed=random_seed)

    # starting_lr = hp.Float("std", min_value=1e-6, max_value=0.01, step=10, sampling="log")
    starting_lr = 0.0001
    # optimizer_type = hp.Choice("optimizer", ["adam", "rmsp", "sgd"])
    optimizer_type = "adam"

    if optimizer_type == "adam":
        optimizer = Adam(learning_rate=starting_lr)
    elif optimizer_type == "rmsp":
        optimizer = RMSprop(learning_rate=starting_lr)
    else:
        optimizer = SGD(learning_rate=starting_lr)

    # dropout = [0.9, 0.7, 0.5, 0.3, 0.]
    dropout = [0.5]
    params = ModelParamsVars(num_of_blocks=(1, 3),
                             layers_size=layer_size,
                             activation=["relu", "leacky", "elu", "sigmoid", "mish"],
                             normalization=["batch", "layer"],
                             regularization_coef=regularization_coef,
                             dropout=dropout)

    model = create_lstm_model(hp,
                          input_size=input_size,
                          output_size=output_size,
                          initializer=initializer,
                          params=params)

    model.compile(optimizer=optimizer,
                  loss=MeanSquaredError)
    return model


if __name__ == "__main__":
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # x_train = np.load(f"{data_path}_ffnn/Russia_v2/x_train.npy")
    # y_train = np.load(f"{data_path}_ffnn/Russia_v2/y_range_train.npy")
    # x_val = np.load(f"{data_path}_ffnn/Russia_v2/x_val.npy")
    # y_val = np.load(f"{data_path}_ffnn/Russia_v2/y_range_val.npy")

    x_train = np.load(f"{data_path}train_data.npy")
    y_train = np.load(f"{data_path}train_5d_change.npy")
    x_val = np.load(f"{data_path}val_data.npy")
    y_val = np.load(f"{data_path}val_5d_change.npy")

    input_size = x_train.shape[1:]
    # output_size = y_train.shape[1]
    output_size = 1

    early_stop_callback = callbacks.EarlyStopping(patience=20)
    # lr_callback = callbacks.LearningRateScheduler(lr_schedule)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss',
                                            factor=0.5,
                                            patience=5,
                                            cooldown=5,
                                            min_lr=1e-8)

    # tuner = keras_tuner.BayesianOptimization(hypermodel=build_ffn_model,
    #                                          objective="val_loss",
    #                                          max_trials=600,
    #                                          num_initial_points=100,
    #                                          seed=42,
    #                                          overwrite=False)
    tuner = keras_tuner.RandomSearch(hypermodel=build_lstm_model,
                                     objective="val_loss",
                                     max_trials=30,
                                     seed=42,
                                     overwrite=False)
    tuner.search(x_train, y_train, epochs=400, validation_data=(x_val, y_val), batch_size=512,
                 callbacks=[early_stop_callback, reduce_lr])

    for model in tuner.get_best_models(6):
        model.save(f"models/{model.name}.h5")

    trials = tuner.oracle.get_best_trials(num_trials=30)
    with open("trials.txt", "w") as f:
        for t in trials:
            f.write("\n\n")
            hp = t.hyperparameters.get_config()["values"]
            for p, v in hp.items():
                f.write(f"{p}: {v}\n")
            f.write(f"Score: {t.score}")

