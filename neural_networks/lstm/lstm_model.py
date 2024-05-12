import tensorflow as tf
from keras.api.activations import sigmoid, tanh
from keras.api.layers import Dense, BatchNormalization, ReLU, concatenate, Add, LeakyReLU, ELU, \
    Activation, LayerNormalization, Dropout, LSTM
from keras import Input, Model, callbacks
import keras_tuner
from dataclasses import fields

from neural_networks.lstm.default_blocks import create_dense_block, create_lstm_block, ModelParamsVars, ModelParams


def create_lstm_model(hp,
                      input_size,
                      output_size,
                      initializer,
                      params: ModelParamsVars):
    bidirectional = hp.Boolean("bidirectional")
    residual = hp.Boolean("residual")

    inputs = Input(input_size)
    normalization = hp.Choice("normalization", params.normalization)
    if normalization == "batch":
        x = BatchNormalization(axis=[1, 2])(inputs)
    elif normalization == "layer":
        x = LayerNormalization()(inputs)
    else:
        x = inputs

    layers_outputs = [x]
    num_of_blocks = hp.Int(name="num_of_blocks",
                           min_value=params.num_of_blocks[0],
                           max_value=params.num_of_blocks[1])
    normalization = hp.Choice("normalization", params.normalization)
    dropout = hp.Choice("dropout", params.dropout)
    for i in range(num_of_blocks - 1):
        x = create_lstm_block(x,
                              params.layers_size,
                              normalization,
                              initializer,
                              params.regularization_coef,
                              bidirectional)
        if residual:
            layers_outputs.append(x)
            x = concatenate(layers_outputs)
        if dropout > 0:
            x = Dropout(rate=dropout)(x)
    x = create_lstm_block(x,
                          params.layers_size,
                          normalization,
                          initializer,
                          params.regularization_coef,
                          bidirectional,
                          final=True)
    if dropout > 0:
        x = Dropout(rate=dropout)(x)
    activation = hp.Choice("activation", params.activation)
    x = create_dense_block(x, params.layers_size, activation, normalization, initializer, params.regularization_coef)
    outputs = Dense(output_size)(x)
    model_name = f"lstm_nbk_{num_of_blocks}_size_{params.layers_size}_a_{activation}_norm_{normalization}"
    if bidirectional:
        model_name += "_bidir"
    if residual:
        model_name += "_res"
    return Model(inputs, outputs, name=model_name)


def dense_net_static(input_size, output_size,
                     initializer,
                     params: ModelParams,
                     direction: bool = False):
    inputs = Input(input_size)

    num_of_blocks = params.num_of_blocks
    normalization = params.normalization
    dropout = params.dropout
    if normalization == "batch":
        x = BatchNormalization(axis=[1, 2])(inputs)
    elif normalization == "layer":
        x = LayerNormalization()(inputs)
    else:
        x = inputs

    layers_outputs = [x]

    for i in range(num_of_blocks - 1):
        x = create_lstm_block(x, params.layers_size, normalization, initializer, params.regularization_coef)
        layers_outputs.append(x)
        x = concatenate(layers_outputs)
        if dropout > 0:
            x = Dropout(rate=dropout)(x)
    x = create_lstm_block(x, params.layers_size, normalization, initializer, params.regularization_coef, final=True)
    if dropout > 0:
        x = Dropout(rate=dropout)(x)
    activation = params.activation
    x = create_dense_block(x, params.layers_size, activation, normalization, initializer, params.regularization_coef)
    outputs = Dense(output_size)(x)
    if direction:
        outputs = Activation(tanh)(outputs)

    name = "lstm_DenseNet"
    for field in fields(ModelParams):
        p = field.name
        v = getattr(params, p)
        name += f"_{p}_{v}"
    return Model(inputs, outputs, name=name)
