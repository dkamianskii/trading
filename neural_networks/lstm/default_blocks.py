from keras.api.activations import sigmoid, mish
from keras.api.layers import Dense, BatchNormalization, ReLU, LeakyReLU, ELU, Activation, LayerNormalization, Dropout, \
    Multiply, LSTM, Bidirectional
from keras.api.regularizers import L1L2

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ModelParamsVars:
    num_of_blocks: Tuple[int, int]
    layers_size: int
    regularization_coef: float
    activation: List[str]
    normalization: List[str]
    dropout: List[float]


@dataclass
class ModelParams:
    num_of_blocks: int
    layers_size: int
    regularization_coef: float
    activation: str
    normalization: str
    dropout: float


def create_dense_block(input_vector,
                       layer_size,
                       activation,
                       normalization,
                       initializer,
                       regularization_coef):
    x = Dense(layer_size, kernel_initializer=initializer, kernel_regularizer=L1L2(regularization_coef,
                                                                                  regularization_coef))(input_vector)

    if normalization == "batch":
        x = BatchNormalization()(x)
    elif normalization == "layer":
        x = LayerNormalization()(x)

    if activation == "relu":
        x = ReLU()(x)
    elif activation == "leacky":
        x = LeakyReLU()(x)
    elif activation == "elu":
        x = ELU()(x)
    elif activation == "sigmoid":
        x = Activation(sigmoid)(x)
    elif activation == "mish":
        x = Activation(mish)(x)

    return x


def create_lstm_block(input_vector,
                      layer_size,
                      normalization,
                      initializer,
                      regularization_coef,
                      bidirection: bool = False,
                      final: bool = False):
    if bidirection:
        x = Bidirectional(LSTM(layer_size,
                               kernel_initializer=initializer,
                               kernel_regularizer=L1L2(regularization_coef, regularization_coef),
                               return_sequences=(not final)))(input_vector)
    else:
        x = LSTM(layer_size,
                 kernel_initializer=initializer,
                 kernel_regularizer=L1L2(regularization_coef, regularization_coef),
                 return_sequences=(not final))(input_vector)

    if normalization == "batch":
        if not final:
            x = BatchNormalization(axis=[1, 2])(x)
        else:
            x = BatchNormalization()(x)
    elif normalization == "layer":
        x = LayerNormalization()(x)

    return x
