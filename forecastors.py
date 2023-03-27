import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from typing import List, Tuple
import math

class Forecastor:
    def __init__(
        self,
        data_vect: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        optim: str = "adam",
        loss: str = "mse",
        activation: str = "relu",
        verbose: int = 0,
        n_features: int = 1,
    ) -> None:
        self.data_vect_raw = data_vect
        self.labels = labels
        self.epochs = epochs
        self.optimizer = optim
        self.loss = loss
        self.activation = activation
        self.verbose = verbose
        self.n_features = n_features
        self.window_size = data_vect.shape[1]

    @staticmethod
    def _handle_vect_shape(data_vect: np.ndarray, n_features: int, is_conv: bool):
        if not is_conv:
            return data_vect.reshape(
                (data_vect.shape[0], data_vect.shape[1], n_features)
            )
        kernel_len = data_vect.shape[1] ** (1 / 2)
        return data_vect.reshape(
            (data_vect.shape[0], kernel_len, kernel_len, n_features)
        )

    def vanilla_lstm(self, cell_size: int = 50) -> Sequential:
        data_vect = self._handle_vect_shape(
            self.data_vect_raw, self.n_features, is_conv=False
        )
        model = Sequential()
        model.add(
            LSTM(
                cell_size,
                activation=self.activation,
                input_shape=(self.window_size, self.n_features),
            )
        )
        model.add(Dense(self.n_features))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.fit(data_vect, self.labels, epochs=self.epochs, verbose=self.verbose)
        return model

    def stacked_lstm(self, n_layers: int, cell_size: int = 50) -> Sequential:
        if n_layers < 2:
            raise ValueError("n_layers must be greater than or equal to 2")

        data_vect = self._handle_vect_shape(
            self.data_vect_raw, self.n_features, is_conv=False
        )
        model = Sequential()
        model.add(
            LSTM(
                cell_size,
                activation=self.activation,
                return_sequences=True,
                input_shape=(self.window_size, self.n_features),
            )
        )
        for _ in range(n_layers - 2):
            model.add(
                LSTM(cell_size, activation=self.activation, return_sequences=True)
            )
        model.add(LSTM(cell_size, activation=self.activation))
        model.add(Dense(self.n_features))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.fit(data_vect, self.labels, epochs=self.epochs, verbose=self.verbose)
        return model

    def bidirectional_lstm(self, cell_size: int = 50) -> Sequential:
        data_vect = self._handle_vect_shape(
            self.data_vect_raw, self.n_features, is_conv=False
        )
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(cell_size, activation=self.activation),
                input_shape=(self.window_size, self.n_features),
            )
        )
        model.add(Dense(self.n_features))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.fit(data_vect, self.labels, epochs=self.epochs, verbose=self.verbose)
        return model

    def cnn_lstm( self, cell_size: int = 50, filters: int = 64, kernel_size: int = 1) -> Sequential:
        n = math.sqrt(self.data_vect_raw.shape[1])
        if n % 1 != 0:
            raise ValueError("Please preapare your data as nxn matrix")

        data_vect = self._handle_vect_shape(
            self.data_vect_raw, self.n_features, is_conv=True
        )
        model = Sequential()
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=filter, kernel_size=kernel_size, activation=self.activation
                ),
                input_shape=(None, n, self.n_features),
            )
        )
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(cell_size, activation=self.activation))
        model.add(Dense(self.n_features))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.fit(data_vect, self.labels, epochs=self.epochs, verbose=self.verbose)
        return model

    def forecast_one_step(self, model: Sequential, model_input: np.ndarray, isConv: bool = False) -> np.ndarray:
        x_input = np.array(model_input)
        if not isConv:
            x_input = x_input.reshape((1, self.window_size, 1))
        else:
            n = math.sqrt(self.data_vect_raw.shape[1])
            x_input = x_input.reshape((1, n, n, 1))
        yhat = model.predict(x_input, verbose=0)
        return yhat
