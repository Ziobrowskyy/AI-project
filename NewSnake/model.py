import keras.models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


class CNN:
    def __init__(self, board_size: int):
        self.model_path = "/saved-model"
        num_filters = 8
        filter_size = 3
        pool_size = 2
        actions_number = 3

        self.model = Sequential(
            [
                Conv2D(num_filters, filter_size, input_shape=(board_size, board_size, 1)),
                MaxPooling2D(pool_size=pool_size),
                Flatten(),
                Dense(actions_number, activation='softmax'),
            ]
        )
        self.model.compile(
            "adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def save_model(self):
        self.model.save(self.model_path)

    def import_model(self):
        self.model = keras.models.load_model(self.model_path)

    def train(self, data):
        self.model.fit()

    def next_gen(self):
        # 1. get models with best fitness (maybe 2 models)
        # 2. perform crossover between them
        # 3. add mutation to get new batch of models
        # 4. repeat
        # 5. profit
        pass

    def get_actions(self, game_state: [[int]]):
        p = self.model.predict(game_state)
        print(p)
        return p
