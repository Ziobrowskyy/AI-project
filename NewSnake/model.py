import pygad
import numpy
from tensorflow import keras
import pygad.kerasga

from NewSnake.game import SnakeGame

BOARD_H, BOARD_W = 20, 20

# input_layer = keras.layers.Input(BOARD_W * BOARD_H)
# input_layer = keras.layers.Input(shape=(BOARD_W, BOARD_H))

model = keras.Sequential([
    keras.layers.Input(shape=(20, )),
    # keras.layers.GlobalAveragePooling2D(),
    keras.layers.Flatten(),
    # keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(64, activation="linear"),
    keras.layers.Dense(3, activation="softmax"),
])
model.build()
model.summary()
# model.add(input_layer)
# model.add(dense_layer1)
# model.add(dense_layer2)
# model.add(output_layer)

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)


def fitness_func(solution, sol_idx):
    # global data_inputs, data_outputs, keras_ga, model
    global keras_ga, model

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)

    snake = SnakeGame(BOARD_W, BOARD_H)
    game_score = 0
    while True:
        data_inputs = snake.get_model_data()
        predictions = model.predict(data_inputs)
        print(predictions.shape)
        predictions = predictions[0] >= predictions[0].max()
        is_alive, step_score, game_score = snake.play_step(predictions)
        if not is_alive:
            break

    # mae = keras.losses.MeanAbsoluteError()
    # solution_fitness = 1.0 / (mae(data_outputs, predictions).numpy() + 0.00000001)
    solution_fitness = game_score
    return solution_fitness


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


num_generations = 250
num_parents_mating = 5
initial_population = keras_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

ga_instance.run()
