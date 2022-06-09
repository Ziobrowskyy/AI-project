import pygad
from tensorflow import keras
import tensorflow as tf
import pygad.kerasga
import numpy as np

from NewSnake4.game import SnakeGame

BOARD_SIZE = 40

model = keras.Sequential([
    keras.layers.Dense(24, activation="linear", input_shape=(24,)),
    keras.layers.Dense(32, activation="linear"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])
model.build()
model.summary()

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=20)

record = 0


def fitness_func(solution, sol_idx):
    global keras_ga, model, record
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)

    snake = SnakeGame(BOARD_SIZE, BOARD_SIZE)
    while True:
        data_inputs = snake.get_model_data().reshape((1, 24))
        # print(f"inputs: {data_inputs.shape}")
        predictions = model.predict(data_inputs)
        # print(f"predictions: {predictions.shape}")
        is_alive, step_score, game_score = snake.play_step(predictions)
        if not is_alive:
            break
    if game_score > record:
        record = game_score
        print("######### NEW RECORD ##########")
    print(f"record: {record}, score: {game_score}")
    return game_score


def callback_generation(ga_instance):
    if ga_instance.generations_completed % 20 == 0:
        ga_instance.save(f"{ga_instance.generations_completed}")
        # model.save(f"{ga_instance.generations_completed}")
        # model.save_weights(f"{ga_instance.generations_completed}.h5")

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


# Prepare the PyGAD parameters
num_generations = 5000  # Number of generations

# Number of solutions to be selected as parents in the mating pool
num_parents_mating = 5

# Initial population of network weights
initial_population = keras_ga.population_weights
parent_selection_type = "sss"  # Type of parent selection
crossover_type = "single_point"  # Type of the crossover operator
mutation_type = "random"  # Type of the mutation operator

# Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists
mutation_percent_genes = 10

# Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing
keep_parents = -1

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

MODEL_TO_LOAD = None
# MODEL_TO_LOAD = "3020"
if MODEL_TO_LOAD is not None:
    print("load model")
    ga_instance = pygad.load(filename=MODEL_TO_LOAD)

if __name__ == '__main__':
    ga_instance.run()
