import pandas as pd
import numpy as np
import random
import itertools
from perceptron import Perceptron
from sklearn.datasets import load_iris
random.seed(1782152)


def print_perceptron(perceptron):
    """
    takes a perceptron with 2 weights and prints the name, input1, input2 and the output.
    :param perceptron: gets a perceptron with 2 weights.
    :return: prints a 4 by 3 grid of possible inputs and the output.
    """
    print(f"{perceptron}\n{perceptron.name}\nin1 in2 | out")
    for i, j in itertools.product(range(2), repeat=2):
        print(f"{i}   {j}   | {perceptron.activate([i, j])}")
    print("")


def train_perceptron(perceptron, activation, error_threshold=0.05):
    """
    takes a perceptron with 2 weights and generates all possible inputs, [(0,0), (0,1), (1,0), (1,1)].
    calls the perceptron.update function on these inputs and keeps count of the amount of loops.
    if the perceptron reaches its error_threshold, stops the loop early, otherwise, keep going till 1000.
    :param perceptron: gets a perceptron with 2 weights.
    :param activation: gets a list of targets for each input.
    :param error_threshold: creates an early stopping value for the while loop, (default 0.05).
    :return: trains the perceptron and prints out some information when the training is done, doesn't return anything.
    """
    p_input = list(itertools.product([0, 1], repeat=2))
    stop = False
    count = 0
    while not stop:
        count += 1
        if perceptron.update(p_input, activation, error_threshold):
            print(f"finished training this perceptron in {count} loops")
            stop = True
        if count == 10:
            print(f"couldn't train this perceptron in {count} loops")
            stop = True
    print("")


def train_iris_perceptron(perceptron, p_input, activation, error_threshold=0.05):
    """
    takes a perceptron with an unspecified amount of weights and inputs, and trains it.
    calls the perceptron.update function on these inputs and keeps count of the amount of loops.
    if the perceptron reaches its error_threshold, stops the loop early, otherwise, keep going till 1000.
    :param perceptron: gets a perceptron with an unspecified amount of weights.
    :param p_input: gets a list of inputs for each weight of the perceptron.
    :param activation: gets a list of targets for each input.
    :param error_threshold: creates an early stopping value for the while loop, (default 0.05).
    :return: trains the perceptron and prints out some information when the training is done, doesn't return anything.
    """
    stop = False
    count = 0
    while not stop:
        count += 1
        if perceptron.update(p_input, activation, error_threshold):
            print(f"finished training this perceptron in {count} loops")
            stop = True
        if count == 1000:
            print(f"couldn't train this perceptron further in {count} loops")
            stop = True
    print("")


"""creates "and" and "xor" perceptrons and their target values."""
p_and_input = list(itertools.product([0, 1], repeat=2))
p_and_target = [0, 0, 0, 1]
p_and = Perceptron([random.uniform(-1, 1) for _ in range(2)],
                   random.uniform(-1, 1),
                   "AND")
# p_and = Perceptron([-0.5, 0.5], 1.5, "AND")
p_xor_input = list(itertools.product([0, 1], repeat=2))
activation_xor = [0, 1, 1, 0]
p_xor = Perceptron([random.uniform(0, 1) for _ in range(2)],
                   random.uniform(-1, 1),
                   "XOR")


print_perceptron(p_and)

p_and.train(p_and_input, p_and_target, 0.05, 100, 0.1)

print_perceptron(p_and)

# """prints the "before training" output of the perceptrons, trains them, and then prints the "after training" output."""
# print_perceptron(p_and)
# train_perceptron(p_and, activation_and)
# print_perceptron(p_and)
# print("----------------------------------------------------\n")
# print_perceptron(p_xor)
# train_perceptron(p_xor, activation_xor)
# print_perceptron(p_xor)
# print("----------------------------------------------------\n")


# """loads in the iris dataset, converts it to a pandas DataFrame (df)."""
# iris = load_iris()
# df_iris = pd.DataFrame(data=np.c_[iris["data"], iris["target"]],
#                      columns= iris["feature_names"] + ["target"])
# """splits the iris df into 2 separate dfs, 1 containing "setosa" and "versicolor", the other containing:
# "setosa" and "virginica".
# creates 2 "iris" perceptrons that have 4 weights and 1 bias.
# as well as well as formatting all the inputs to a list of tuples, and the targets as a list of ints, for each perceptron."""
# df_iris1 = df_iris.loc[df_iris["target"] != 2.0]
# p_iris1 = Perceptron([random.uniform(0, 1) for _ in range(4)],
#                      random.uniform(-1, 1),
#                      "p_iris1")
# iris_input1 = [tuple(row[:4]) for row in df_iris1.values]
# iris_activation1 = [int(row[4]) for row in df_iris1.values]
#
# df_iris2 = df_iris.loc[df_iris["target"] != 1.0]
# p_iris2 = Perceptron([random.uniform(0, 1) for _ in range(4)],
#                      random.uniform(-1, 1),
#                      "p_iris2")
# iris_input2 = [tuple(row[:4]) for row in df_iris2.values]
# iris_activation2 = [int(row[4]) for row in df_iris2.values]
#
#
# """trains both iris perceptrons in their respected inputs and targets,
# using a stricter threshold compared to the and and xor perceptrons."""
# train_iris_perceptron(p_iris1, iris_input1, iris_activation1, 0.01)
# train_iris_perceptron(p_iris2, iris_input2, iris_activation2, 0.01)
# print("----------------------------------------------------\n")
#
#
# """adds the results and the amount of correct classifications to 2 lists,
#  and calculates the percentage of correct classifications.
#  also prints the name, final bias and final weights of the perceptrons."""
# results1 = [p_iris1.activate(row[:4]) for row in df_iris1.values]
# correct1 = sum(1 for i in range(len(iris_activation1)) if results1[i] == iris_activation1[i])
# print(p_iris1)
# print(f"percentage correct = {correct1/len(iris_activation1)*100}%")
# print("\n----------------------------------------------------\n")
#
#
# """because the target for the "virginica" is 2, and the perceptrons currently use the step_function,
# it's impossible to get a 2 as the target output.
# there are a few ways to "fix" this, one of them being, changing the step_function output from 1 to 2.
# because for this assignment I only want to show the accuracy of the training model,
# I chose to multiply the result by 2, this changes a 1 into a 2, but keeps a 0 as a 0."""
# results2 = [p_iris2.activate(row[:4])*2 for row in df_iris2.values]
# correct2 = sum(1 for i in range(len(iris_activation2)) if results2[i] == iris_activation2[i])
# print(p_iris2)
# print(f"percentage correct = {correct2/len(iris_activation2)*100}%")
#
#
# """in conclusion:
# 3.a. the final parameters of the and perceptron are:
#     weights = [2.099383380832772, 0.8017412383211584], bias = -2.124479997506068
#
# 3.b. the final parameters of the xor perceptron are:
#     weights = [4.494618387453568, 4.13285886300068], bias = -1.834556204282468
#     but obviously it doesn't work, as it needs more than 1 layer to function.
#
# 3.c.i. the final parameters of the iris1 perceptron are:
#     weights = [7.622027044609713, 7.2885968113681345, 7.58056793096198, 7.4579294510930305], bias = -85.0322175923695
#
# 3.c.ii. the final parameters of the iris2 perceptron are:
#     weights = [3.2551122422360015, 3.662541654494767, 3.3763164577640734, 4.004124690069453], bias = -41.70539527849178
#
#
# interestingly the iris2 perceptron is really quick to learn, even with a higher error threshold,
# getting a 100% accuracy on the "setosa" and "virginica" data in only a few runs and with a lot more certainty.
# looking over this data it's quite clear why, as there is a large difference in sizes for multiple parameters,
# the difference between the "setosa" and "versicolor" data isn't as strong,
# therefore it takes the iris1 perceptron a lot longer to train, and doesn't score a 100% accuracy, which is to be expected.
# """
