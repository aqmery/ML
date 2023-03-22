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


"""creates "and" and "xor" perceptrons and their target values."""
p_and_input = list(itertools.product([0, 1], repeat=2))
p_and_target = [0, 0, 0, 1]
p_and = Perceptron([random.uniform(-1, 1) for _ in range(2)], random.uniform(-1, 1), "AND")

p_xor_input = list(itertools.product([0, 1], repeat=2))
p_xor_target = [0, 1, 1, 0]
p_xor = Perceptron([random.uniform(-1, 1) for _ in range(2)], random.uniform(-1, 1), "XOR")


"""prints the "before training" output of the perceptrons, trains them, and then prints the "after training" output."""
print_perceptron(p_and)
p_and.train(p_and_input, p_and_target, 0.05, 100, 0.1)
print_perceptron(p_and)
print("----------------------------------------------------\n")

print_perceptron(p_xor)
p_xor.train(p_xor_input, p_xor_target, 0.05, 100, 0.1)
print_perceptron(p_xor)
print("----------------------------------------------------\n")


"""loads in the iris dataset, converts it to a pandas DataFrame (df)."""
iris = load_iris()
df_iris = pd.DataFrame(data=np.c_[iris["data"], iris["target"]],
                     columns= iris["feature_names"] + ["target"])

"""splits the iris df into 2 separate dfs, 1 containing "setosa" and "versicolor", the other containing:
"versicolor" and "virginica".
creates 2 "iris" perceptrons that have 4 weights and 1 bias.
as well as well as formatting all the inputs to a list of tuples, and the targets as a list of ints, for each perceptron."""
df_iris1 = df_iris.loc[df_iris["target"] != 2.0].copy()
p_iris1 = Perceptron([random.uniform(-1, 1) for _ in range(4)], random.uniform(-1, 1), "p_iris1")
iris_input1 = [tuple(row[:4]) for row in df_iris1.values]
iris_target1 = [int(row[4]) for row in df_iris1.values]

df_iris2 = df_iris.loc[df_iris["target"] != 0.0].copy()
"""replaces the output from 1 to 0 and 2 to 1"""
df_iris2["target"].replace({1.0: 0, 2.0: 1}, inplace=True)
p_iris2 = Perceptron([random.uniform(-1, 1) for _ in range(4)], random.uniform(-1, 1), "p_iris2")
iris_input2 = [tuple(row[:4]) for row in df_iris2.values]
iris_target2 = [int(row[4]) for row in df_iris2.values]


"""trains the iris1 perceptron, collects the results, and calculates the % of accurate "guesses"."""
p_iris1.train(iris_input1, iris_target1, 0.05, 100, 0.1)
results1 = [p_iris1.activate(row[:4]) for row in df_iris1.values]
correct1 = sum(1 for i in range(len(iris_target1)) if results1[i] == iris_target1[i])
print(p_iris1)
print(f"percentage correct = {correct1/len(iris_target1)*100}%")
print("\n----------------------------------------------------\n")


"""trains the iris2 perceptron, collects the results, and calculates the % of accurate "guesses"."""
p_iris2.train(iris_input2, iris_target2, 0.05, 100, 0.1)
results2 = [p_iris2.activate(row[:4]) for row in df_iris2.values]
correct2 = sum(1 for i in range(len(iris_target2)) if results2[i] == iris_target2[i])
print(p_iris2)
print(f"percentage correct = {correct2/len(iris_target2)*100}%")
print("\n----------------------------------------------------\n")


"""in conclusion:
3.a. the final parameters of the and perceptron are:
    weights = [0.2525833808327712, 0.15494123832115794], bias = -0.33837999750607023

3.b. the final parameters of the xor perceptron are:
    weights = [-0.05376322509276876, -0.17728227399861682], bias = 0.047543795717545495
    but obviously it doesn't work, as it needs more than 1 layer to function.

3.c.i. the final parameters of the iris1 perceptron are:
    weights = [0.12885408921935237, -0.7780063772638284, 0.6959358619238899, 0.18065890218600233], bias = -0.6678175923610029

3.c.ii. the final parameters of the iris2 perceptron are:
    weights = [-3.8785755155277313, -1.0937166910101628, 4.28383291552843, 4.379449380139247], bias = 0.06980472151137684
"""
