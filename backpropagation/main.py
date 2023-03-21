import pandas as pd
import numpy as np
import random
import itertools

from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neuron import Neuron
from neuron_layer import NeuronLayer
from neuron_network import NeuronNetwork
random.seed(1782152)


def print_network(network):
    """
    takes a neuron network and prints the name, inputs and the output.
    :param neuron: gets a neuron with n weights.
    :return: prints a grid of possible inputs and the output.
    """
    num_weights = len(network.neuron_layers[0].neurons[0].weights)
    print(f"{network}\n{network.name}")
    for w in range(num_weights):
        print(f"in{w + 1} ", end="")
    print("| out")
    for out in itertools.product(range(2), repeat=num_weights):
        input_str = "   ".join(str(i) for i in out)
        print(f"{input_str}   | {network.activate(out)}")
    print("")


def create_network(neurons_list, amount_weights, network_name):
    """
    creates and sets up a neural network with specified layers, neurons and weights.
    :param layers: the amount of layers the network needs
    :param neurons_list: the amount of neurons in each layer, given as a list
    :param amount_weights: the amount of weights each neuron has in each layer, given as a list
    :param network_name: the name of the network
    :return: creates and returns a neural network
    """
    layers = len(neurons_list)
    neuron_layers = []
    for l in range(layers):
        neurons = []
        for n in range(neurons_list[l]):
            neurons.append(Neuron([random.uniform(-1, 1) for _ in range(amount_weights[l])],
                                  random.uniform(-1, 1), "L"+str(l)+"N"+str(n)))
        neuron_layers.append(NeuronLayer(neurons, "Layer"+str(l)))
    return NeuronNetwork(neuron_layers, str(network_name))


"""creates an "and" gate as a neural network,
creates the inputs and targets for the "and_network"."""
and_network = create_network([1], [2], "and_network")
inputs_and = list(itertools.product([0, 1], repeat=2))
target_and = [0, 0, 0, 1]

"""prints the network before training, trains the network, and prints the network after training."""
print_network(and_network)
and_network.train(inputs_and, target_and, 0.001, 10000, 0.5)
print("")
print_network(and_network)
print("------------------------------------------------------------------------")




"""creates an "xor" gate as a neural network,
creates the inputs and targets for the "xor_network"."""
xor_network = create_network([2, 1], [2, 2], "xor_network")
inputs_xor = list(itertools.product([0, 1], repeat=2))
target_xor = [0, 1, 1, 0]

"""prints the network before training, trains the network, and prints the network after training."""
print_network(xor_network)
xor_network.train(inputs_xor, target_xor, 0.001, 10000, 0.5)
print("")
print_network(xor_network)
print("------------------------------------------------------------------------")




"""creates a "half_adder" neural network,
creates the inputs and targets for the "half_adder"."""
half_adder = create_network([3, 2], [2, 3], "half_adder")
target_half_adder = [[0,0], [0,1], [0,1], [1,0]]
inputs_half_adder = list(itertools.product([0, 1], repeat=2))

"""prints the network before training, trains the network, and prints the network after training."""
print_network(half_adder)
half_adder.train(inputs_half_adder, target_half_adder, 0.01, 10000, 0.2)
print("")
print_network(half_adder)
print("------------------------------------------------------------------------")




"""loads in the iris data set and convert it to a dataframe"""
iris = load_iris()
df_iris = pd.DataFrame(data=np.c_[iris["data"], iris["target"]],
                     columns= iris["feature_names"] + ["target"])

"""creates a "iris_network" neural network,
creates the inputs and targets for the "iris_network"."""
iris_network = create_network([4, 3], [4, 4], "iris_network")
targets_iris = []
for i in range(len(iris["target"])):
    if iris["target"][i] == 0.0:
        targets_iris.append([1, 0, 0])
    elif iris["target"][i] == 1:
        targets_iris.append([0, 1, 0])
    else:
        targets_iris.append([0, 0, 1])
inputs_iris = [list(row[:4]) for row in df_iris.values]
"""as this network is a lot harder to print due to the many more datapoints, i won't be printing it in the console."""


"""splits the iris dataset into a train and test version."""
train_inputs_iris, test_inputs_iris, train_targets_iris, test_targets_iris = train_test_split(inputs_iris,
                                                            targets_iris, test_size=.4, random_state=1782152)
print(f"amount of training samples: {len(train_inputs_iris)}, amount of test samples: {len(test_targets_iris)}")


"""trains the "iris_network"."""
iris_network.train(train_inputs_iris, train_targets_iris, 0.001, 1000, 0.5)


"""saves the results of how good the "iris_network" preforms on the test set."""
results_iris = [iris_network.collapse_activate(test_inputs_iris[row]) for row in range(len(test_inputs_iris))]
"""sums the amount of correct answers the network gives."""
correct_iris = sum(1 for i in range(len(test_targets_iris)) if results_iris[i] == test_targets_iris[i])
"""prints the ratio of correct and wrong answers on the train set."""
print(correct_iris/len(test_targets_iris))
print("")
print("------------------------------------------------------------------------")




"""loads in the digits data set and convert it to a dataframe"""
digits = load_digits()
df_digits = pd.DataFrame(data=np.c_[digits["data"], digits["target"]],
                     columns= digits["feature_names"] + ["target"])

"""creates a "digits_network" neural network,
creates the inputs and targets for the "digits_network"."""
digits_network = create_network([30, 10], [64, 30], "digits_network")
targets_digits = []
for i in range(len(digits["target"])):
    for j in range(10):
        if digits["target"][i] == j:
            targets_digits.append(([0]*j+[1]+[0]*(9-j)))
inputs_digits = [list(row[:64]) for row in df_digits.values]
"""as this network is a lot harder to print due to the many more datapoints, i won't be printing it in the console."""


"""splits the  digits dataset into a train and test version."""
train_inputs_digits, test_inputs_digits, train_targets_digits, test_targets_digits = train_test_split(inputs_digits,
                                                            targets_digits, test_size=.4, random_state=1782152)
print(f"amount of training samples: {len(train_inputs_digits)}, amount of test samples: {len(test_targets_digits)}")


"""trains the "digits_network"."""
"""THIS TAKES A WHILE, to train on a 100 epochs it takes multiple minutes
the accuracy using 10 epochs is ~88.4%, using 100 epochs is ~96.5%"""
digits_network.train(train_inputs_digits, train_targets_digits, 0.001, 100, 0.1)


"""saves the results of how good the "digits_network" preforms on the test set."""
results_digits = [digits_network.collapse_activate(inputs_digits[row]) for row in range(len(inputs_digits))]
"""sums the amount of correct answers the network gives."""
correct_digits = sum(1 for i in range(len(targets_digits)) if results_digits[i] == targets_digits[i])
"""prints the ratio of correct and wrong answers on the train set."""
print(f"{correct_digits/len(targets_digits)}")
print("")
print("------------------------------------------------------------------------")
