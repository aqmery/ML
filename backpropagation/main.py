from neuron import Neuron
from neuron_layer import NeuronLayer
from neuron_network import NeuronNetwork
import itertools
import random


def print_neuron1(neuron):
    """
    takes a neuron with 1 weight and prints the name, input1 and the output.
    :param neuron: gets a neuron with 1 weights.
    :return: prints a 2 by 2 grid of possible inputs and the output.
    """
    print(f"{neuron}\n{neuron.name}\nin1 | out")
    for i in range(2):
        print(f"{i}   | {neuron.activate([i])}")
    print("")


def print_neuron2(neuron):
    """
    takes a neuron with 2 weights and prints the name, input1, input2 and the output.
    :param neuron: gets a neuron with 2 weights.
    :return: prints a 4 by 3 grid of possible inputs and the output.
    """
    print(f"{neuron}\n{neuron.name}\nin1 in2 | out")
    for i, j in itertools.product(range(2), repeat=2):
        print(f"{i}   {j}   | {neuron.activate([i, j])}")
    print("")


def print_neuron3(neuron):
    """
    takes a neuron with 3 weights and prints the name, input1, input2, input3 and the output.
    :param neuron: gets a neuron with 3 weights.
    :return: prints an 8 by 4 grid of possible inputs and the output.
    """
    print(f"{neuron}\n{neuron.name}\nin1 in2 in3 | out")
    for i, j, k in itertools.product(range(2), repeat=3):
        print(f"{i}   {j}   {k}   | {neuron.activate([i, j, k])}")
    print("")


def train_neuron(neuron, activation):
    inputs = list(itertools.product([0, 1], repeat=2))
    for j in range(1):
        for sample, target in zip(inputs, activation):
            # print(sample, target)
            neuron.update(sample, target)
            neuron.hidden_error()
            # print(neuron)

def train_neuron_network(neuron_network, activation):
    inputs = list(itertools.product([0, 1], repeat=2))
    for j in range(1):
        for sample, target in zip(inputs, activation):
            neuron_network.update(sample, target)
            print("")

# n_and = Neuron([-0.5, 0.5], 1.5, "AND")
# activation_and = [0, 0, 0, 1]
#
# print_neuron2(n_and)
#
# train_neuron(n_and, activation_and)
#
# print_neuron2(n_and)


aa1 = Neuron([2.1, 2.2], 0, "aa1")
aa2 = Neuron([1.9, 2.0], 0, "aa2")
bb1 = Neuron([1.7, 1.8], 0, "bb1")
bb2 = Neuron([1.5, 1.6], 0, "bb2")
m1 = Neuron([1.0, 1.0], 0, "m1")
n1 = Neuron([1.0, 1.0], 0, "n1")
f2 = Neuron([0.0, 0.1], 0, "f2")
g2 = Neuron([0.2, 0.3], 0, "g2")
h2 = Neuron([0.4, 0.5], 0, "h2")
s3 = Neuron([0.6, 0.7, 0.8], 0, "s3")
c3 = Neuron([0.9, 1.0, 1.1], 0, "c3")


n_la = NeuronLayer([aa1, aa2], "layer aa")
n_lb = NeuronLayer([bb1, bb2], "layer bb")
n_l0 = NeuronLayer([m1, n1], "input layer 2n")
n_l1 = NeuronLayer([f2, g2, h2], "hidden layer 3n")
n_l2 = NeuronLayer([s3, c3], "output layer 2n")


half_adder = NeuronNetwork([n_la, n_lb, n_l0, n_l1, n_l2], "half_adder")
activation_half_adder = [[0,0], [1,0], [1,0], [0,1]]


# print(f2.activate([1,1]))
# print("")
# print(half_adder.activate([1,1]))
# half_adder.activate([1, 1])
half_adder.backprop([1, 1], activation_half_adder[3])
# half_adder.calculate_error([1,1], activation_half_adder[3])
print("")
print("")
print("half_adder.error", half_adder.error)

# print(half_adder.neuron_layers[1].neurons[0].weights)
# print(half_adder.neuron_layers[1].neurons[0].weightchange)


