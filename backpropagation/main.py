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


# def train_neuron(neuron, activation):
#     inputs = list(itertools.product([0, 1], repeat=2))
#     for j in range(1):
#         for sample, target in zip(inputs, activation):
#             # print(sample, target)
#             neuron.update(sample, target)
#             neuron.hidden_error()
#             # print(neuron)

# def train_neuron_network(neuron_network, activation):
#     inputs = list(itertools.product([0, 1], repeat=2))
#     for j in range(1):
#         for sample, target in zip(inputs, activation):
#             neuron_network.update(sample, target)
#             print("")

# n_and = Neuron([-0.5, 0.5], 1.5, "AND")
# inputs_and = list(itertools.product([0, 1], repeat=2))
# target_and = [0, 0, 0, 1]
# and_layer = NeuronLayer([n_and], "and layer")
# n_and_network = NeuronNetwork([and_layer], "and network")
#
# print_neuron2(n_and)
#
# n_and_network.train(inputs_and, target_and, 0.0001, 10000)
#
# print_neuron2(n_and)



# f2 = Neuron([0.0, 0.1], 0, "f2")
# g2 = Neuron([0.2, 0.3], 0, "g2")
# h2 = Neuron([0.4, 0.5], 0, "h2")
# s3 = Neuron([0.6, 0.7, 0.8], 0, "s3")
# c3 = Neuron([0.9, 1.0, 1.1], 0, "c3")
#
# n_l1 = NeuronLayer([f2, g2, h2], "hidden layer 3n")
# n_l2 = NeuronLayer([s3, c3], "output layer 2n")
#
# half_adder = NeuronNetwork([n_l1, n_l2], "half_adder")
# target_half_adder = [[0,0], [0,1], [0,1], [1,0]]
# inputs_half_adder = list(itertools.product([0, 1], repeat=2))
#
# print_neuron2(half_adder)
#
# half_adder.train(inputs_half_adder, target_half_adder, 0.0001, 10000)
#
# print("")
# print_neuron2(half_adder)



# f2 = Neuron([1.2, 1.2], 0, "f2")
# g2 = Neuron([1.5, 1.43], 0, "g2")
# o3 = Neuron([1.1, 1.9], 0, "03")
#
# inputs_xor = list(itertools.product([0, 1], repeat=2))
# target_xor = [0, 1, 1, 0]
# xor_layer1 = NeuronLayer([f2, g2], "layer 1")
# xor_layer2 = NeuronLayer([o3], "layer 2")
# xor_network = NeuronNetwork([xor_layer1, xor_layer2], "xor network")
#
# print_neuron2(xor_network)
#
# xor_network.train(inputs_xor, target_xor, 0.0001, 10000)
#
# print_neuron2(xor_network)
#
# print(f2)
# print(g2)
# print(o3)
