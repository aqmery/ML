from neuron import Neuron
from neuron_layer import NeuronLayer
from neuron_network import NeuronNetwork
import itertools


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


"""setup neurons using the old weights and biases."""
n_AND_old = Neuron([0.5, 0.5], -1.0, "AND_old")
n_OR_old = Neuron([1.0, 1.0], -1.0, "OR_old")
n_NOT_old = Neuron([-1.0], 0.0, "NOT_old")


"""when using the step_function this works because, if the weighted_sum is higher than 0 it activates.
this does not work for the sigmoid function because the sigmoid function only outputs values between 0 and 1
meaning it can never get below 0.
1 solution to this is to use 0.5 as the activation threshold.
a different solution is scaling the weights and biases of the neurons so they approach 0 or 1"""
print_neuron2(n_AND_old)
print_neuron2(n_OR_old)
print_neuron1(n_NOT_old)


"""setup for the half adder neurons with new weights"""
n_AND = Neuron([12, 12], -20.0, "AND")
n_OR = Neuron([15.0, 15.0], -10.0, "OR")
n_NAND = Neuron([-27.0, -27.0], 50.0, "NAND")
n_through = Neuron([20, 0, 0], -10, "through")
n_AND3 = Neuron([0, 10, 10], -16.0, "AND3")

"""prints the new neurons"""
print_neuron2(n_AND)
print_neuron2(n_OR)
print_neuron2(n_NAND)
print_neuron3(n_through)
print_neuron3(n_AND3)


"""creates the half adder layers and prints the half adder"""
n_layer1 = NeuronLayer([n_AND, n_OR, n_NAND], "AND_OR_NAND")
n_layer2 = NeuronLayer([n_through, n_AND3], "through_AND3")
half_adder = NeuronNetwork([n_layer1, n_layer2], "half_adder")

print_neuron2(half_adder)
