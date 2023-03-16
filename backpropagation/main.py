from neuron import Neuron
# from neuron_layer import NeuronLayer
# from neuron_network import NeuronNetwork
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


def train_neuron(neuron, activation):
    inputs = list(itertools.product([0, 1], repeat=2))
    for j in range(1000):
        for sample, target in zip(inputs, activation):
            # print(sample, target)
            neuron.calculate_error(sample, target)
            neuron.update()
            # neuron.hidden_error()




n_and = Neuron([-0.5, 0.5], 1.5, "AND")
activation_and = [0, 0, 0, 1]

print_neuron2(n_and)

train_neuron(n_and, activation_and)

print_neuron2(n_and)
# print(n_and.activate([0, 0]))
























