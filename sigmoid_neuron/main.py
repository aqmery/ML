from neuron import Neuron
from neuron_layer import NeuronLayer
from neuron_network import NeuronNetwork
import itertools


def print_neuron(neuron):
    """
    takes a neuron with 2 weights and prints the name, input1, input2 and the output.
    :param neuron: gets a neuron with 2 weights.
    :return: prints a 4 by 3 grid of possible inputs and the output.
    """
    print(f"{neuron}\n{neuron.name}\nin1 in2 | out")
    for i, j in itertools.product(range(2), repeat=2):
        print(f"{i}   {j}   | {neuron.activate([i, j])}")
    print("")


n_AND = Neuron([0.5, 0.5], -1.0, "AND")
n_OR = Neuron([1.0, 1.0], -1.0, "OR")
n_NOT = Neuron([-1.0], 0.0, "NOT")


"""when using the step_function this works because, when the weighted_sum is higher than 0 it activates.
this does not work for the sigmoid function because the sigmoid function only outputs values between 0 and 1
meaning it can never get below 0.
a solution to this is to use 0.5 as the activation threshold."""
print_neuron(n_AND)
print_neuron(n_OR)
p_lst_1 = [n_NOT]
print(f"{n_NOT.name}\nin1 | out")
for i in range(2):
    print(f"{i}   | {n_NOT.activate([i])}")
print("")


n_AND2 = Neuron([1.0, 1.0], -1.5, "AND2")
n_NAND = Neuron([-1.0, -1.0], 1.0, "NAND")
n_through = Neuron([1, 0, 0], -0.5, "through")
n_AND3 = Neuron([0, 0.5, 0.5], -1.0, "AND3")

print_neuron(n_AND2)


n_layer1 = NeuronLayer([n_AND, n_OR, n_NAND], "AND_OR_NAND")
n_layer2 = NeuronLayer([n_through, n_AND3], "through_AND3")
half_adder = NeuronNetwork([n_layer1, n_layer2], "half_adder")


print_neuron(half_adder)
