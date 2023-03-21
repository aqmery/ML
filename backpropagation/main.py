import pandas as pd
import numpy as np
import random
import itertools
from sklearn.datasets import load_iris
from neuron import Neuron
from neuron_layer import NeuronLayer
from neuron_network import NeuronNetwork
random.seed(1782152)


def print_neuron(network):
    num_weights = len(network.neuron_layers[0].neurons[0].weights)
    print(f"{network}\n{network.name}")
    for w in range(num_weights):
        print(f"in{w + 1} ", end="")
    print("| out")
    for out in itertools.product(range(2), repeat=num_weights):
        input_str = "   ".join(str(i) for i in out)
        print(f"{input_str}   | {network.activate(out)}")
    print("")


def create_network(layers, neurons_list, amount_weights, network_name):
    neuron_layers = []
    for l in range(layers):
        neurons = []
        for n in range(neurons_list[l]):
            neurons.append(Neuron([random.uniform(-1, 1) for _ in range(amount_weights[l])],
                                  random.uniform(-1, 1), "L"+str(l)+"N"+str(n)))
        neuron_layers.append(NeuronLayer(neurons, "Layer"+str(l)))
    return NeuronNetwork(neuron_layers, str(network_name))


and_network = create_network(1, [1], [2], "and_network")
inputs_and = list(itertools.product([0, 1], repeat=2))
target_and = [0, 0, 0, 1]

print_neuron(and_network)

and_network.train(inputs_and, target_and, 0.001, 10000, 0.5)

print("")
print_neuron(and_network)
print("------------------------------------------------------------------------")


half_adder = create_network(2, [3, 2], [2, 3], "half_adder")
target_half_adder = [[0,0], [0,1], [0,1], [1,0]]
inputs_half_adder = list(itertools.product([0, 1], repeat=2))

print_neuron(half_adder)

half_adder.train(inputs_half_adder, target_half_adder, 0.01, 10000, 0.2)

print("")
print_neuron(half_adder)
print("------------------------------------------------------------------------")


xor_network = create_network(2, [2, 1], [2, 2], "xor_network")
inputs_xor = list(itertools.product([0, 1], repeat=2))
target_xor = [0, 1, 1, 0]

print_neuron(xor_network)

xor_network.train(inputs_xor, target_xor, 0.001, 10000, 0.5)

print("")
print_neuron(xor_network)
print("------------------------------------------------------------------------")


iris = load_iris()
df_iris = pd.DataFrame(data=np.c_[iris["data"], iris["target"]],
                     columns= iris["feature_names"] + ["target"])

iris_network = create_network(2, [4, 3], [4, 4], "iris_network")
target_iris = []
for i in range(len(iris["target"])):
    if iris["target"][i] == 0.0:
        target_iris.append([1, 0, 0])
    elif iris["target"][i] == 1:
        target_iris.append([0, 1, 0])
    else:
        target_iris.append([0, 0, 1])
inputs_iris = [list(row[:4]) for row in df_iris.values]

iris_network.train(inputs_iris, target_iris, 0.001, 1000, 0.5)

results = [iris_network.collapse_activate(inputs_iris[row]) for row in range(len(inputs_iris))]

correct = sum(1 for i in range(len(target_iris)) if results[i] == target_iris[i])
print(correct/len(target_iris))
