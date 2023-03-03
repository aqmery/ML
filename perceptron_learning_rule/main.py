import pandas as pd
import numpy as np
import random
import itertools
from perceptron import Perceptron
from sklearn.datasets import load_iris
random.seed(1782152)


def print_perceptron(perceptron):
    print(f"{perceptron}\n{perceptron.name}\nin1, in2 | out")
    for i, j in itertools.product(range(2), repeat=2):
        print(f"{i}  {j}  | {perceptron.activate([i, j])}")
    print("")


def train_perceptron(perceptron, activation, error_treshold=0.05):
    p_input = list(itertools.product([0, 1], repeat=2))
    stop = False
    count = 0
    while not stop:
        count += 1
        if perceptron.update(p_input, activation, error_treshold):
            print(f"finished training this perceptron in {count} loops")
            stop = True
        if count == 1000:
            print(f"couldn't train this perceptron in {count} loops")
            stop = True
    print("")


def train_iris_perceptron(perceptron, p_input, activation, error_treshold=0.05):
    stop = False
    count = 0
    while not stop:
        count += 1
        if perceptron.update(p_input, activation, error_treshold):
            print(f"finished training this perceptron in {count} loops")
            stop = True
        if count == 1000:
            print(f"couldn't train this perceptron in {count} loops")
            stop = True
    print("")


activation_and = [0, 0, 0, 1]
p_and = Perceptron([random.uniform(-1, 1) for _ in range(2)],
                   random.uniform(-1, 1),
                   "AND")
activation_xor = [0, 1, 1, 0]
p_xor = Perceptron([random.uniform(0, 1) for _ in range(2)],
                   random.uniform(-1, 1),
                   "XOR")


print_perceptron(p_and)
train_perceptron(p_and, activation_and)
print_perceptron(p_and)
print("----------------------------------------------------\n")
print_perceptron(p_xor)
train_perceptron(p_xor, activation_xor)
print_perceptron(p_xor)


iris = load_iris()
df_iris = pd.DataFrame(data=np.c_[iris["data"], iris["target"]],
                     columns= iris["feature_names"] + ["target"])
df_iris1 = df_iris.loc[df_iris["target"] != 2.0]
df_iris2 = df_iris.loc[df_iris["target"] != 1.0]
p_iris1 = Perceptron([random.uniform(0, 1) for _ in range(4)],
                     random.uniform(-1, 1),
                     "p_iris1")
p_iris2 = Perceptron([random.uniform(0, 1) for _ in range(4)],
                     random.uniform(-1, 1),
                     "p_iris2")
iris_input1 = []
iris_activation1 = []
for i in range(len(df_iris1.values)):
    iris_input1.append(tuple(df_iris1.values[i][0:4]))
    iris_activation1.append(int(df_iris1.values[i][4:5]))

train_iris_perceptron(p_iris1, iris_input1, iris_activation1, 0.01)

results = []
for i in range(len(df_iris1.values)):
    results.append(p_iris1.activate(df_iris1.values[i][0:4]))

print(results)

correct = sum(1 for i in range(len(iris_activation1)) if results[i] == iris_activation1[i])
print(f"percentage correct = {correct/len(iris_activation1)*100}%")
