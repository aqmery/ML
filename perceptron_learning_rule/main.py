import random
import itertools
from perceptron import Perceptron
random.seed(1782152)


def print_perceptron(perceptron, inputs=2):
    print(f"{perceptron}\n{perceptron.name}\nin1, in2 | out")
    for i, j in itertools.product(range(inputs), repeat=2):
        print(f"{i}  {j}  | {perceptron.activate([i, j])}")
    print("")


def train_perceptron(perceptron, activation):
    x = list(itertools.product([0, 1], repeat=2))
    stop = False
    count = 0
    while not stop:
        count += 1
        if perceptron.update(x, activation):
            print(f"finished training this perceptron in {count} loops")
            stop = True
        if count == 1000:
            print(f"couldn't train this perceptron in {count} loops")
            stop = True
    print("")


y_and = [0, 0, 0, 1]
p_and = Perceptron([random.uniform(-1, 1), random.uniform(-1, 1)],
                   random.uniform(-1, 1),
                   "AND")
y_xor = [0, 1, 1, 0]
p_xor = Perceptron([random.uniform(-1, 1), random.uniform(-1, 1)],
                   random.uniform(-1, 1),
                   "XOR")


print_perceptron(p_and)
train_perceptron(p_and, y_and)
print_perceptron(p_and)
print("----------------------------------------------------\n")
print_perceptron(p_xor)
train_perceptron(p_xor, y_xor)
print_perceptron(p_xor)