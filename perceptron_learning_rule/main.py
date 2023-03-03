import random
import itertools
from perceptron import Perceptron
random.seed(1782152)


def print_perceptron(perceptron):
    print(perceptron)
    inputs = 2
    print(perceptron.name)
    print("in1, in2 | out")
    for i in range(inputs):
        for j in range(inputs):
            print(i, "  ", j, "  |", perceptron.activate([i, j]))
    print("")


x = list(itertools.product([0, 1], repeat=2))
y_and = [0, 0, 0, 1]
p_and = Perceptron([-0.5, 0.5], 1.5, "AND")

print_perceptron(p_and)

stop = False
count = 0
while not stop:
    count += 1
    if p_and.update(x, y_and):
        stop = True
    if count == 1000:
        stop = True
print("amount of updates:", count)
print("")

print_perceptron(p_and)


y_xor = [0, 1, 1, 0]
p_xor = Perceptron([-0.5, 0.5], 1.5, "XOR")

print_perceptron(p_xor)

stop = False
count = 0
while not stop:
    count += 1
    if p_xor.update(x, y_xor):
        stop = True
    if count == 1000:
        stop = True
print("amount of updates:", count)
print("")

print_perceptron(p_xor)