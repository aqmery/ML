from Perceptron import Perceptron
from PerceptronLayer import PerceptronLayer
from PerceptronNetwork import PerceptronNetwork

# I'm using the name variable's to make it clear which perceptron I'm using,
# removing this variable from the perceptrons (and the functions) does not make a difference.

# initialize perceptrons
p_AND = Perceptron([0.5, 0.5], -1.0, "AND")
p_OR = Perceptron([1.0, 1.0], -1.0, "OR")
p_NAND = Perceptron([-1.0, -1.0], 1.0, "NAND")
p_NOT = Perceptron([-1.0], 0.0, "NOT")
p_NOR = Perceptron([-1.0, -1.0, -1.0], 0.0, "NOR")
p_through = Perceptron([1, 0, 0], -0.5, "through")
p_AND3 = Perceptron([0, 0.5, 0.5], -1.0, "AND3")

# initialize layers for XOR gate
p_layer_OR_NAND = PerceptronLayer([p_OR, p_NAND], "OR_NAND")
p_layer_AND = PerceptronLayer([p_AND], "AND")

# initialize network for XOR perceptron
p_XOR = PerceptronNetwork([p_layer_OR_NAND, p_layer_AND], "XOR")

# p_layer1 = PerceptronLayer([p_AND, p_OR, p_NAND], "test_XOR")
# p_layer2 = PerceptronLayer([p_through, p_AND3], "test_OR_NAND")
# test = PerceptronNetwork([p_layer1, p_layer2], "test_XOR3")


# working half adder:
p_layer1 = PerceptronLayer([p_AND, p_OR, p_NAND], "test_XOR")
p_layer2 = PerceptronLayer([p_through, p_AND3], "test_OR_NAND")
test = PerceptronNetwork([p_layer1, p_layer2], "test_XOR3")


inputs = 2
# test perceptrons with 2 different inputs
p_lst_2 = [test]
for p in p_lst_2:
    print(p.name)
    results = []
    print("i1, i2, out")
    for i in range(inputs):
        for j in range(inputs):
            results.append([[i, j], p.activate([i, j])])
            print(i, " ", j, " ", p.activate([i, j]))
    print("")


# p_lst_3 = [p_through, p_AND3]
# for p in p_lst_3:
#     print(p.name)
#     print("i1, i2, i3, out")
#     for i in range(inputs):
#         for j in range(inputs):
#             results = []
#             for k in range(inputs):
#                 results.append([[i, j, k], p.activate([i, j, k])])
#                 print(i, " ", j, " ", k, " ", p.activate([i, j, k]))
#     print("")


# # gives 0 and 1 as inputs to all functions bellow
# inputs = 2
# # test perceptrons with 2 different inputs
# p_lst_2 = [p_AND, p_OR, p_NAND, p_XOR]
# for p in p_lst_2:
#     print(p.name)
#     results = []
#     print("i1, i2, out")
#     for i in range(inputs):
#         for j in range(inputs):
#             results.append([[i, j], p.activate([i, j])])
#             print(i, j, p.activate([i, j]))
#     print("")
#
# # test perceptrons with 1 input
# p_lst_1 = [p_NOT]
# for p in p_lst_1:
#     print(p.name)
#     results = []
#     for i in range(inputs):
#         results.append([[i],p.activate([i])])
#     print(results)
#     print("")
#
# # test perceptrons with 3 different inputs
# p_lst_3 = [p_NOR]
# for p in p_lst_3:
#     print(p.name)
#     for i in range(inputs):
#         for j in range(inputs):
#             results = []
#             for k in range(inputs):
#                 results.append([[i, j, k], p.activate([i, j, k])])
#             print(results)
#     print("")
