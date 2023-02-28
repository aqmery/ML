import random
import itertools
from scipy.special import expit
random.seed(1782152)

class Perceptron:
    """
    creates a Perceptron object.
    :param weights: takes in a list of floats as weights for the perceptron.
    :param bias: takes in a float as bias for the perceptron.
    :param name: takes in a string for the name of the perceptron.
    """
    def __init__(self, weights, bias, name):
        self.weights = weights
        self.bias = bias
        self.name = name
        self.eta = 0.1

    def __str__(self):
        return f"weights = {self.weights}, bias = {self.bias}, name = {self.name}"

    def step_function(self, weighted_sum):
        """
        :param weighted_sum: gets a float and checks if it's higher than 0.
        :return: returns 1 or 0 based on the weighted_sum.
        """
        if weighted_sum >= 0:
            return 1
        else:
            return 0

    def activate(self, inputs):
        """
        :param inputs: gets a list of ints and calculates the weighted sum based on the weights and the bias.
        :return: returns a step_function call with weighted_sum as argument.
        """
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]
        weighted_sum = weighted_sum + self.bias
        return self.step_function(weighted_sum)

    def update(self, x, y):
        loss_weight = []
        error = None
        for i in range(len(x)):
            # print(x[i])
            # print(self.activate(x[i]))
            # print(self.weights, self.bias)
            # print(y)
            # print(x[i][0])
            t = round(expit(self.weights[0]*x[i][0] + self.weights[1]*x[i][1] + self.bias), 3)
            loss_weight.append(t)
            # print(t)
            error = round(t*((1-t)*-(y[i]-t)), 3)
            # print(error)
            self.back_propagation(x[i], y[i], error)
        #     print("")
        # print(loss_weight)

    def back_propagation(self, inputs, activation, error):
        # print("in back propagation")
        # print(inputs)
        # print(error)
        print(self.weights)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]-self.eta*activation*error
        self.bias = self.bias-self.eta*1*error





x = list(itertools.product([0, 1], repeat=2))
y = [0, 0, 0, 1]


p_and = Perceptron([-0.5, 0.5], 1.5, "AND")


inputs = 2
# test perceptrons with 2 different inputs
p_lst_2 = [p_and]
for p in p_lst_2:
    print(p.name)
    print("in1, in2 | out")
    for i in range(inputs):
        for j in range(inputs):
            print(i, "  ", j, "  |", p.activate([i, j]))
    print("")


for i in range(140):
    p_and.update(x, y)
    print(p_and)

# p_and.update(x, y)



# test perceptrons with 2 different inputs
p_lst_2 = [p_and]
for p in p_lst_2:
    print(p.name)
    print("in1, in2 | out")
    for i in range(inputs):
        for j in range(inputs):
            print(i, "  ", j, "  |", p.activate([i, j]))
    print("")













# def update(self, activation):
#     print(expit(self.weights[0]*activation[0]+self.weights[1]*activation[1]+self.bias))
#     self.loss_weight.append(expit(self.weights[0]*activation[0]+self.weights[1]*activation[1]+self.bias))
#
#     # for i in range(len(self.weights)):
#     #     print(activation[i]*self.weights[i]+self.bias)
#     #     print(expit(activation[i]*self.weights[i]+self.bias))
#
#
# def loss(self, outputs):
#     loss = []
#     for j in range(len(self.loss_weight)):
#         loss.append((outputs[j]-self.loss_weight[j])**2)
#     print(sum(loss)/(2*len(outputs)))


# x = list(itertools.product([0, 1], repeat=2))
# y = [0, 0, 0, 1]
#
# # p_and = Perceptron([random.uniform(-1, 1) for _ in range(2)],
# #                    random.uniform(-1, 1),
# #                    "AND")
#
# p_and = Perceptron([-0.5, 0.5], 1.5, "AND")
#
# for i in range(len(x)):
#     print(p_and)
#     print(x[i])
#     p_and.update(x[i])
#     if p_and.activate(x[i]) != y[i]:
#         print("false")
#     else:
#         print("true")
#     print("")
#     print(p_and.activate(x[i]))
#     print(y[i])
#     print("---------")
#
# print(p_and.loss_weight)
# p_and.loss(y)
# # test_p_AND = Perceptron([0.5, 0.5], -1.0, "AND")


