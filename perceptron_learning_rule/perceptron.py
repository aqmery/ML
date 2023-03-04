from scipy.special import expit

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
        self.eta = .1

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
        for i_in in range(len(inputs)):
            weighted_sum += inputs[i_in] * self.weights[i_in]
        weighted_sum = weighted_sum + self.bias
        return self.step_function(weighted_sum)

    def update(self, p_input, activation, error_threshold):
        weight_error = []
        for i_input in range(len(p_input)):
            temp_error = round(expit(sum([self.weights[i] * p_input[i_input][i] for i in range(len(self.weights))]) + self.bias), 3)
            weight_error.append(temp_error)
        if self.loss(weight_error, activation) < error_threshold:
            return True
        self.back_propagation(weight_error, activation)
        return False

    def back_propagation(self, weight_error, activation):
        for i_result in range(len(activation)):
            error = round(weight_error[i_result]*((1-weight_error[i_result])*
                                                  -(activation[i_result]-weight_error[i_result])), 3)
            for i_weight in range(len(self.weights)):
                self.weights[i_weight] = self.weights[i_weight] - self.eta * activation[i_result] * error
            self.bias = self.bias - self.eta * 1 * error

    def loss(self, weight_error, activation):
        error_sum = 0
        for i_error in range(len(weight_error)):
            error_sum += (activation[i_error]-weight_error[i_error])**2
        mean_squared_error = error_sum/(2*(len(weight_error)))
        return mean_squared_error