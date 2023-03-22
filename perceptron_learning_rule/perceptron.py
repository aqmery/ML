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
        :param inputs: gets a list of floats and calculates the weighted sum based on the weights and the bias.
        :return: returns a step_function call with weighted_sum as argument.
        """
        weighted_sum = 0
        for i_in in range(len(inputs)):
            weighted_sum += inputs[i_in] * self.weights[i_in]
        weighted_sum = weighted_sum + self.bias
        return self.step_function(weighted_sum)

    def calculate_error(self, activation, target):
        error = target-activation
        return error

    def calculate_weight_delta(self, error, inputs):
        weight_changes = []
        for j in range(len(self.weights)):
            weight_changes.append((self.eta*error*inputs[j]))
        bias_change = self.eta*error
        return weight_changes, bias_change

    def update(self, weight_changes, bias_change):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]+weight_changes[i]
        self.bias = self.bias+bias_change

    def loss(self, inputs, target):
        errors = []
        for i in range(len(target)):
            errors.append((self.calculate_error(self.activate(inputs[i]), target[i]))**2/2)
        print(errors)
        mse = (sum(errors))/(len(target))
        return mse

    def train(self, inputs, target, error_threshold, train_threshold, eta):
        self.eta = eta
        cont = True
        epochs = 0
        while cont:
            # out = []
            # print(self.weights)
            # print(self.bias)
            for i in range(len(target)):
                # out.append(self.activate(inputs[i]))
                error = self.calculate_error(self.activate(inputs[i]), target[i])
                # print(error)
                weight_changes, bias_change = self.calculate_weight_delta(error, inputs[i])
                self.update(weight_changes, bias_change)
            epochs += 1
            # print(out)
            loss = self.loss(inputs, target)
            print(loss)
            if loss <= error_threshold:
                print("epochs trained :", epochs)
                cont = False
            if epochs >= train_threshold:
                print("epochs trained :", epochs)
                cont = False
            # cont = False











    # def update(self, p_input, activation, error_threshold):
    #     """
    #     calculates how "wrong" a prediction is, appends this to the "weight_error" list.
    #     using this list, calculates the MSE calling the "loss" function,
    #     as well as updating the weights of the perceptron calling the "update_weights" function.
    #     :param p_input: gets a list of tuples with all available inputs.
    #     :param activation: gets a list of ints with the target value for each tuple of the previous input.
    #     :param error_threshold: gives the "max" value for the loss function, if the loss function is lower, end the loop.
    #     :return: True when the error threshold is achieved, returns False if not.
    #     """
    #     weight_error = []
    #     for i_input in range(len(p_input)):
    #         temp_error = round(expit(sum([self.weights[i] * p_input[i_input][i] for i in range(len(self.weights))]) + self.bias), 3)
    #         print(temp_error)
    #         weight_error.append(temp_error)
    #     if self.loss(weight_error, activation) < error_threshold:
    #         return True
    #     self.update_weights(weight_error, activation)
    #     return False
    #
    # def update_weights(self, weight_error, activation):
    #     """
    #     calculates the error of the bias and each weight, and changes it based on the error.
    #     :param weight_error: gets a list of floats that indicate the error of each weight, compared to the target.
    #     :param activation: gets the list of targets.
    #     :return: changes the self.weights and self.bias instance variables, doesn't return anything.
    #     """
    #     for i_result in range(len(activation)):
    #         error = round(weight_error[i_result]*((1-weight_error[i_result])*
    #                                               -(activation[i_result]-weight_error[i_result])), 3)
    #         for i_weight in range(len(self.weights)):
    #             self.weights[i_weight] = self.weights[i_weight] - self.eta * activation[i_result] * error
    #         self.bias = self.bias - self.eta * 1 * error

    # def calculate_loss(self, weight_error, activation):
    #     """
    #     calculates the mean squared error (MSE).
    #     :param weight_error: gets a list of floats that indicate the error of each weight, compared to the target.
    #     :param activation: gets the list of targets.
    #     :return: calculates the MSE and returns this as a float.
    #     """
    #     error_sum = 0
    #     for i in range(len(weight_error)):
    #         error_sum += (activation[i]-weight_error[i])**2
    #     mean_squared_error = error_sum/(2*(len(weight_error)))
    #     return mean_squared_error
