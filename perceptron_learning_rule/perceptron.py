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

    def calculate_error(self, output, target):
        """
        calculates the error of a datapoint.
        :param output: the output of the datapoint.
        :param target: the target of the datapoint.
        :return: returns the error of the datapoint.
        """
        error = target-output
        return error

    def calculate_weight_delta(self, error, inputs):
        """
        calculates the changes of the weights and bias.
        :param error: the error of the datapoint.
        :param inputs: the inputs of the datapoint.
        :return: returns the changes in the weights and bias.
        """
        weight_changes = []
        for j in range(len(self.weights)):
            weight_changes.append((self.eta*error*inputs[j]))
        bias_change = self.eta*error
        return weight_changes, bias_change

    def update(self, weight_changes, bias_change):
        """
        changes the weights and bias.
        :param weight_changes: the change in the weights.
        :param bias_change: the change in the bias.
        :return: sets the new weights and bias.
        """
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]+weight_changes[i]
        self.bias = self.bias+bias_change

    def loss(self, inputs, target):
        """
        calculates how "wrong" the list of all datapoints.
        :param inputs: a list of all the inputs.
        :param target: a list of all the targets.
        :return: returns the mean squared error.
        """
        errors = []
        for i in range(len(target)):
            errors.append((self.calculate_error(self.activate(inputs[i]), target[i]))**2/2)
        mse = (sum(errors))/(len(target))
        return mse

    def train(self, inputs, target, error_threshold, train_threshold, eta):
        """
        trains the perceptron.
        :param inputs: a list of all the inputs.
        :param target: a list of all the targets.
        :param error_threshold: how "wrong" the final output can be, before stopping with training.
        :param train_threshold: the amount of loops the perceptron can train to prevent an infinite loop,
        if the error_threshold can't be reached.
        :param eta: the "stepsize" of the changes.
        :return: returns a perceptron that should be able to classify a dataset used to train it.
        """
        self.eta = eta
        cont = True
        epochs = 0
        while cont:
            for i in range(len(target)):
                error = self.calculate_error(self.activate(inputs[i]), target[i])
                weight_changes, bias_change = self.calculate_weight_delta(error, inputs[i])
                self.update(weight_changes, bias_change)
            epochs += 1
            loss = self.loss(inputs, target)
            if loss <= error_threshold:
                print(f"finished training this perceptron in {epochs} loops")
                cont = False
            if epochs >= train_threshold:
                print(f"couldn't train this perceptron further in {epochs} epochs")
                cont = False
