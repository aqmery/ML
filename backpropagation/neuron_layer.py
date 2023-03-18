class NeuronLayer:
    """
    creates a NeuronLayer object.
    :param neurons: takes in a list of neurons.
    :param name: takes in a string for the name of the neuron-layer.
    """
    def __init__(self, neurons, name):
        self.neurons = neurons
        self.name = name

    def __str__(self):
        n_names = []
        for i in range(len(self.neurons)):
            n_names.append(self.neurons[i].name)
        return str(f"neuron names = {n_names}, layer name = {self.name}")

    def backprop(self, error, eta):
        print("layer name =", self.name)
        # self.neurons[0].backprop(error[0], eta)
        print(len(self.neurons))
        for i in range(len(self.neurons)):
            print("neuron name =", self.neurons[i].name)
            print("neuron weights = ", self.neurons[i].weights)
            print("error = ", error[0])
            # print(self.neurons[i].error)
            self.neurons[i].backprop(error[0], eta)
            print("")
        #     # print(self.neurons[i].weights)
        #     # print(self.neurons[i].bias)
        print("")

    def activate(self, inputs):
        """
        :param inputs: gets a list of Neuron objects.
        :return: a list of the outputs of the activated Neuron objects.
        """
        output = []
        for n in self.neurons:
            output.append(n.activate(inputs))
        return output
