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
        p_names = []
        for i in range(len(self.neurons)):
            p_names.append(self.neurons[i].name)
        return str(f"neuron names = {p_names}, layer name = {self.name}")

    def activate(self, inputs):
        """
        :param inputs: gets a list of Neuron objects.
        :return: a list of the outputs of the activated Neuron objects.
        """
        output = []
        for p in self.neurons:
            output.append(p.activate(inputs))
        return output
