class NeuronNetwork:
    """
    creates a NeuronNetwork object.
    :param neuron_layers: takes in a list of neuronLayers.
    :param name: takes in a string for the name of the neuron-network.
    """
    def __init__(self, neuron_layers, name):
        self.neuron_layers = neuron_layers
        self.name = name

    def __str__(self):
        l_names = []
        for i in range(len(self.neuron_layers)):
            l_names.append(self.neuron_layers[i].name)
        return str(f"neuron layer names = {l_names}, network name = {self.name}")

    def activate(self, inputs):
        """
        calculates the outputs of a neuronLayer object and gives these outputs to the next layer,
        this goes on until the loop has gone through all the layers and returns the last outputs.
        :param inputs: gets a list of NeuronLayer objects.
        :return: a list of the outputs of last activated NeuronLayer object.
        """
        outputs = []
        for i in range(len(self.neuron_layers)):
            outputs = self.neuron_layers[i].activate(inputs)
            inputs = outputs
        return self.feed_forward(outputs)

    def feed_forward(self, outputs):
        """
        decides if the network should output a 0 or a 1.
        :param outputs: gets the outputs of all the layers for a specific input.
        :return: returns either a 1 or a 0 based on the values of the outputs.
        """
        new_out = []
        for output in outputs:
            if output >= 0.5:
                new_out.append(1)
            else:
                new_out.append(0)
        return new_out
