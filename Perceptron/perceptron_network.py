class PerceptronNetwork:
    """
    creates a PerceptronNetwork object.
    :param perceptronLayers: takes in a list of perceptronLayers.
    :param name: takes in a string for the name of the perceptron-network.
    """
    def __init__(self, perceptron_layers, name):
        self.perceptron_layers = perceptron_layers
        self.name = name

    def __str__(self):
        l_names = []
        for i in range(len(self.perceptron_layers)):
            l_names.append(self.perceptron_layers[i].name)
        return str(f"perceptron layer names = {l_names}, network name = {self.name}")

    def activate(self, inputs):
        """
        calculates the outputs of a perceptronLayer object and gives these outputs to the next layer,
        this goes on until the loop has gone through all the layers and returns the last outputs.
        :param inputs: gets a list of PerceptronLayer objects.
        :return: a list of the outputs of last activated PerceptronLayer object.
        """
        outputs = []
        for i in range(len(self.perceptron_layers)):
            outputs = self.perceptron_layers[i].activate(inputs)
            inputs = outputs
        return outputs
