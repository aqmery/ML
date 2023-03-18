class NeuronNetwork:
    """
    creates a NeuronNetwork object.
    :param neuron_layers: takes in a list of neuronLayers.
    :param name: takes in a string for the name of the neuron-network.
    """
    def __init__(self, neuron_layers, name):
        self.neuron_layers = neuron_layers
        self.name = name
        self.eta = 1
        self.inputs = None
        self.error = None
        self.output = None

    def __str__(self):
        l_names = []
        for i in range(len(self.neuron_layers)):
            l_names.append(self.neuron_layers[i].name)
        return str(f"neuron layer names = {l_names}, network name = {self.name}")

    def calculate_error(self, inputs, target):
        output = self.activate(inputs)
        error_lst = []
        for i in range(len(target)):
            error = (output[i]*(1-output[i]))*-(target[i]-output[i])
            error_lst.append(error)
        self.error = error_lst
        self.output = output

    def calculate_output_layer(self, inputs, error):
        output_layer = len(self.neuron_layers)-2
        for e in range(len(error)):
            weight_change = []
            for i in range(len(self.neuron_layers[output_layer].neurons)):
                weight_change.append(self.neuron_layers[output_layer].neurons[i].activate(inputs) * self.eta * error[e])
            bias_change = self.eta * error[e]
            self.neuron_layers[output_layer + 1].neurons[e].weightchanges = weight_change
            self.neuron_layers[output_layer + 1].neurons[e].biaschange = bias_change

    def calculate_hidden_layers(self):
        error = self.error
        print("")
        for i in reversed(range(len(self.neuron_layers)-1)):
            print("layer =", i, "layer name =", self.neuron_layers[i].name)
            temp = []
            print(len(self.neuron_layers[i].neurons))
            for j in range(len(self.neuron_layers[i].neurons)):
                weights = []
                for k in range(len(error)):
                    weights.append(self.neuron_layers[len(self.neuron_layers) - 1].neurons[k].weights[j])
                print("print weights from network =", weights)
                temp.append(self.neuron_layers[i].neurons[j].hidden_error(weights, error, self.eta))
            print("")
            print("-----------------------------")
            print("")
            print("")

    def backprop(self, inputs, target):
        self.activate(inputs)
        self.calculate_error(inputs, target)
        error = self.error
        print("self.error", self.error)
        print("self.output", self.output)
        print("")
        self.calculate_output_layer(inputs, error)
        print("-----------------------------")
        self.calculate_hidden_layers()
        # for i in reversed(range(len(self.neuron_layers))):
        #     self.neuron_layers[i].backprop(error, self.eta)
        #     print("-----------------------------")
        # error =

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
        return outputs
        # return self.feed_forward(outputs)
