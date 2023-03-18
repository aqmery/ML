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
            # print(f"output[i] {output[i]}, 1-output[i] {1-output[i]}, target[i] {target[i]}, output[i] {output[i]}")
            error = (output[i]*(1-output[i]))*-(target[i]-output[i])
            error_lst.append(error)
        self.error = error_lst
        self.output = output

    def calculate_output_layer(self, inputs, error):
        last = len(self.neuron_layers)-2
        for e in range(len(error)):
            weight_change = []
            for i in range(len(self.neuron_layers[last].neurons)):
                print(self.neuron_layers[last].neurons[i].name)
                print(self.neuron_layers[last].neurons[i].activate(inputs))
                print(self.neuron_layers[last].neurons[i].weights)
                print(self.neuron_layers[last].neurons[i].activate(inputs) * self.eta * error[e])
                print("")
            print(self.neuron_layers[last+1].neurons[e].name)
            print(self.eta*error[e])
            print("")
            print("")
    #         for j in range(len(self.neuron_layers[last].neurons[i].weights)):
    #             weight_change.append(self.eta*self.neuron_layers[last].neurons[i].weights[j]*error[i])
    #             print(self.neuron_layers[last].neurons[i].weights[j])
    #         print(weight_change)


    def backprop(self, inputs, target):
        self.activate(inputs)
        self.calculate_error(inputs, target)
        error = self.error
        print("self.error", self.error)
        print("self.output", self.output)
        self.calculate_output_layer(inputs, error)

        print("-----------------------------")
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

    # def feed_forward(self, outputs):
    #     """
    #     decides if the network should output a 0 or a 1.
    #     :param outputs: gets the outputs of all the layers for a specific input.
    #     :return: returns either a 1 or a 0 based on the values of the outputs.
    #     """
    #     new_out = []
    #     for output in outputs:
    #         if output >= 0.5:
    #             new_out.append(1)
    #         else:
    #             new_out.append(0)
    #     return new_out
