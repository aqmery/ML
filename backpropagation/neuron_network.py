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
        if type(target) == int:
            error = (output[0]*(1-output[0]))*-(target-output[0])
            error_lst.append(error)
        else:
            for i in range(len(target)):
                error = (output[i]*(1-output[i]))*-(target[i]-output[i])
                error_lst.append(error)
        self.error = error_lst
        self.output = output

    def calculate_loss(self, error, target):
        total_sum = 0
        for i in range(len(error)):
            for j in range(len(error[0])):
                total_sum += (target[i][j]-error[i][j])**2
        return total_sum/(2*len(error)*len(error[0]))

    def calculate_output_layer(self, inputs, error):
        # print("asdf")
        # print(self.neuron_layers)
        # # print("layer name ", self.neuron_layers.name)
        # # print(self.neuron_layers.neurons)
        # print("asdf")
        # if len(self.neuron_layers) < 2:
        #     output_layer = 0
        # else:
        #     output_layer = len(self.neuron_layers)-2
        output_layer = len(self.neuron_layers) - 2
        # print(inputs)
        # print(error)
        # print(output_layer)
        for e in range(len(error)):
            weight_change = []
            for i in range(len(self.neuron_layers[output_layer].neurons)):
                weight_change.append(self.neuron_layers[output_layer].neurons[i].activate(inputs) * self.eta * error[e])
            bias_change = self.eta * error[e]
            self.neuron_layers[output_layer + 1].neurons[e].weightchanges = weight_change
            self.neuron_layers[output_layer + 1].neurons[e].biaschange = bias_change


    def calculate_hidden_layers(self):
        error = self.error
        count = 1
        amount_weights = len(error)
        for i in reversed(range(len(self.neuron_layers)-1)):
            amount_neurons = len(self.neuron_layers[i].neurons)
            temp = []
            for j in range(amount_neurons):
                weights = []
                for k in range(amount_weights):
                    weights.append(self.neuron_layers[len(self.neuron_layers) - count].neurons[k].weights[j])
                temp.append(self.neuron_layers[i].neurons[j].hidden_error(weights, error, self.eta))
            amount_weights = len(temp)
            error = []
            for j in range(len(temp)):
                error.append(temp[j][0])
            count += 1

    def update(self):
        # if len(self.neuron_layers) == 1 and len(self.neuron_layers[0].neurons) == 1:
        #     self.neuron_layers[0].neurons[0].update()
        for i in range(len(self.neuron_layers)):
            for j in range(len(self.neuron_layers[i].neurons)):
                self.neuron_layers[i].neurons[j].hidden_update()

    def backprop(self, inputs, target):
        self.calculate_error(inputs, target)
        self.calculate_output_layer(inputs, self.error)
        self.calculate_hidden_layers()

    def train(self, inputs, target, error_threshold, train_threshold):
        cont = True
        epochs = 0
        while cont:
            out = []
            for i in range(len(target)):
                out.append(self.activate(inputs[i]))
                self.backprop(inputs[i], target[i])
                self.update()
            loss = self.calculate_loss(out, target)
            epochs += 1
            if loss <= error_threshold:
                print("epochs trained :", epochs)
                cont = False
            if epochs >= train_threshold:
                print("epochs trained :", epochs)
                cont = False

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
