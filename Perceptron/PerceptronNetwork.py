class PerceptronNetwork:
    def __init__(self, perceptronLayers, name):
        self.perceptronLayers = perceptronLayers
        self.name = name

    def activate(self, inputs):
        outputs = []
        for i in range(len(self.perceptronLayers)):
            outputs = self.perceptronLayers[i].activate(inputs)
            inputs = outputs
        return outputs

    def __str__(self):
        l_names = []
        for i in range(len(self.perceptronLayers)):
            l_names.append(self.perceptronLayers[i].name)
        return str(f"perceptron layer names = {l_names}, network name = {self.name}")