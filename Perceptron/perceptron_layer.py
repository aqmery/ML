class PerceptronLayer:
    """
    creates a PerceptronLayer object.
    :param perceptrons: takes in a list of perceptrons.
    :param name: takes in a string for the name of the perceptron-layer.
    """
    def __init__(self, perceptrons, name):
        self.perceptrons = perceptrons
        self.name = name

    def __str__(self):
        p_names = []
        for i in range(len(self.perceptrons)):
            p_names.append(self.perceptrons[i].name)
        return str(f"perceptron names = {p_names}, layer name = {self.name}")

    def activate(self, inputs):
        """
        :param inputs: gets a list of Perceptron objects.
        :return: a list of the outputs of the activated Perceptron objects.
        """
        output = []
        for p in self.perceptrons:
            output.append(p.activate(inputs))
        return output
