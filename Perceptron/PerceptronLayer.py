class PerceptronLayer:
    def __init__(self, perceptrons, name):
        self.perceptrons = perceptrons
        self.name = name

    def __str__(self):
        p_names = []
        for i in range(len(self.perceptrons)):
            p_names.append(self.perceptrons[i].name)
        return str(f"perceptron names = {p_names}, layer name = {self.name}")

    def activate(self, inputs):
        # print("input layer =", inputs)

        output = []
        for i in self.perceptrons:
            output.append(i.activate(inputs))
            # print("output layer =", output)
        return output