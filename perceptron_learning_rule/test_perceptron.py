import unittest
import perceptron
import itertools


class TestPerceptron(unittest.TestCase):
    def setUp(self) -> None:
        self.p_input_and = list(itertools.product([0, 1], repeat=2))
        self.activation_and = [0, 0, 0, 1]
        self.p_and = perceptron.Perceptron([-0.5, 0.5], 1.5, "AND")

        self.p_input_xor = list(itertools.product([0, 1], repeat=2))
        self.activation_xor = [0, 1, 1, 0]
        self.p_xor = perceptron.Perceptron([-0.5, 0.5], 1.5, "XOR")

    def tearDown(self) -> None:
        print("teardown")

    def test_perceptron_variables(self):
        self.assertEqual(self.p_and.name, "AND")
        self.assertEqual(self.p_xor.name, "XOR")

        self.assertEqual(self.p_and.weights, [-0.5, 0.5])
        self.assertEqual(self.p_xor.weights, [-0.5, 0.5])

        self.assertEqual(self.p_and.bias, 1.5)
        self.assertEqual(self.p_xor.bias, 1.5)

        self.assertEqual(self.p_and.eta, 0.1)
        self.assertEqual(self.p_xor.eta, 0.1)

    def test_update(self):
        self.p_and.update(self.p_input_and, self.activation_and, 0.05)
        self.assertNotEqual(self.p_and.weights, [-0.5, 0.5])

        self.p_xor.update(self.p_input_xor, self.activation_xor, 0.05)
        self.assertNotEqual(self.p_xor.weights, [-0.5, 0.5])

    def test_train_and(self):
        results = []
        stop = False
        count = 0
        while not stop:
            count += 1
            if self.p_and.update(self.p_input_and, self.activation_and, 0.05):
                stop = True
            if count == 1000:
                stop = True
        self.assertLess(count, 1000)
        for i in range(len(self.p_input_and)):
            results.append(self.p_and.activate(self.p_input_and[i]))
        self.assertEqual(results, self.activation_and)

    def test_train_xor(self):
        results = []
        stop = False
        count = 0
        while not stop:
            count += 1
            if self.p_xor.update(self.p_input_xor, self.activation_xor, 0.05):
                stop = True
            if count == 1000:
                stop = True
        self.assertEqual(count, 1000)
        for i in range(len(self.p_input_xor)):
            results.append(self.p_xor.activate(self.p_input_xor[i]))
        self.assertNotEqual(results, self.activation_xor)


if __name__ == "__main__":
    unittest.main()


