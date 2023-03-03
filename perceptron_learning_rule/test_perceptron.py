import unittest
import perceptron
import itertools


class TestPerceptron(unittest.TestCase):
    def setUp(self) -> None:
        self.x_and = list(itertools.product([0, 1], repeat=2))
        self.y_and = [0, 0, 0, 1]
        self.p_and = perceptron.Perceptron([-0.5, 0.5], 1.5, "AND")

        self.x_xor = list(itertools.product([0, 1], repeat=2))
        self.y_xor = [0, 1, 1, 0]
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
        self.p_and.update(self.x_and, self.y_and)
        self.assertNotEqual(self.p_and.weights, [-0.5, 0.5])

        self.p_xor.update(self.x_xor, self.y_xor)
        self.assertNotEqual(self.p_xor.weights, [-0.5, 0.5])

    def test_train_and(self):
        results = []
        stop = False
        count = 0
        while not stop:
            count += 1
            if self.p_and.update(self.x_and, self.y_and):
                stop = True
            if count == 1000:
                stop = True
        self.assertLess(count, 1000)
        for i in range(len(self.x_and)):
            results.append(self.p_and.activate(self.x_and[i]))
        self.assertEqual(results, self.y_and)

    def test_train_xor(self):
        results = []
        stop = False
        count = 0
        while not stop:
            count += 1
            if self.p_xor.update(self.x_xor, self.y_xor):
                stop = True
            if count == 1000:
                stop = True
        self.assertEqual(count, 1000)
        for i in range(len(self.x_xor)):
            results.append(self.p_xor.activate(self.x_xor[i]))
        self.assertNotEqual(results, self.y_xor)


if __name__ == "__main__":
    unittest.main()


