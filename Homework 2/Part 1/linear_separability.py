import pickle

import numpy as np


class SimplePerceptron():
    def __init__(self, input, target, learning_rate=0.2, updates=100000):
        self.inputs = inputs
        self.targets = target
        self.learning_rate = learning_rate
        self.updates = updates
        self.weights = self._initialize_weights()
        self.thresholds = self._initialize_thresholds()
        self.function = 0

    def _initialize_weights(self, n=4, low=-0.2, high=0.2):
        u = np.random.rand(n)
        w = low + (high - low) * u
        return w

    def _initialize_thresholds(self, n=1, low=-1, high=1):
        u = np.random.rand(n)
        t = low + (high - low) * u
        return t

    def _get_random_sample(self):
        length = len(self.inputs)
        index = np.random.randint(length)
        input = self.inputs[index, :]
        target = self.targets[index, self.function]
        return input, target

    def feedforward(self, input):
        return np.tanh(0.5 * np.dot(self.weights, input) - self.thresholds)

    def backpropagate(self, input, target, output):
        error = (target - output) * (1 - output ** 2)
        dw = self.learning_rate * error * input
        dt = -1 * self.learning_rate * error
        return dw, dt

    def _update(self):
        input, target = self._get_random_sample()
        output = self.feedforward(input)
        dw, dt = self.backpropagate(input, target, output)
        self.weights += dw
        self.thresholds += dt

    def train(self):
        for index in range(self.updates):
            self._update()
            converged = self._check_convergence()
            if converged:
                print(f"Converged after {index} iterations...")
                self.print()
                return True
        print("No convergence...")
        return False

    def _check_convergence(self):
        length = self.inputs.shape[0]
        for index in range(length):
            input = self.inputs[index, :]
            output = np.sign(self.feedforward(input))
            target = self.targets[index, self.function]
            if output != target:
                return False
        return True

    def print(self):
        print(f"Weights: {self.weights}")
        print(f"Threshold: {self.thresholds}")


def main():
    with open('output.pkl', 'rb') as f:
        targets = pickle.load(f)

    with open('input.pkl', 'rb') as f:
        inputs = pickle.load(f)

    network = SimplePerceptron(inputs, targets)
    network.train()


if __name__ == "__main__":
    main()
