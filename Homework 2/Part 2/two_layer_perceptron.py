import pickle

import numpy as np

# Parameters
m1 = 8
m2 = 4


class TwoLayerPerceptron():
    def __init__(self, training, validation, learning_rate=0.02, epochs=1000):
        self.training = training
        self.validation = validation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w1 = self._initialize_weights(m1, 2)
        self.w2 = self._initialize_weights(m2, m1)
        self.w3 = self._initialize_weights(1, m2)
        self.t1 = self._initialize_thresholds(m1, 1)
        self.t2 = self._initialize_thresholds(m2, 1)
        self.t3 = self._initialize_thresholds(1, 1)

    def _initialize_weights(self, m, n, mu=0, sigma=1):
        size = (m, n)
        w = np.random.normal(mu, sigma, size=size)
        return w

    def _initialize_thresholds(self, m, n):
        size = (m, n)
        t = np.zeros(size)
        return t

    def feed_forward(self, inputs):
        output1 = np.tanh(self.w1 @ np.transpose(inputs) - self.t1)
        output2 = np.tanh(self.w2 @ output1 - self.t2)
        output3 = np.tanh(self.w3 @ output2 - self.t3)
        return output1, output2, output3

    def propagate_back(self, inputs, target, output1, output2, output3):
        error3 = (target - output3) * (1 - output3 ** 2)
        error2 = np.multiply(
            (np.transpose(self.w3) @ error3), (1 - output2 ** 2))
        error1 = np.multiply(
            (np.transpose(self.w2) @ error2), (1 - output1 ** 2))
        dw3 = -1 * self.learning_rate * (-1 * error3 * np.transpose(output2))
        dw2 = -1 * self.learning_rate * \
            np.multiply(-1 * error2, np.transpose(output1))
        dw1 = -1 * self.learning_rate * np.multiply(-1 * error1, inputs)
        dt3 = -1 * self.learning_rate * error3
        dt2 = -1 * self.learning_rate * error2
        dt1 = -1 * self.learning_rate * error1
        return dw1, dw2, dw3, dt1, dt2, dt3

    def _update(self):
        training = np.array(self.training.copy())
        np.random.shuffle(training)
        for pattern in training:
            inputs = pattern[:-1].reshape(1, 2)
            target = pattern[-1].reshape(1, 1)
            output1, output2, output3 = self.feed_forward(inputs)
            dw1, dw2, dw3, dt1, dt2, dt3 = self.propagate_back(
                inputs, target, output1, output2, output3)
            self.w1 += dw1
            self.w2 += dw2
            self.w3 += dw3
            self.t1 += dt1
            self.t2 += dt2
            self.t3 += dt3

    def _classification_error(self):
        validation = np.array(self.validation.copy())
        length = validation.shape[0]
        inputs = validation[:, :-1].reshape(length, 2)
        targets = validation[:, -1].reshape(length, 1)
        output3 = self.feed_forward(inputs)[-1]
        errors = np.sum(np.abs(np.sign(output3) - np.transpose(targets)))
        return 0.5 * errors / length

    def train(self):
        for epoch in range(self.epochs):
            self._update()
            error = self._classification_error()
            print(f"Epoch {epoch}: {error}")
            if error < 0.12:
                print(f"Converged after {epoch} iterations...")
                return True
        print("\nNo convergence...\n")
        return False

    def print(self):
        print(f"w1: {self.w1}")
        print(f"w2: {self.w2}")
        print(f"w3: {self.w3}")
        print(f"t1: {self.t1}")
        print(f"t2: {self.t2}")
        print(f"t3: {self.t3}")


def main():
    training = np.genfromtxt('training_set.csv', delimiter=',')
    validation = np.genfromtxt('validation_set.csv', delimiter=',')

    network = TwoLayerPerceptron(training, validation)
    network.train()


if __name__ == "__main__":
    main()
