import numpy as np

PATTERNS = np.matrix([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [+1, -1, -1, +1, -1, -1, +1, -1, -1],
                      [-1, +1, -1, -1, +1, -1, -1, +1, -1],
                      [-1, -1, +1, -1, -1, +1, -1, -1, +1],
                      [+1, +1, -1, +1, +1, -1, +1, +1, -1],
                      [-1, +1, +1, -1, +1, +1, -1, +1, +1],
                      [+1, -1, +1, +1, -1, +1, +1, -1, +1],
                      [+1, +1, +1, +1, +1, +1, +1, +1, +1],
                      [+1, +1, +1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, +1, +1, +1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, +1, +1, +1],
                      [+1, +1, +1, +1, +1, +1, -1, -1, -1],
                      [-1, -1, -1, +1, +1, +1, +1, +1, +1],
                      [+1, +1, +1, -1, -1, -1, +1, +1, +1]])

N_PATTERNS = 14
N_VISIBLE = 9
N_HIDDEN = 2

N_EPOCHS = 1000
N_SAMPLES = 10000
N_TIMES = 100
N_TRANSIENT = 20


class Boltzmann:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weights = self.get_weights()
        self.v_bias = self.get_thresholds(True)
        self.h_bias = self.get_thresholds(False)
        self.learning_rate = 0.1

    def get_weights(self, mu = 0, sigma = 0.01):
        normal = np.random.randn(self.n_visible, self.n_hidden)
        return mu + sigma * normal

    def get_thresholds(self, visible = True):
        if visible:
            # TODO Do the fancy initialization
            return np.zeros(self.n_visible)
        return np.zeros(self.n_hidden)

    def get_pattern(self, size):
        return 2 * np.random.randint(2, size=size) - 1

    def get_pattern_index(self, output):
        for counter, pattern in enumerate(PATTERNS):
            if (pattern == output).all():
                return counter
        return None

    def sigmoid(self, b):
        return 1 / (1 + np.exp(-2 * b))

    def stochastic_update(self, local_field):
        p = self.sigmoid(local_field)
        r = np.random.rand()
        output = 1 if r < p else -1
        return output

    def kullback_leibler(self, pbs):
        result = 0
        p_data = 1 / N_PATTERNS
        for pb in pbs:
            temp = p_data * np.log(p_data / pb)
            result += temp
        return result


    def train(self):
        for _ in range(N_EPOCHS):
            # Choose random pattern
            p0 = PATTERNS[np.random.randint(N_PATTERNS)]

            # Set visible neurons
            v0 = p0
            bh0 = v0 * self.weights - self.h_bias

            # Update hidden neurons
            h0 = np.zeros((1, 2))
            for index in range(self.n_hidden):
                h0[0, index] = self.stochastic_update(bh0[0, index])

            # Iterate
            vk = v0
            hk = h0
            for _ in range(N_TIMES):
                bvk = self.weights @ np.transpose(hk) - self.v_bias
                for index in range(self.n_visible):
                    vk[0, index] = self.stochastic_update(bvk[0, index])
                bhk = vk * self.weights - self.h_bias
                for index in range(self.n_hidden):
                    hk[0, index] = self.stochastic_update(bhk[0, index])

            # Calculate changes
            dw = np.outer(v0, np.tanh(bh0)) - np.outer(vk, np.tanh(bhk))
            dtv = v0 - vk
            dth = np.tanh(bh0) - np.tanh(bhk)

            # Update
            self.weights = self.weights + dw * self.learning_rate
            self.v_bias = self.v_bias - dtv * self.learning_rate
            self.h_bias = self.h_bias - dth * self.learning_rate

            # Kullback-Leibler divergence
            pb = self.approximate_distribution()
            kld = self.kullback_leibler(pb)
            print(kld)

    def run(self, data):
        v0 = np.reshape(data, (1, self.n_visible))
        bh0 = v0 @ self.weights - self.h_bias
        
        # Update hidden neurons
        h0 = np.zeros((1, 2))
        for index in range(self.n_hidden):
            h0[0, index] = self.stochastic_update(bh0[0, index])

        # Iterate
        vk = v0
        hk = h0
        for _ in range(N_TRANSIENT):
            bvk = self.weights @ np.transpose(hk) - self.v_bias
            for index in range(self.n_visible):
                vk[0, index] = self.stochastic_update(bvk[0, index])
            bhk = vk @ self.weights - self.h_bias
            for index in range(self.n_hidden):
                hk[0, index] = self.stochastic_update(bhk[0, index])
        return vk

    def approximate_distribution(self):
        outputs = np.zeros(N_PATTERNS)
        for _ in range(N_SAMPLES):
            pattern = self.get_pattern(self.n_visible)
            output = self.run(pattern)
            index = self.get_pattern_index(output)
            if index is not None:
                outputs[index] += 1
        return outputs / np.sum(outputs)


def main():
    boltzmann = Boltzmann(N_VISIBLE, N_HIDDEN)
    boltzmann.train()


if __name__ == '__main__':
    main()
