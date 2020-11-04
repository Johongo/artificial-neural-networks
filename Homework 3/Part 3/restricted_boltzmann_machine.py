import numpy as np

# Patterns
PATTERNS = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
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

# Network parameters
N_PATTERNS = 14
N_VISIBLE = 9
N_HIDDEN = 8

# Training parameters
N_EPOCHS = 50
N_SAMPLES = 10000
N_TIMES = 100
N_TRANSIENT = 20


class Boltzmann:
    def __init__(self, n_visible=N_VISIBLE, n_hidden=N_HIDDEN):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weights = self.get_weights(n_hidden, n_visible)
        self.v_bias = self.get_thresholds(n_visible)
        self.h_bias = self.get_thresholds(n_hidden)
        self.learning_rate = 0.01

    # ------------------------
    # Initialization functions
    # ------------------------
    def get_weights(self, x, y):
        return 0.01 * np.random.randn(x, y)

    def get_thresholds(self, x):
        return np.zeros(x)

    # -----------------
    # Utility functions
    # -----------------
    def get_random_pattern(self, size=N_VISIBLE):
        return 2 * np.random.randint(2, size=size) - 1

    def get_index(self, output, patterns=PATTERNS):
        for index, pattern in enumerate(patterns):
            if (pattern == output).all():
                return index

    # -------------------
    # Boltzmann functions
    # -------------------
    def train(self, patterns=PATTERNS, n_patterns=N_PATTERNS, n_epochs=N_EPOCHS, n_times=N_TIMES):
        divergences = np.zeros(n_epochs)

        for index in range(n_epochs):

            # Choose random pattern
            pattern_index = np.random.randint(n_patterns)
            pattern = patterns[pattern_index]

            # Initialize neurons
            v0, h0 = pattern, self.update_hidden(pattern)

            # Iterate for contrastive divergence
            vk, hk = v0, h0
            for _ in range(n_times):
                vk = self.update_visible(hk)
                hk = self.update_hidden(vk)

            # Calculate local fields
            bh0 = self.weights @ v0 - self.h_bias
            bhk = self.weights @ vk - self.h_bias

            # Calculate changes
            dw = np.outer(np.tanh(bh0), v0) - np.outer(np.tanh(bhk), vk)
            dtv = v0 - vk
            dth = np.tanh(bh0) - np.tanh(bhk)

            # Update
            self.weights = self.weights + dw * self.learning_rate
            self.v_bias = self.v_bias - dtv * self.learning_rate
            self.h_bias = self.h_bias - dth * self.learning_rate

            # Kullback-Leibler divergence
            distribution = self.approximate_distribution()
            divergence = self.kullback_leibler(distribution)
            divergences[index] = divergence

        return divergences

    def run(self, pattern, n_transient=N_TRANSIENT):

        # Initialize neurons
        v0, h0 = pattern, self.update_hidden(pattern)

        # Iterate beyond transient
        vk, hk = v0, h0
        for _ in range(n_transient):
            vk = self.update_visible(hk)
            hk = self.update_hidden(vk)

        return vk

    def update_hidden(self, visible):
        hidden = np.zeros(self.n_hidden)
        for h in range(self.n_hidden):
            w = self.weights[h, :]
            b = np.dot(w, visible) - self.h_bias[h]
            hidden[h] = self.stochastic_update(b)
        return hidden

    def update_visible(self, hidden):
        visible = np.zeros(self.n_visible)
        for v in range(self.n_visible):
            w = self.weights[:, v]
            b = np.dot(w, hidden) - self.v_bias[v]
            visible[v] = self.stochastic_update(b)
        return visible

    def stochastic_update(self, local_field):
        p = self.sigmoid(local_field)
        r = np.random.rand()
        output = 1 if r < p else -1
        return output

    def sigmoid(self, b):
        return 1 / (1 + np.exp(-2 * b))

    def kullback_leibler(self, distribution, n_patterns=N_PATTERNS):
        result = 0
        p_data = 1 / n_patterns
        for pb in distribution:
            temp = p_data * np.log(p_data / pb)
            result += temp
        return result

    def approximate_distribution(self, n_patterns=N_PATTERNS, n_samples=N_SAMPLES):
        outputs = np.zeros(n_patterns)
        for _ in range(n_samples):
            pattern = self.get_random_pattern()
            output = self.run(pattern)
            index = self.get_index(output)
            if index is not None:
                outputs[index] += 1
        return outputs / n_samples


def main():
    boltzmann = Boltzmann()
    divergences = boltzmann.train()
    # plot divergences here


if __name__ == '__main__':
    main()
