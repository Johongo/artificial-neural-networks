import numpy as np

# Parameters
n_bits = 200
n_pattern = 45
n_trials = 100
n_updates = 200000


def generate_patterns(n_bits, n_pattern):
    size = (n_bits, n_pattern)
    p = np.random.randint(2, size=size)
    p_transform = 2 * p - 1
    return p_transform


def weights(x):
    size = (n_bits, n_bits)
    weights = np.zeros(size)
    for i in range(n_bits):
        for j in range(n_bits):
            weights[i, j] = weight(x, i, j)
    return weights


def weight(x, i, j):
    if i == j:
        return 0
    return np.dot(x[i, :], x[j, :]) / n_bits


def sigmoid(b, beta=2):
    return 1 / (1 + np.exp(-2 * b * beta))


def stochastic_asynchronous_update(state, weights, i_bit):
    sum = np.dot(state, weights[i_bit, :])
    p = sigmoid(sum)
    r = np.random.rand()
    output = 1 if r < p else -1
    return output


def main():
    m1_avg = 0
    for _ in range(n_trials):
        m1 = 0
        patterns = generate_patterns(n_bits, n_pattern)
        w = weights(patterns)
        first = patterns[:, 0]
        state = first.copy()
        for _ in range(n_updates):
            i_bit = np.random.randint(n_bits)
            output = stochastic_asynchronous_update(state, w, i_bit)
            state[i_bit] = output
            m1 += np.dot(state, first) / n_bits
        m1_avg += m1 / n_updates
    m1_avg /= n_trials
    print(f"\nDone: {m1_avg:.4f}")


if __name__ == '__main__':
    main()
