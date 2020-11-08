import pickle

import numpy as np

# Parameters
n_bits = 160
n_pattern = 5


def draw_digit(pattern):
    rows, cols = 16, 10
    for i in range(rows):
        for j in range(cols):
            if pattern[cols * i + j] == 1:
                print('#', end='')
            else:
                print('-', end='')
        print()


def weights(x):
    weights = np.zeros((n_bits, n_bits))
    for i in range(n_bits):
        for j in range(n_bits):
            weights[i, j] = weight(x, i, j)
    return weights


def weight(x, i, j):
    if i == j:
        return 0
    sum = 0
    rows, cols = np.shape(x)
    for mu in range(cols):
        sum += x[i, mu] * x[j, mu]
    sum /= rows
    return sum


def sgn(x):
    return 1 if x >= 0 else -1


def asynchronous_update(distorted, weights, i_bit):
    sum = 0
    for j in range(n_bits):
        sum += distorted[j] * weights[i_bit, j]
    output = sgn(sum)
    return output


def check_converged(patterns, distorted):
    for i in range(patterns.shape[1]):
        if np.array_equal(patterns[:, i], distorted):
            return True
        if np.array_equal(-patterns[:, i], distorted):
            return True
    return False


def main():
    with open('patterns.pkl', 'rb') as f:
        patterns = pickle.load(f)

    with open('distorted.pkl', 'rb') as f:
        distorted = pickle.load(f)

    w = weights(patterns)
    dist = distorted[:, 2]

    print("Starting...\n")
    draw_digit(dist)

    converged = check_converged(patterns, dist)
    while not converged:
        for i_bit in range(n_bits):
            before = dist[i_bit]
            output = asynchronous_update(dist, w, i_bit)
            dist[i_bit] = output
            if before != output:
                converged = check_converged(patterns, dist)

    print("\nDone!\n")
    draw_digit(dist)


if __name__ == '__main__':
    main()
