import numpy as np

# Parameters
trials = 100000
n_bits = 120
n_patterns = [12, 24, 48, 70, 100, 120]
zero_diagonal = False


def generate_patterns(n_bits, n_pattern):
    size = (n_bits, n_pattern)
    p = np.random.randint(2, size=size)
    p_transform = 2 * p - 1
    return p_transform


def weight(x, i, j):
    if i == j and zero_diagonal:
        return 0
    sum = 0
    rows, cols = np.shape(x)
    for mu in range(cols):
        sum += x[i, mu] * x[j, mu]
    sum /= rows
    return sum


def sgn(x):
    return 1 if x >= 0 else -1


def asynchronous_update(x, i_pattern, i_bit):
    sum = 0
    for j in range(n_bits):
        sum += x[j, i_pattern] * weight(x, i_bit, j)
    output = sgn(sum)
    return output


def main():
    for n_pattern in n_patterns:
        n_error = 0
        for _ in range(trials):
            patterns = generate_patterns(n_bits, n_pattern)
            i_pattern = np.random.randint(n_pattern)
            i_bit = np.random.randint(n_bits)
            output = asynchronous_update(patterns, i_pattern, i_bit)
            n_error += 0 if patterns[i_bit, i_pattern] == output else 1
        p_error = n_error / trials
        print(f"{p_error:.4f}")


if __name__ == '__main__':
    main()
