import numpy as np
import matplotlib.pyplot as plt

def probabilistic_encryption_analysis(rate, curve):
    C1 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-1.npy")
    C2 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-2.npy")
    C3 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-3.npy")
    C4 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-4.npy")
    C5 = np.load(f"ciphertext/rate-{rate}-curve-{curve}-5.npy")
    stacked_arrays = np.stack((C1, C2, C3, C4, C5))
    std = np.std(stacked_arrays, axis=0)
    mean_std = np.mean(std)
    std_std = np.std(std)
    return mean_std, std_std

def plot_std(num_samples, sec224r1, sec256k1, secp256r1, sec384r1, sec521r1):
    plt.figure(figsize=(8, 6))

    # Plot each curve on the graph
    plt.plot(num_samples, sec224r1['std_std'], 'ro-', label='sec224r1')
    plt.plot(num_samples, sec256k1['std_std'], 'gs-', label='sec256k1')
    plt.plot(num_samples, secp256r1['std_std'], 'b^-', label='secp256r1')
    plt.plot(num_samples, sec384r1['std_std'], 'c*-', label='sec384r1')
    plt.plot(num_samples, sec521r1['std_std'], 'm+-', label='sec521r1')

    plt.plot(num_samples, sec224r1['mean_std'], 'ro--', label='sec224r1')
    plt.plot(num_samples, sec256k1['mean_std'], 'gs--', label='sec256k1')
    plt.plot(num_samples, secp256r1['mean_std'], 'b^--', label='secp256r1')
    plt.plot(num_samples, sec384r1['mean_std'], 'c*--', label='sec384r1')
    plt.plot(num_samples, sec521r1['mean_std'], 'm+--', label='sec521r1')

    # Adding labels and title
    plt.xlabel('Dropout Rate')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    rate = 0.01
    curve = "secp224r1"
    print(probabilistic_encryption_analysis(rate, curve))