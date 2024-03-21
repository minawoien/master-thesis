import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

def probabilistic_encryption_analysis(rate: float, curve: str) -> Tuple[float, float]:
    """Calculates the mean and standard deviation of five ciphertexts given a dropout rate and curve.
    Used to calculate the mean and standard deviation of the ciphertexts to compare the probabilistic encryption.
    
    Args:
        rate: Dropout rate.
        curve: Elliptic curve.
    
    Returns:
        mean_std: Mean of the standard deviation of the ciphertexts.
        std_std: Standard deviation of the standard deviation of the ciphertexts.
    """
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

def plot_std(dropout_rates: List[float], sec224r1: Tuple[List[float], List[float]], sec256k1: Tuple[List[float], List[float]], secp256r1: Tuple[List[float], List[float]], sec384r1: Tuple[List[float], List[float]], sec521r1: Tuple[List[float], List[float]]):
    """Plots the mean and standard deviation of five ciphertexts given a dropout rate and curve.
    
    Args:
        dropout_rates: Dropout rate.
        sec224r1: Tuple of mean and standard deviation of the standard deviation of the ciphertexts for secp224r1.
        sec256k1: Tuple of mean and standard deviation of the standard deviation of the ciphertexts for secp256k1.
        secp256r1: Tuple of mean and standard deviation of the standard deviation of the ciphertexts for secp256r1.
        sec384r1: Tuple of mean and standard deviation of the standard deviation of the ciphertexts for sec384r1.
        sec521r1: Tuple of mean and standard deviation of the standard deviation of the ciphertexts for sec521r1.
    """
    plt.figure(figsize=(8, 6))

    # Plot each curve on the graph
    plt.plot(dropout_rates, sec224r1[0], 'ro-', label='sec224r1')
    plt.plot(dropout_rates, sec256k1[0], 'gs-', label='sec256k1')
    plt.plot(dropout_rates, secp256r1[0], 'b^-', label='secp256r1')
    plt.plot(dropout_rates, sec384r1[0], 'c*-', label='sec384r1')
    plt.plot(dropout_rates, sec521r1[0], 'm+-', label='sec521r1')

    plt.plot(dropout_rates, sec224r1[1], 'ro--', label='sec224r1')
    plt.plot(dropout_rates, sec256k1[1], 'gs--', label='sec256k1')
    plt.plot(dropout_rates, secp256r1[1], 'b^--', label='secp256r1')
    plt.plot(dropout_rates, sec384r1[1], 'c*--', label='sec384r1')
    plt.plot(dropout_rates, sec521r1[1], 'm+--', label='sec521r1')

    # Adding labels and title
    plt.xlabel('Dropout Rate')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    rate = 0.01
    curve = "secp224r1"
    print(probabilistic_encryption_analysis(rate, curve))