import numpy as np
import matplotlib.pyplot as plt

def box_muller_transform(N, mu=0, sigma=1):
    """Generate N Gaussian distributed samples using the Box-Muller Transform."""
    U1 = np.random.uniform(0, 1, N)
    U2 = np.random.uniform(0, 1, N)
    
    Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    
    samples = np.concatenate((Z0, Z1))[:N]  # Ensuring we return exactly N samples
    return mu + sigma * samples

# Sampling sizes
N_values = [10, 100, 1000, 10000, 100000]

plt.figure(figsize=(12, 8))
for i, N in enumerate(N_values):
    samples = box_muller_transform(N)
    plt.subplot(3, 2, i + 1)
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='b')
    plt.title(f'Box–Muller Transform Histogram (N={N})')
    plt.xlabel('Value')
    plt.ylabel('Density')

plt.tight_layout()
plt.show()
plt.savefig("plot_e.png")