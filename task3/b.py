import numpy as np
import matplotlib.pyplot as plt

# Given parameters
a = 4
b = 4

# Define the target probability density function
def p_t(t):
    return np.exp(-b * t) * np.cos(a * t)**2

# Define the exponential proposal function
def q_t(t):
    return np.exp(-2 * t)  # Exponential distribution with lambda=2

# Rejection Sampling using exponential proposal function
def rejection_sampling_exponential(N):
    samples = []
    count_accept = 0
    count_reject = 0
    
    while len(samples) < N:
        t_proposed = np.random.exponential(1/2)  # Sample from Exp(2)
        u = np.random.uniform(0, 1)  # Random number for acceptance
        
        # Find maximum M for p_t / q_t
        M = 1  # A reasonable upper bound
        
        if u < p_t(t_proposed) / (M * q_t(t_proposed)):
            samples.append(t_proposed)
            count_accept += 1
        else:
            count_reject += 1
    
    rejection_ratio = count_accept / (count_accept + count_reject)
    return np.array(samples), rejection_ratio

# Generate and plot samples for different N
Ns = [100, 1000, 10000]
plt.figure(figsize=(12, 4))

for i, N in enumerate(Ns):
    samples, rejection_ratio = rejection_sampling_exponential(N)
    plt.subplot(1, 3, i + 1)
    plt.hist(samples, bins=30, density=True, alpha=0.7, color='r', label='Exp Proposal')
    plt.title(f'N={N}, Rej. Ratio={rejection_ratio:.3f}')
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("plot_b.png")