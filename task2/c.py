import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Define the function inside the integral
def integrand(rho, beta, c):
    numerator = c**2 * rho**2
    denominator = beta**2 * (beta**2 - rho**2)
    return 2 * np.pi * rho * np.sqrt(1 + numerator / denominator)

# Monte Carlo Integration
def monte_carlo_integration(beta, c, N):
    rho_samples = np.random.uniform(0, beta, N)  # Uniform samples in [0, beta]
    integral_estimate = (beta / N) * np.sum(integrand(rho_samples, beta, c))
    return 2 * integral_estimate  # Multiply by 2 as in the given integral

# Exact surface area formula
def exact_surface_area(beta, c):
    e = 1 - beta**2 / c**2
    return 2 * np.pi * beta**2 * (1 + (c / beta) * np.arcsin(e))

# Define parameters
beta = 1
c = 1  # Given in the problem: 2β = c = 1
N_values = [10, 100, 1000, 10000, 100000]  # Different sampling sizes

# Compute errors for different N
errors = []
for N in N_values:
    mc_area = monte_carlo_integration(beta, c, N)
    exact_area = exact_surface_area(beta, c)
    error = abs(mc_area - exact_area) / exact_area  # Relative error
    errors.append(error)

# Plot error vs N
plt.figure(figsize=(8, 5))
plt.loglog(N_values, errors, marker='o', linestyle='-', label="Monte Carlo Error")
plt.xlabel("Number of Samples (N)")
plt.ylabel("Relative Error")
plt.title("Monte Carlo Integration Error vs. N")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()
plt.savefig("plot_c.png")