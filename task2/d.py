import numpy as np
import matplotlib.pyplot as plt

# Define the function inside the integral
def integrand(rho, beta, c):
    numerator = c**2 * rho**2
    denominator = beta**2 * (beta**2 - rho**2)
    return 2 * np.pi * rho * np.sqrt(1 + numerator / denominator)

# Exact surface area function
def exact_surface_area(beta, c):
    e = 1 - beta**2 / c**2
    return 2 * np.pi * beta**2 * (1 + (c / beta) * np.arcsin(e))

# Monte Carlo with uniform sampling
def monte_carlo_uniform(beta, c, N):
    rho_samples = np.random.uniform(0, beta, N)
    integral_estimate = (beta / N) * np.sum(integrand(rho_samples, beta, c))
    return 2 * integral_estimate

# Monte Carlo with truncated exponential importance sampling (q1(x) = exp(-3x))
def monte_carlo_importance_exp(beta, c, N):
    rho_samples = []
    while len(rho_samples) < N:
        sample = -np.log(np.random.uniform(0, 1)) / 3  # Exponential sampling
        if sample <= beta:  # Only keep samples within [0, beta]
            rho_samples.append(sample)
    rho_samples = np.array(rho_samples)
    weights = np.exp(3 * rho_samples)  # 1/q1(x)
    integral_estimate = np.mean(integrand(rho_samples, beta, c) * weights)
    return 2 * integral_estimate

# Monte Carlo with sine-squared importance sampling (q2(x) = sin^2(5x))
def monte_carlo_importance_sin(beta, c, N):
    u_samples = np.random.uniform(0, 1, N)  # Uniform random numbers
    rho_samples = np.arcsin(np.sqrt(u_samples)) / 5  # Inverse transform of sin^2(5x)
    weights = 1 / (5 * np.sin(5 * rho_samples) ** 2)  # 1/q2(x)
    integral_estimate = np.mean(integrand(rho_samples, beta, c) * weights)
    return 2 * integral_estimate

# Define parameters
beta = 1
c = 1  # Given in the problem: 2β = c = 1
N_values = [10, 100, 1000, 10000, 100000]  # Different sampling sizes

# Compute errors for different sampling methods
errors_uniform = []
errors_exp = []
errors_sin = []
exact_area = exact_surface_area(beta, c)

for N in N_values:
    mc_uniform = monte_carlo_uniform(beta, c, N)
    mc_exp = monte_carlo_importance_exp(beta, c, N)
    mc_sin = monte_carlo_importance_sin(beta, c, N)

    errors_uniform.append(abs(mc_uniform - exact_area) / exact_area)
    errors_exp.append(abs(mc_exp - exact_area) / exact_area)
    errors_sin.append(abs(mc_sin - exact_area) / exact_area)

# Plot error comparison
plt.figure(figsize=(8, 5))
plt.loglog(N_values, errors_uniform, marker='o', linestyle='-', label="Uniform Sampling")
plt.loglog(N_values, errors_exp, marker='s', linestyle='-', label="Exponential Sampling")
plt.loglog(N_values, errors_sin, marker='^', linestyle='-', label="Sine-Squared Sampling")
plt.xlabel("Number of Samples (N)")
plt.ylabel("Relative Error")
plt.title("Monte Carlo Integration Error with Different Sampling Methods")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()
plt.savefig("plot_d.png")